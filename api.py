from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import json
import io
import re
import os
import csv
import tempfile
import pickle
import threading
import time
import numpy as np
from typing import Optional, List, Dict, Any
from pydantic import BaseModel

from chatbot_model import VnexAIChatbot
from transformer_model import TransformerChatbot
from chatbot_tokenizer import ChatbotTokenizer

app = FastAPI(title="VnexAI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Global state ──────────────────────────────────────────────────────────────
state: Dict[str, Any] = {
    "model": None,
    "tokenizer": None,
    "training_data": None,
    "is_trained": False,
    "chat_history": [],
    "model_bytes": None,
    "tokenizer_bytes": None,
    "gguf_bytes": None,
    "training_data_profile": None,
    "model_config": None,
    "model_type": None,
}

training_state: Dict[str, Any] = {
    "is_training": False,
    "progress": 0.0,
    "current_epoch": 0,
    "total_epochs": 0,
    "current_batch": 0,
    "total_batches": 0,
    "current_loss": 0.0,
    "avg_loss": 0.0,
    "losses": [],
    "status": "idle",
    "error": None,
    "gpu_available": False,
}
training_lock = threading.Lock()


# ─── Helpers ───────────────────────────────────────────────────────────────────
def parse_code_parquet(df, max_pairs=2000):
    col_map = {c.lower(): c for c in df.columns}
    content_col = col_map.get('content')
    lang_col = col_map.get('lang')
    if not content_col:
        return []
    if lang_col:
        df = df[df[lang_col].str.lower().isin(['python', 'py'])].copy()
    data = []
    pattern = re.compile(
        r'((?:async\s+)?def\s+\w+\s*\([^)]*\)(?:\s*->[^\n:]+)?:)'
        r'[ \t]*\n[ \t]+'
        r'(?:\"\"\"(.*?)\"\"\"|\'\'\'(.*?)\'\'\')'
        r'(.*?)(?=\n[ \t]*(?:async\s+)?def |\Z)',
        re.DOTALL
    )
    for _, row in df.iterrows():
        if len(data) >= max_pairs:
            break
        code = str(row[content_col])
        for m in pattern.finditer(code):
            sig = m.group(1).strip()
            doc = (m.group(2) or m.group(3) or '').strip()
            body = m.group(4).strip()
            if doc and len(doc) > 5 and len(body) > 5 and len(body) < 3000:
                data.append({'user': f'Write a Python function that: {doc}', 'bot': f'{sig}\n    """{doc}"""\n    {body}'})
                if len(data) >= max_pairs:
                    break
    if len(data) < 10:
        for _, row in df.iterrows():
            if len(data) >= max_pairs:
                break
            code = str(row[content_col])
            lines = [l for l in code.split('\n') if l.strip()]
            if len(lines) >= 8:
                mid = max(2, len(lines) // 3)
                user_msg = '\n'.join(lines[:mid])
                bot_msg = '\n'.join(lines[mid:])
                if len(user_msg) > 20 and len(bot_msg) > 20:
                    data.append({'user': user_msg, 'bot': bot_msg})
    return data


def parse_uploaded_file(content: bytes, filename: str) -> List[Dict]:
    data = []
    if filename.endswith('.json') or filename.endswith('.jsonl'):
        text = content.decode('utf-8')
        try:
            raw = json.loads(text)
        except json.JSONDecodeError:
            raw = []
            for line in text.strip().split('\n'):
                if line.strip():
                    try:
                        raw.append(json.loads(line))
                    except:
                        continue
        if isinstance(raw, list) and raw:
            first = raw[0]
            if 'original_src' in first and 'changed_src' in first:
                for item in raw:
                    lang = item.get('language', 'code')
                    status = item.get('original_status', 'error')
                    orig = item.get('original_src', '')
                    fixed = item.get('changed_src', '')
                    if orig and fixed:
                        data.append({'user': f'Fix this {lang} code with {status}: {orig}', 'bot': fixed})
            elif 'user' in first and 'bot' in first:
                data = raw
            else:
                raise ValueError("Unknown JSON format. Use {user, bot} or code debugging format.")
        else:
            data = raw if isinstance(raw, list) else []

    elif filename.endswith('.txt'):
        text = content.decode('utf-8')
        entries = re.split(r'\n\d+:\{', text)
        parsed_json = False
        for i, entry in enumerate(entries):
            if not entry.strip():
                continue
            if i > 0:
                entry = '{' + entry
            else:
                if not entry.strip().startswith('{'):
                    continue
            try:
                last_brace = entry.rfind('}')
                if last_brace != -1:
                    obj = json.loads(entry[:last_brace + 1])
                    if 'original_src' in obj and 'changed_src' in obj:
                        lang = obj.get('language', 'code')
                        status = obj.get('original_status', 'error')
                        orig = obj.get('original_src', '')
                        fixed = obj.get('changed_src', '')
                        if orig and fixed:
                            data.append({'user': f'Fix this {lang} code with {status}: {orig}', 'bot': fixed})
                            parsed_json = True
            except:
                continue
        if not parsed_json:
            for line in text.split('\n'):
                for sep in ['|', '→', '\t', ' - ']:
                    if sep in line:
                        parts = line.split(sep, 1)
                        if len(parts) == 2:
                            u = parts[0].strip()
                            b = parts[1].strip()
                            for lbl in ['user:', 'question:', 'input:', 'q:', 'human:']:
                                if u.lower().startswith(lbl):
                                    u = u[len(lbl):].strip()
                                    break
                            for lbl in ['bot:', 'answer:', 'output:', 'a:', 'assistant:', 'response:']:
                                if b.lower().startswith(lbl):
                                    b = b[len(lbl):].strip()
                                    break
                            data.append({'user': u, 'bot': b})
                            break

    elif filename.endswith('.csv') or filename.endswith('.tsv'):
        text = content.decode('utf-8')
        delimiter = '\t' if filename.endswith('.tsv') else ','
        reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
        for row in reader:
            u = b = None
            for key in row.keys():
                kl = key.lower().strip()
                if kl in ['user', 'question', 'input', 'q', 'prompt', 'human'] and row[key]:
                    u = row[key].strip()
                    break
            for key in row.keys():
                kl = key.lower().strip()
                if kl in ['bot', 'answer', 'output', 'a', 'response', 'assistant', 'reply'] and row[key]:
                    b = row[key].strip()
                    break
            if u is None or b is None:
                cols = list(row.values())
                if len(cols) >= 2:
                    u = cols[0].strip() if cols[0] else None
                    b = cols[1].strip() if cols[1] else None
            if u and b:
                data.append({'user': u, 'bot': b})

    elif filename.endswith('.parquet'):
        import pandas as pd
        df = pd.read_parquet(io.BytesIO(content))
        col_lower_set = {c.lower() for c in df.columns}
        if 'content' in col_lower_set:
            data = parse_code_parquet(df, max_pairs=2000)
        else:
            user_cols = ['user', 'question', 'input', 'human', 'original_src', 'prompt', 'query', 'instruction', 'text']
            bot_cols = ['bot', 'answer', 'output', 'assistant', 'changed_src', 'response', 'reply', 'completion', 'target']
            user_col = next((c for c in df.columns if c.lower() in user_cols), None)
            bot_col = next((c for c in df.columns if c.lower() in bot_cols), None)
            if user_col and bot_col:
                for _, row in df.iterrows():
                    u = str(row[user_col]).strip()
                    b = str(row[bot_col]).strip()
                    if u and b:
                        data.append({'user': u, 'bot': b})
            elif len(df.columns) == 2:
                for _, row in df.iterrows():
                    u = str(row.iloc[0]).strip()
                    b = str(row.iloc[1]).strip()
                    if u and b:
                        data.append({'user': u, 'bot': b})
            else:
                raise ValueError(f"Cannot auto-detect columns. Found: {list(df.columns)}")
    return data


# ─── Training thread ───────────────────────────────────────────────────────────
def run_training_thread(model, tokenizer, data, config):
    epochs = config["epochs"]
    batch_size = config["batch_size"]
    shuffle = config.get("shuffle_data", True)
    use_sft = config.get("use_sft", False)
    max_len = int(model.max_length if hasattr(model, "max_length") else model.max_seq_len)

    with training_lock:
        training_state["status"] = "training"
        training_state["is_training"] = True
        training_state["progress"] = 0.0
        training_state["losses"] = []
        training_state["current_epoch"] = 0
        training_state["total_epochs"] = epochs
        training_state["error"] = None
        training_state["gpu_available"] = bool(getattr(model, "gpu_available", False))

    try:
        input_seqs_all = []
        target_seqs_all = []
        sft_starts_all = []

        if hasattr(model, "clear_update_coverage"):
            model.clear_update_coverage()

        for conv in data:
            if getattr(model, 'decoder_only', False):
                user_seq = tokenizer.encode(conv['user'], add_special_tokens=True)
                bot_seq = tokenizer.encode(conv['bot'], add_special_tokens=True)
                full_seq = np.array(user_seq + bot_seq, dtype=np.int32)[:max_len]
                if len(full_seq) < 2:
                    continue
                input_seqs_all.append(full_seq[:-1])
                target_seqs_all.append(full_seq[1:])
                sft_start = min(len(user_seq) - 1, len(full_seq) - 2)
                sft_starts_all.append(max(0, sft_start))
            else:
                user_seq = np.array(tokenizer.encode(conv['user'], add_special_tokens=True), dtype=np.int32)[:max_len]
                bot_seq = np.array(tokenizer.encode(conv['bot'], add_special_tokens=True), dtype=np.int32)[:max_len]
                if len(user_seq) > 1 and len(bot_seq) > 1:
                    input_seqs_all.append(user_seq)
                    target_seqs_all.append(bot_seq)
                    sft_starts_all.append(0)

        total_samples = len(input_seqs_all)
        num_batches = max(1, (total_samples + batch_size - 1) // batch_size)

        with training_lock:
            training_state["total_batches"] = num_batches

        losses = []
        for epoch in range(epochs):
            epoch_losses = []
            if shuffle:
                indices = np.random.permutation(total_samples)
                inp = [input_seqs_all[i] for i in indices]
                tgt = [target_seqs_all[i] for i in indices]
                sft = [sft_starts_all[i] for i in indices]
            else:
                inp, tgt, sft = input_seqs_all, target_seqs_all, sft_starts_all

            batch_num = 0
            for i in range(0, len(inp), batch_size):
                bi = inp[i:i+batch_size]
                bt = tgt[i:i+batch_size]
                bs = sft[i:i+batch_size] if use_sft and getattr(model, "decoder_only", False) else None
                if not bi:
                    continue
                batch_num += 1
                if bs is not None:
                    loss = model.train_batch(bi, bt, sft_loss_starts=bs)
                else:
                    loss = model.train_batch(bi, bt)
                epoch_losses.append(float(loss))
                with training_lock:
                    training_state["current_batch"] = batch_num
                    training_state["current_loss"] = float(loss)

            avg_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
            losses.append(avg_loss)

            if hasattr(model, 'step_lr'):
                model.step_lr(total_epochs=epochs)

            current_lr = float(getattr(model, 'learning_rate', 0.0))
            with training_lock:
                training_state["current_epoch"] = epoch + 1
                training_state["progress"] = (epoch + 1) / epochs
                training_state["avg_loss"] = avg_loss
                training_state["losses"] = losses[:]

        model.training_history['loss'] = losses
        state["is_trained"] = True

        with training_lock:
            training_state["is_training"] = False
            training_state["status"] = "done"
            training_state["progress"] = 1.0

    except Exception as e:
        with training_lock:
            training_state["is_training"] = False
            training_state["status"] = "error"
            training_state["error"] = str(e)


# ─── Routes: status ────────────────────────────────────────────────────────────
@app.get("/api/status")
async def get_status():
    with training_lock:
        ts = dict(training_state)
    return {
        "has_training_data": state["training_data"] is not None,
        "training_data_count": len(state["training_data"]) if state["training_data"] else 0,
        "has_tokenizer": state["tokenizer"] is not None,
        "tokenizer_vocab_size": state["tokenizer"].vocab_size if state["tokenizer"] else 0,
        "has_model": state["model"] is not None,
        "is_trained": state["is_trained"],
        "model_config": state["model_config"],
        "model_type": state["model_type"],
        "training_data_profile": state["training_data_profile"],
        "chat_history": state["chat_history"],
        "training": ts,
    }


# ─── Routes: data ──────────────────────────────────────────────────────────────
@app.post("/api/data/upload-file")
async def upload_file(file: UploadFile = File(...)):
    content = await file.read()
    try:
        data = parse_uploaded_file(content, file.filename)
        if not data:
            raise HTTPException(status_code=400, detail="No valid training pairs found in file.")
        state["training_data"] = data
        state["training_data_profile"] = None
        return {"count": len(data), "preview": data[:3]}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/data/upload-text")
async def upload_text(request: Request):
    body = await request.json()
    pairs = body.get("pairs", [])
    if not pairs:
        raise HTTPException(status_code=400, detail="No pairs provided.")
    state["training_data"] = pairs
    state["training_data_profile"] = None
    return {"count": len(pairs), "preview": pairs[:3]}


@app.post("/api/data/load-builtin")
async def load_builtin(request: Request):
    body = await request.json()
    kind = body.get("type", "assistant")
    if kind == "assistant":
        target_mb = body.get("target_mb", 90)
        from builtin_assistant_dataset import build_builtin_assistant_dataset, dataset_json_size_bytes
        samples = build_builtin_assistant_dataset(target_bytes=int(target_mb * 1024 * 1024))
        state["training_data"] = samples
        state["training_data_profile"] = "assistant_pack"
        mb = dataset_json_size_bytes(samples) / (1024 * 1024)
        return {"count": len(samples), "mb": round(mb, 1), "profile": "assistant_pack"}
    elif kind == "sft":
        rows = body.get("rows", 25000)
        from builtin_sft_dataset import build_builtin_sft_dataset, dataset_json_size_bytes
        samples = build_builtin_sft_dataset(target_rows=int(rows))
        state["training_data"] = samples
        state["training_data_profile"] = "sft_pack"
        mb = dataset_json_size_bytes(samples) / (1024 * 1024)
        return {"count": len(samples), "mb": round(mb, 1), "profile": "sft_pack"}
    else:
        raise HTTPException(status_code=400, detail="Unknown type")


@app.get("/api/data/info")
async def data_info():
    if not state["training_data"]:
        return {"count": 0, "preview": [], "profile": None}
    return {
        "count": len(state["training_data"]),
        "preview": state["training_data"][:5],
        "profile": state["training_data_profile"],
    }


# ─── Routes: model ─────────────────────────────────────────────────────────────
@app.post("/api/model/estimate-vocab")
async def estimate_vocab(request: Request):
    body = await request.json()
    max_vocab = body.get("max_vocab_size", 12000)
    if not state["training_data"]:
        raise HTTPException(status_code=400, detail="No training data loaded.")
    tok = ChatbotTokenizer(max_vocab_size=max_vocab)
    sample = state["training_data"][:8000]
    unique: set = set()
    for conv in sample:
        unique.update(tok.tokenize(conv.get("user", "")))
        unique.update(tok.tokenize(conv.get("bot", "")))
    return {"unique_words": len(unique), "sampled_pairs": len(sample)}


@app.post("/api/model/build-vocab")
async def build_vocab(request: Request):
    body = await request.json()
    max_vocab = body.get("max_vocab_size", 12000)
    pad_vocab = body.get("pad_vocab", True)
    pad_target = body.get("pad_target", 6000)
    if not state["training_data"]:
        raise HTTPException(status_code=400, detail="No training data loaded.")
    all_texts = []
    for conv in state["training_data"]:
        all_texts.append(conv["user"])
        all_texts.append(conv["bot"])
    tokenizer = ChatbotTokenizer(max_vocab_size=max_vocab)
    pad_list = None
    pad_until = 0
    if pad_vocab:
        from english_vocab_padding import get_padding_word_list
        pad_list = get_padding_word_list()
        pad_until = int(min(pad_target, max_vocab))
    tokenizer.build_vocabulary(all_texts, pad_with_words=pad_list if pad_vocab else None, pad_until=pad_until if pad_vocab else 0)
    state["tokenizer"] = tokenizer
    return tokenizer.get_vocab_info()


@app.post("/api/model/create")
async def create_model(request: Request):
    body = await request.json()
    if not state["tokenizer"]:
        raise HTTPException(status_code=400, detail="Build vocabulary first.")
    vocab_size = state["tokenizer"].vocab_size
    model_type = body.get("model_type", "transformer_decoder")

    try:
        if model_type == "rnn":
            model = VnexAIChatbot(
                vocab_size=vocab_size,
                embedding_dim=body.get("embedding_dim", 256),
                hidden_dim=body.get("hidden_dim", 256),
                max_length=body.get("max_length", 50),
                learning_rate=body.get("learning_rate", 0.05),
            )
            state["model_type"] = "RNN"
            total_params = (
                model.embedding.size + model.Wxh_enc.size + model.Whh_enc.size + model.bh_enc.size +
                model.Wxh_dec.size + model.Whh_dec.size + model.bh_dec.size + model.Why.size + model.by.size
            )
        else:
            embed_dim = body.get("embedding_dim", 256)
            num_heads = body.get("num_heads", 8)
            if embed_dim % num_heads != 0:
                raise HTTPException(status_code=400, detail=f"embedding_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")
            decoder_only = model_type == "transformer_decoder"
            model = TransformerChatbot(
                vocab_size=vocab_size,
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_layers=body.get("num_layers", 4),
                ff_dim=body.get("ff_dim", 1024),
                max_seq_len=body.get("max_length", 128),
                learning_rate=body.get("learning_rate", 0.003),
                optimizer=body.get("optimizer", "adam"),
                weight_decay=body.get("weight_decay", 0.01),
                scheduler=body.get("scheduler", "warmup_cosine"),
                warmup_epochs=body.get("warmup_epochs", 5),
                dropout_rate=body.get("dropout_rate", 0.1),
                grad_clip=body.get("grad_clip", 5.0),
                decoder_only=decoder_only,
            )
            state["model_type"] = "TransformerDecoder" if decoder_only else "TransformerEncoderDecoder"
            total_params = model._count_parameters()

        state["model"] = model
        state["is_trained"] = False
        state["model_config"] = {**body, "vocab_size": vocab_size, "model_type": model_type}

        gpu_info = model.get_device_info() if hasattr(model, 'get_device_info') else (
            "GPU: Enabled (CuPy)" if model.gpu_available else "CPU Only (NumPy)"
        )
        return {
            "total_params": total_params,
            "total_params_m": round(total_params / 1e6, 2),
            "gpu_available": bool(model.gpu_available),
            "gpu_info": gpu_info,
            "model_type": state["model_type"],
            "vocab_size": vocab_size,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── Routes: training ──────────────────────────────────────────────────────────
@app.post("/api/training/start")
async def start_training(request: Request):
    body = await request.json()
    if not state["model"]:
        raise HTTPException(status_code=400, detail="No model created.")
    if not state["tokenizer"]:
        raise HTTPException(status_code=400, detail="No tokenizer built.")
    if not state["training_data"]:
        raise HTTPException(status_code=400, detail="No training data loaded.")
    with training_lock:
        if training_state["is_training"]:
            raise HTTPException(status_code=409, detail="Training already in progress.")
    t = threading.Thread(
        target=run_training_thread,
        args=(state["model"], state["tokenizer"], state["training_data"], body),
        daemon=True,
    )
    t.start()
    return {"message": "Training started"}


@app.get("/api/training/stream")
async def training_stream():
    async def event_generator():
        last_epoch = -1
        consecutive_idle = 0
        while True:
            with training_lock:
                snap = dict(training_state)
                snap["losses"] = list(training_state["losses"])
            yield f"data: {json.dumps(snap)}\n\n"
            if snap["status"] in ("done", "error"):
                break
            if snap["status"] == "idle" and not snap["is_training"]:
                consecutive_idle += 1
                if consecutive_idle > 3:
                    break
            else:
                consecutive_idle = 0
            await asyncio.sleep(0.4)
    return StreamingResponse(event_generator(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/training/status")
async def training_status():
    with training_lock:
        snap = dict(training_state)
        snap["losses"] = list(training_state["losses"])
    return snap


# ─── Routes: chat ──────────────────────────────────────────────────────────────
@app.post("/api/chat/send")
async def chat_send(request: Request):
    body = await request.json()
    message = body.get("message", "").strip()
    temperature = float(body.get("temperature", 0.8))
    if not message:
        raise HTTPException(status_code=400, detail="Empty message.")
    model = state["model"]
    tokenizer = state["tokenizer"]
    if not model or not tokenizer:
        raise HTTPException(status_code=400, detail="Model not ready.")
    if not state["is_trained"]:
        raise HTTPException(status_code=400, detail="Model not trained yet.")

    max_retries = 3
    response_text = ""
    temp = temperature
    for attempt in range(max_retries):
        try:
            if hasattr(model, 'generate_response'):
                input_seq = np.array(tokenizer.encode(message, add_special_tokens=True))
                response_indices = model.generate_response(input_seq, temperature=temp)
                response_text = tokenizer.decode(response_indices.tolist())
            else:
                response_text = model.generate(message, tokenizer, temperature=temp)
        except Exception as e:
            response_text = f"[Error: {str(e)}]"
            break
        if ',,,' not in response_text and '...' not in response_text:
            break
        temp = min(temp + 0.3, 2.0)

    state["chat_history"].append({"role": "user", "content": message})
    state["chat_history"].append({"role": "bot", "content": response_text})
    return {"response": response_text, "history": state["chat_history"]}


@app.post("/api/chat/clear")
async def chat_clear():
    state["chat_history"] = []
    return {"ok": True}


# ─── Routes: export ────────────────────────────────────────────────────────────
@app.post("/api/export/prepare-model")
async def prepare_model():
    model = state["model"]
    if not model:
        raise HTTPException(status_code=400, detail="No model.")
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.bin') as tmp:
            tmp_path = tmp.name
        if hasattr(model, 'save_model'):
            model.save_model(tmp_path)
        else:
            model.save(tmp_path)
        with open(tmp_path, 'rb') as f:
            state["model_bytes"] = f.read()
        os.unlink(tmp_path)
        return {"size_bytes": len(state["model_bytes"]), "ready": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/export/download-model")
async def download_model(name: str = "vnexai_chatbot"):
    if not state["model_bytes"]:
        raise HTTPException(status_code=404, detail="Model not prepared. Call prepare-model first.")
    return StreamingResponse(io.BytesIO(state["model_bytes"]),
                             media_type="application/octet-stream",
                             headers={"Content-Disposition": f'attachment; filename="{name}.bin"'})


@app.post("/api/export/prepare-tokenizer")
async def prepare_tokenizer():
    tokenizer = state["tokenizer"]
    if not tokenizer:
        raise HTTPException(status_code=400, detail="No tokenizer.")
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.bin') as tmp:
            tmp_path = tmp.name
        tokenizer.save(tmp_path)
        with open(tmp_path, 'rb') as f:
            state["tokenizer_bytes"] = f.read()
        os.unlink(tmp_path)
        return {"size_bytes": len(state["tokenizer_bytes"]), "ready": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/export/download-tokenizer")
async def download_tokenizer(name: str = "vnexai_chatbot"):
    if not state["tokenizer_bytes"]:
        raise HTTPException(status_code=404, detail="Tokenizer not prepared.")
    return StreamingResponse(io.BytesIO(state["tokenizer_bytes"]),
                             media_type="application/octet-stream",
                             headers={"Content-Disposition": f'attachment; filename="{name}_tokenizer.bin"'})


@app.post("/api/export/prepare-gguf")
async def prepare_gguf(request: Request):
    body = await request.json()
    model_name = body.get("model_name", "vnexai_chatbot")
    model = state["model"]
    tokenizer = state["tokenizer"]
    if not model:
        raise HTTPException(status_code=400, detail="No model.")
    is_rnn = hasattr(model, 'hidden_dim')
    if not is_rnn and not getattr(model, 'decoder_only', False):
        raise HTTPException(status_code=400, detail="For reliable GGUF export, use Decoder-Only Transformer.")
    try:
        from gguf_writer import export_rnn_to_gguf, export_transformer_to_gguf
        if is_rnn:
            gguf_bytes = export_rnn_to_gguf(model, model_name)
        else:
            if not tokenizer:
                raise HTTPException(status_code=400, detail="No tokenizer.")
            gguf_bytes = export_transformer_to_gguf(model, tokenizer, model_name)
        state["gguf_bytes"] = gguf_bytes
        return {"size_bytes": len(gguf_bytes), "size_mb": round(len(gguf_bytes) / (1024 * 1024), 1), "ready": True}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/export/download-gguf")
async def download_gguf(name: str = "vnexai_chatbot"):
    if not state["gguf_bytes"]:
        raise HTTPException(status_code=404, detail="GGUF not prepared.")
    return StreamingResponse(io.BytesIO(state["gguf_bytes"]),
                             media_type="application/octet-stream",
                             headers={"Content-Disposition": f'attachment; filename="{name}.gguf"'})


@app.post("/api/export/modelfile")
async def generate_modelfile(request: Request):
    body = await request.json()
    model = state["model"]
    if not model:
        raise HTTPException(status_code=400, detail="No model.")
    model_name = body.get("model_name", "vnexai_chatbot")
    ollama_username = body.get("ollama_username", "yourusername")
    system_prompt = body.get("system_prompt", f"You are {model_name}, a custom AI assistant.")
    num_ctx = body.get("num_ctx", 512)
    temperature_val = body.get("temperature", 0.8)
    extra_stop = body.get("extra_stop", "")
    is_rnn = hasattr(model, 'hidden_dim')
    stop_lines = 'PARAMETER stop "<END>"'
    if extra_stop.strip():
        stop_lines += f'\nPARAMETER stop "{extra_stop.strip()}"'
    arch_note = (
        f"# Architecture : {'RNN encoder-decoder' if is_rnn else 'Transformer'}\n"
        f"# Trained with : VnexAI\n"
        f"# Vocab size   : {model.vocab_size}\n"
    )
    modelfile_text = f"""FROM ./{model_name}.gguf

{arch_note}
SYSTEM \"\"\"{system_prompt}\"\"\"

PARAMETER temperature {temperature_val}
PARAMETER num_ctx {num_ctx}
{stop_lines}

TEMPLATE \"\"\"{{{{ .Prompt }}}}\"\"\"
"""
    return {"modelfile": modelfile_text, "full_tag": f"{ollama_username}/{model_name}"}


@app.get("/api/export/model-info")
async def model_info():
    model = state["model"]
    tokenizer = state["tokenizer"]
    if not model:
        return {"has_model": False}
    is_rnn = hasattr(model, 'hidden_dim')
    info: Dict[str, Any] = {
        "has_model": True,
        "is_rnn": is_rnn,
        "is_trained": state["is_trained"],
        "model_type": state["model_type"],
        "vocab_size": model.vocab_size,
    }
    if is_rnn:
        info.update({
            "embedding_dim": model.embedding_dim,
            "hidden_dim": model.hidden_dim,
            "max_length": model.max_length,
            "learning_rate": model.learning_rate,
        })
    else:
        info.update({
            "embedding_dim": model.embed_dim,
            "num_heads": model.num_heads,
            "num_layers": model.num_layers,
            "ff_dim": model.ff_dim,
            "max_seq_len": model.max_seq_len,
            "learning_rate": model.learning_rate,
            "decoder_only": getattr(model, 'decoder_only', False),
        })
    if state["is_trained"] and model.training_history.get('loss'):
        info["final_loss"] = model.training_history['loss'][-1]
        info["loss_history"] = model.training_history['loss']
    return info


# ─── Serve frontend ────────────────────────────────────────────────────────────
DIST_DIR = os.path.join(os.path.dirname(__file__), "frontend", "dist")
if os.path.exists(DIST_DIR):
    app.mount("/", StaticFiles(directory=DIST_DIR, html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
