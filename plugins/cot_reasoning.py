"""cot_reasoning plugin -- single-file distributable.

Drop ONLY this file into plugins/. On startup it generates its own companion
files on disk:
  - cot_reasoning_dataset.py  (CoT training pairs, importable by other scripts)
  - reasoning_handler.py      (standalone inference harness, bundled with
                                CoT-mode exports so the model actually
                                reasons-then-answers outside this panel too)

Decoder-only requirement: reasoning (thinking before answering, in a single
continuous generated stream) is only meaningful for Decoder-Only Transformer
models. Encoder-decoder (RNN) models encode user/bot as separate sequences,
so this plugin refuses to enable reasoning mode, chat-with-reasoning, or
CoT export for anything that isn't decoder_only=True.

"Second-questions itself": generate_with_reasoning() does two passes --
pass 1 force-opens a <think> block and produces a draft reasoning + answer,
pass 2 feeds that draft back in and asks the model to re-examine it before
finalizing. This is a real, working self-verification mechanism. It is NOT
the same thing as RL-trained reasoning (what produces o1/R1-style models) --
it's a lightweight, honest approximation that works at inference time on top
of a from-scratch model, with no fine-tuning stage involved.
"""

import os
import re
import json
import numpy as np

name = "cot_reasoning"
description = (
    "Chain-of-Thought reasoning trainer for Decoder-Only Transformer models. "
    "Generates its own dataset + export-time reasoning handler on first run. "
    "Adds a reasoning on/off toggle and a self-questioning chat mode."
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
COT_STATE_FILE = os.path.join(BASE_DIR, "cot_reasoning_data.json")
DATASET_COMPANION_FILE = os.path.join(PROJECT_ROOT, "cot_reasoning_dataset.py")
HANDLER_COMPANION_FILE = os.path.join(PROJECT_ROOT, "reasoning_handler.py")

_THINK_RE = re.compile(r"<\s*think\s*>(.*?)<\s*/\s*think\s*>", re.IGNORECASE | re.DOTALL)


# -- Embedded CoT training data -----------------------------------------------
# This data lives IN this file (no separate import needed). On startup the
# plugin also writes it out to cot_reasoning_dataset.py for transparency/reuse
# by other scripts, but cot_reasoning.py itself never depends on that file
# existing -- it's generated FROM this plugin, not the other way around.

def _think(user, reasoning_steps, answer):
    reasoning = "\n".join(f"Step {i+1}: {s}" for i, s in enumerate(reasoning_steps))
    bot = f"<think>\n{reasoning}\n</think>\n{answer}"
    return {"user": user.strip(), "bot": bot}


CORE_COT_PAIRS = [
    _think(
        "What is 14 plus 27?",
        ["Identify the two numbers: 14 and 27.", "Add them together: 14 + 27 = 41."],
        "14 plus 27 is 41.",
    ),
    _think(
        "If I have 3 apples and buy 5 more, how many do I have?",
        ["Start with 3 apples.", "Add the 5 new apples: 3 + 5 = 8."],
        "You'd have 8 apples.",
    ),
    _think(
        "A train travels 60 miles in 1.5 hours. What is its speed?",
        ["Speed is distance divided by time.", "Distance = 60 miles, time = 1.5 hours.", "60 / 1.5 = 40."],
        "The train's speed is 40 miles per hour.",
    ),
    _think(
        "What is 9 times 7?",
        ["Multiply 9 by 7.", "9 * 7 = 63."],
        "9 times 7 is 63.",
    ),
    _think(
        "If a shirt costs $20 and is 25% off, what is the price?",
        ["25% of $20 is 0.25 * 20 = 5.", "Subtract the discount from the price: 20 - 5 = 15."],
        "The shirt costs $15 after the discount.",
    ),
    _think(
        "Which is bigger, 0.5 or 1/3?",
        ["Convert 1/3 to a decimal: 1/3 is about 0.333.", "Compare 0.5 and 0.333.", "0.5 is larger."],
        "0.5 is bigger than 1/3.",
    ),
    _think(
        "How many minutes are in 2.5 hours?",
        ["There are 60 minutes in 1 hour.", "Multiply: 2.5 * 60 = 150."],
        "There are 150 minutes in 2.5 hours.",
    ),
    _think(
        "If today is Wednesday, what day is it in 10 days?",
        ["10 days is 1 week (7 days) plus 3 extra days.", "One week from Wednesday is Wednesday again.", "3 more days after Wednesday is Saturday."],
        "In 10 days it will be Saturday.",
    ),
    _think(
        "I have $50 and spend $18 on lunch and $12 on a book. How much is left?",
        ["Total spent: 18 + 12 = 30.", "Subtract from starting amount: 50 - 30 = 20."],
        "You have $20 left.",
    ),
    _think(
        "What is the area of a rectangle 4m by 9m?",
        ["Area of a rectangle is length times width.", "4 * 9 = 36."],
        "The area is 36 square meters.",
    ),
    _think(
        "If a car uses 1 gallon every 25 miles, how many gallons for 100 miles?",
        ["Divide total distance by miles per gallon.", "100 / 25 = 4."],
        "The car would use 4 gallons.",
    ),
    _think(
        "Is 17 a prime number?",
        ["Check if 17 is divisible by any number other than 1 and itself.", "It is not divisible by 2, 3, 5 or anything up to its square root (about 4.1).", "No divisors found other than 1 and 17."],
        "Yes, 17 is a prime number.",
    ),
    _think(
        "What is half of 86?",
        ["Half means dividing by 2.", "86 / 2 = 43."],
        "Half of 86 is 43.",
    ),
    _think(
        "If 3 workers can build a wall in 12 days, how long would 6 workers take?",
        ["More workers means less time, so this is inverse proportion.", "3 workers times 12 days = 36 worker-days of total work needed.", "36 worker-days / 6 workers = 6 days."],
        "6 workers would take 6 days.",
    ),
    _think(
        "What's the next number in the sequence 2, 4, 8, 16?",
        ["Look at the relationship between consecutive numbers.", "Each number is double the previous one.", "16 * 2 = 32."],
        "The next number is 32.",
    ),
]


def get_cot_pairs(repeat_with_shuffle: int = 1, seed: int = 7):
    import random
    rng = random.Random(seed)
    out = []
    for _ in range(max(1, repeat_with_shuffle)):
        block = list(CORE_COT_PAIRS)
        rng.shuffle(block)
        out.extend(block)
    return out


def load_custom_cot_jsonl(path: str):
    """Load user-supplied CoT data: JSONL lines with prompt/reasoning/answer."""
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            prompt = (row.get("prompt") or row.get("user") or "").strip()
            reasoning = (row.get("reasoning") or "").strip()
            answer = (row.get("answer") or "").strip()
            if not prompt or not reasoning or not answer:
                continue
            pairs.append({"user": prompt, "bot": f"<think>\n{reasoning}\n</think>\n{answer}"})
    return pairs


# -- Self-generating companion files ------------------------------------------
def _write_dataset_companion_file():
    """Write cot_reasoning_dataset.py to disk from the embedded data above.
    Uses json.dumps so there's zero risk of quote-escaping bugs -- the
    generated file just decodes a JSON blob, it doesn't re-derive the data.
    """
    pairs_json = json.dumps(CORE_COT_PAIRS)
    content = (
        '"""Auto-generated by the cot_reasoning plugin. Do not edit directly --\n'
        'edit CORE_COT_PAIRS in plugins/cot_reasoning.py and restart instead."""\n'
        "import json\n\n"
        f"_PAIRS_JSON = {pairs_json!r}\n\n"
        "def get_cot_pairs():\n"
        "    return json.loads(_PAIRS_JSON)\n"
    )
    with open(DATASET_COMPANION_FILE, "w", encoding="utf-8") as f:
        f.write(content)


def _is_decoder_only(model) -> bool:
    if model is None:
        return False
    if hasattr(model, "hidden_dim") and not hasattr(model, "decoder_only"):
        return False  # RNN encoder-decoder model
    return bool(getattr(model, "decoder_only", False))


def split_reasoning(raw_text: str):
    match = _THINK_RE.search(raw_text)
    if not match:
        return None, raw_text.strip()
    reasoning = match.group(1).strip()
    answer = (raw_text[:match.start()] + raw_text[match.end():]).strip()
    return reasoning, answer


def _merge_into_pretrain(state, pairs):
    existing = state.get("pretrain_data") or []
    state["pretrain_data"] = existing + pairs


# -- Self-questioning generation ----------------------------------------------
def generate_with_reasoning(model, tokenizer, message: str, temperature: float = 0.8,
                             self_check: bool = True, max_length: int = 80):
    """Force the model to open a <think> block, generate reasoning + a draft
    answer, then (if self_check) feed that draft back in and ask the model to
    re-examine it before finalizing.

    This works WITHOUT modifying transformer_model.py: tokenizer.encode()
    splits '<', 'think', '>' as plain tokens, so seeding the prompt with
    "<think>" pre-loads the model into the reasoning pattern it was trained
    on -- it isn't a special token, just learned context.

    Returns: {"reasoning_pass1", "reasoning_final", "answer", "self_checked"}
    """
    pass1_prompt = f"{message} <think>"
    raw1 = model.generate(pass1_prompt, tokenizer, max_length=max_length, temperature=temperature)
    reasoning1, answer1 = split_reasoning("<think>" + raw1)

    if not self_check:
        return {
            "reasoning_pass1": reasoning1,
            "reasoning_final": reasoning1,
            "answer": answer1,
            "self_checked": False,
        }

    pass2_prompt = (
        f"{message} <think> {reasoning1 or ''} </think> {answer1} "
        f"Let me double check this. <think>"
    )
    raw2 = model.generate(pass2_prompt, tokenizer, max_length=max_length, temperature=temperature)
    reasoning2, answer2 = split_reasoning("<think>" + raw2)

    final_answer = answer2 if answer2 else answer1
    final_reasoning = reasoning2 if reasoning2 else reasoning1

    return {
        "reasoning_pass1": reasoning1,
        "reasoning_final": final_reasoning,
        "answer": final_answer,
        "self_checked": True,
    }


# -- Export-time reasoning handler (bundled with CoT exports) ----------------
def _reasoning_handler_code(gguf_filename: str = "model.gguf") -> str:
    """Generate a standalone CLI script that reproduces the same force-prefill
    + self-check reasoning technique against an exported GGUF, using
    llama-cpp-python. This is what ships alongside the .gguf when exporting
    in CoT mode, since raw GGUF inference (e.g. in Continue, LM Studio) has
    no way to know it should force-open <think> or run a second pass --
    only a real inference loop like this one can do that.
    """
    return f'''"""Standalone reasoning handler for a CoT-trained VnexAI model.
Generated by the cot_reasoning plugin. Requires: pip install llama-cpp-python

Usage:
    python reasoning_handler.py "your question here"
    python reasoning_handler.py "your question" --no-self-check
    python reasoning_handler.py "your question" --hide-think
"""
import sys
import argparse
from llama_cpp import Llama

MODEL_PATH = "{gguf_filename}"
THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"


def split_reasoning(raw_text):
    if THINK_CLOSE not in raw_text:
        return None, raw_text.strip()
    reasoning, _, answer = raw_text.partition(THINK_CLOSE)
    return reasoning.strip(), answer.strip()


def ask(llm, message, temperature, self_check, max_tokens):
    out1 = llm.create_completion(
        f"{{message}} {{THINK_OPEN}}", max_tokens=max_tokens,
        temperature=temperature, stop=["</s>"],
    )["choices"][0]["text"]
    reasoning1, answer1 = split_reasoning(THINK_OPEN + out1)

    if not self_check:
        return reasoning1, answer1

    prompt2 = (
        f"{{message}} {{THINK_OPEN}} {{reasoning1 or ''}} {{THINK_CLOSE}} "
        f"{{answer1}} Let me double check this. {{THINK_OPEN}}"
    )
    out2 = llm.create_completion(
        prompt2, max_tokens=max_tokens, temperature=temperature, stop=["</s>"],
    )["choices"][0]["text"]
    reasoning2, answer2 = split_reasoning(THINK_OPEN + out2)
    return (reasoning2 or reasoning1), (answer2 or answer1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("message")
    parser.add_argument("--no-self-check", action="store_true")
    parser.add_argument("--hide-think", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-tokens", type=int, default=120)
    args = parser.parse_args()

    llm = Llama(model_path=MODEL_PATH, n_ctx=512, verbose=False)
    reasoning, answer = ask(
        llm, args.message, args.temperature,
        not args.no_self_check, args.max_tokens,
    )

    if reasoning and not args.hide_think:
        print("--- thinking ---")
        print(reasoning)
        print("--- answer ---")
    print(answer)


if __name__ == "__main__":
    main()
'''


def _write_handler_companion_file():
    with open(HANDLER_COMPANION_FILE, "w", encoding="utf-8") as f:
        f.write(_reasoning_handler_code("model.gguf"))


def _decoder_only_error():
    return (
        "CoT reasoning requires a Decoder-Only Transformer model. "
        "RNN encoder-decoder and encoder-decoder Transformer models encode "
        "the user message and bot reply as separate sequences, so they "
        "can't generate their own reasoning before answering. Create a "
        "model with model_type 'transformer_decoder' to use this plugin."
    )


def register_routes(app, state, training_state):
    from fastapi import Request, HTTPException, UploadFile, File

    @app.get("/api/plugins/cot-reasoning/status")
    async def cot_status():
        model = state.get("model")
        data = state.get("cot_data")
        return {
            "loaded": data is not None,
            "count": len(data) if data else 0,
            "reasoning_enabled": bool(state.get("cot_reasoning_enabled", False)),
            "decoder_only": _is_decoder_only(model) if model else None,
            "requires_decoder_only": True,
            "dataset_file_generated": os.path.exists(DATASET_COMPANION_FILE),
            "handler_file_generated": os.path.exists(HANDLER_COMPANION_FILE),
        }

    @app.post("/api/plugins/cot-reasoning/load-builtin")
    async def cot_load_builtin(request: Request):
        raw_body = await request.body()
        body = json.loads(raw_body) if raw_body else {}
        repeat = int(body.get("repeat_with_shuffle", 1))
        pairs = get_cot_pairs(repeat_with_shuffle=repeat)
        state["cot_data"] = pairs
        _merge_into_pretrain(state, pairs)
        with open(COT_STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(pairs, f, ensure_ascii=False)
        return {"count": len(pairs), "message": f"Loaded {len(pairs)} built-in CoT pairs into pretrain data."}

    @app.post("/api/plugins/cot-reasoning/upload")
    async def cot_upload(file: UploadFile = File(...)):
        tmp_path = os.path.join(BASE_DIR, "_cot_upload_tmp.jsonl")
        contents = await file.read()
        with open(tmp_path, "wb") as f:
            f.write(contents)
        try:
            pairs = load_custom_cot_jsonl(tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        if not pairs:
            raise HTTPException(
                status_code=400,
                detail="No valid rows. Each JSONL line needs prompt, reasoning, answer fields.",
            )
        existing = state.get("cot_data") or []
        combined = existing + pairs
        state["cot_data"] = combined
        _merge_into_pretrain(state, pairs)
        with open(COT_STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(combined, f, ensure_ascii=False)
        return {"count": len(pairs), "total": len(combined), "message": f"Added {len(pairs)} custom CoT pairs."}

    @app.delete("/api/plugins/cot-reasoning")
    async def cot_delete():
        state["cot_data"] = None
        if os.path.exists(COT_STATE_FILE):
            os.remove(COT_STATE_FILE)
        return {"message": "CoT data deleted."}

    @app.get("/api/plugins/cot-reasoning/toggle")
    async def cot_get_toggle():
        return {"reasoning_enabled": bool(state.get("cot_reasoning_enabled", False))}

    @app.post("/api/plugins/cot-reasoning/toggle")
    async def cot_set_toggle(request: Request):
        body = await request.json()
        enabled = bool(body.get("enabled", False))
        if enabled:
            model = state.get("model")
            if model is not None and not _is_decoder_only(model):
                raise HTTPException(status_code=400, detail=_decoder_only_error())
        state["cot_reasoning_enabled"] = enabled
        return {"reasoning_enabled": enabled}

    @app.post("/api/plugins/cot-reasoning/chat")
    async def cot_chat(request: Request):
        body = await request.json()
        message = (body.get("message") or "").strip()
        temperature = float(body.get("temperature", 0.8))
        self_check = bool(body.get("self_check", True))
        if not message:
            raise HTTPException(status_code=400, detail="Empty message.")

        model = state.get("model")
        tokenizer = state.get("tokenizer")
        if not model or not tokenizer:
            raise HTTPException(status_code=400, detail="Model not ready.")
        if not state.get("is_trained"):
            raise HTTPException(status_code=400, detail="Model not trained yet.")
        if not _is_decoder_only(model):
            raise HTTPException(status_code=400, detail=_decoder_only_error())

        try:
            result = generate_with_reasoning(
                model, tokenizer, message, temperature=temperature, self_check=self_check,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Generation failed: {e}")

        reasoning_enabled = bool(state.get("cot_reasoning_enabled", False))
        returned_reasoning = result["reasoning_final"] if reasoning_enabled else None

        state.setdefault("chat_history", [])
        state["chat_history"].append({"role": "user", "content": message})
        state["chat_history"].append({"role": "bot", "content": result["answer"]})

        return {
            "response": result["answer"],
            "reasoning": returned_reasoning,
            "self_checked": result["self_checked"],
            "raw_has_think_block": result["reasoning_final"] is not None,
            "history": state["chat_history"],
        }

    @app.post("/api/plugins/cot-reasoning/export/prepare")
    async def cot_export_prepare(request: Request):
        body = await request.json()
        model_name = body.get("model_name", "vnexai_chatbot")
        model = state.get("model")
        if not model:
            raise HTTPException(status_code=400, detail="No model.")
        if not _is_decoder_only(model):
            raise HTTPException(status_code=400, detail=_decoder_only_error())
        handler_code = _reasoning_handler_code(f"{model_name}.gguf")
        state["cot_handler_bytes"] = handler_code.encode("utf-8")
        return {"size_bytes": len(state["cot_handler_bytes"]), "ready": True}

    @app.get("/api/plugins/cot-reasoning/export/download")
    async def cot_export_download(name: str = "vnexai_chatbot"):
        from fastapi.responses import StreamingResponse
        import io
        handler_bytes = state.get("cot_handler_bytes")
        if not handler_bytes:
            raise HTTPException(status_code=404, detail="Handler not prepared. Call export/prepare first.")
        return StreamingResponse(
            io.BytesIO(handler_bytes), media_type="text/x-python",
            headers={"Content-Disposition": 'attachment; filename="reasoning_handler.py"'},
        )


def on_startup(app, state, training_state):
    state.setdefault("cot_reasoning_enabled", False)

    try:
        _write_dataset_companion_file()
        dataset_ok = True
    except OSError:
        dataset_ok = False

    try:
        _write_handler_companion_file()
        handler_ok = True
    except OSError:
        handler_ok = False

    if os.path.exists(COT_STATE_FILE):
        try:
            with open(COT_STATE_FILE, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, list) and loaded:
                state["cot_data"] = loaded
                _merge_into_pretrain(state, loaded)
        except (json.JSONDecodeError, OSError):
            pass

    return (
        f"{name} initialized "
        f"(dataset file: {'ok' if dataset_ok else 'failed'}, "
        f"handler file: {'ok' if handler_ok else 'failed'})."
    )
