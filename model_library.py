"""Saved model library — persist, load, import, and register models."""

from __future__ import annotations

import json
import os
import pickle
import re
import shutil
import time
from typing import Any, Dict, List, Optional, Tuple

from chatbot_model import VnexAIChatbot
from chatbot_tokenizer import ChatbotTokenizer
from transformer_model import TransformerChatbot

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LIBRARY_DIR = os.path.join(BASE_DIR, "saved_models")
INDEX_FILE = os.path.join(LIBRARY_DIR, "index.json")

MODEL_BIN = "model.bin"
TOKENIZER_BIN = "tokenizer.bin"
META_FILE = "meta.json"


def _slug(name: str) -> str:
    slug = re.sub(r"[^a-z0-9_-]+", "_", name.strip().lower())
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug or f"model_{int(time.time())}"


def _read_index() -> List[Dict[str, Any]]:
    if not os.path.exists(INDEX_FILE):
        return []
    try:
        with open(INDEX_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def _write_index(entries: List[Dict[str, Any]]) -> None:
    os.makedirs(LIBRARY_DIR, exist_ok=True)
    with open(INDEX_FILE, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)


def _model_dir(slug: str) -> str:
    return os.path.join(LIBRARY_DIR, slug)


def _detect_architecture(model_path: str) -> str:
    with open(model_path, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, dict) and "decoder_layers" in data:
        return "transformer_decoder" if data.get("decoder_only") else "transformer_enc_dec"
    if isinstance(data, dict) and "Wxh_enc" in data:
        return "rnn"
    raise ValueError("Unrecognized model file format.")


def _load_vnexai_model(model_path: str):
    arch = _detect_architecture(model_path)
    if arch == "rnn":
        with open(model_path, "rb") as f:
            data = pickle.load(f)
        model = VnexAIChatbot(
            vocab_size=data["vocab_size"],
            embedding_dim=data["embedding_dim"],
            hidden_dim=data["hidden_dim"],
            max_length=data["max_length"],
            learning_rate=data["learning_rate"],
        )
        model.load_model(model_path)
        model_type = "RNN"
    else:
        model = TransformerChatbot.load(model_path)
        model_type = "TransformerDecoder" if model.decoder_only else "TransformerEncoderDecoder"
    return model, model_type, arch


def list_models() -> List[Dict[str, Any]]:
    entries = _read_index()
    out = []
    for entry in entries:
        slug = entry.get("slug", "")
        folder = _model_dir(slug)
        meta_path = os.path.join(folder, META_FILE)
        meta = {}
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
            except (json.JSONDecodeError, OSError):
                pass
        out.append({**entry, **meta, "exists": os.path.isdir(folder)})
    return sorted(out, key=lambda x: x.get("updated_at", ""), reverse=True)


def save_model(
    state: Dict[str, Any],
    name: str,
    *,
    note: str = "",
    source: str = "trained",
) -> Dict[str, Any]:
    model = state.get("model")
    tokenizer = state.get("tokenizer")
    if not model or not tokenizer:
        raise ValueError("No model or tokenizer in memory.")

    slug = _slug(name)
    folder = _model_dir(slug)
    os.makedirs(folder, exist_ok=True)

    model_path = os.path.join(folder, MODEL_BIN)
    tok_path = os.path.join(folder, TOKENIZER_BIN)

    if hasattr(model, "save_model"):
        model.save_model(model_path)
    else:
        model.save(model_path)
    tokenizer.save(tok_path)

    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    meta = {
        "name": name.strip(),
        "slug": slug,
        "source": source,
        "note": note.strip(),
        "model_type": state.get("model_type"),
        "model_config": state.get("model_config"),
        "is_trained": bool(state.get("is_trained")),
        "decoder_only": bool(getattr(model, "decoder_only", False)),
        "vocab_size": tokenizer.vocab_size,
        "importable": True,
        "finetune_ready": True,
        "created_at": now,
        "updated_at": now,
    }
    with open(os.path.join(folder, META_FILE), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    entries = [e for e in _read_index() if e.get("slug") != slug]
    entries.append({"slug": slug, "name": name.strip(), "updated_at": now, "source": source})
    _write_index(entries)

    state["active_model_slug"] = slug
    state["active_model_name"] = name.strip()
    return meta


def load_model(state: Dict[str, Any], slug: str) -> Dict[str, Any]:
    folder = _model_dir(slug)
    model_path = os.path.join(folder, MODEL_BIN)
    tok_path = os.path.join(folder, TOKENIZER_BIN)
    meta_path = os.path.join(folder, META_FILE)

    if not os.path.exists(model_path) or not os.path.exists(tok_path):
        raise FileNotFoundError(f"Saved model '{slug}' not found.")

    model, model_type, arch = _load_vnexai_model(model_path)
    tokenizer = ChatbotTokenizer()
    tokenizer.load(tok_path)

    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

    state["model"] = model
    state["tokenizer"] = tokenizer
    state["model_type"] = model_type
    state["model_config"] = meta.get("model_config") or {"model_type": arch}
    state["is_trained"] = bool(meta.get("is_trained", True))
    state["active_model_slug"] = slug
    state["active_model_name"] = meta.get("name", slug)
    state["training_mode"] = "finetune"
    state["chat_history"] = []

    return {
        "slug": slug,
        "name": meta.get("name", slug),
        "model_type": model_type,
        "vocab_size": tokenizer.vocab_size,
        "is_trained": state["is_trained"],
        "decoder_only": bool(getattr(model, "decoder_only", False)),
        "source": meta.get("source", "library"),
    }


def delete_model(slug: str) -> None:
    folder = _model_dir(slug)
    if os.path.isdir(folder):
        shutil.rmtree(folder)
    entries = [e for e in _read_index() if e.get("slug") != slug]
    _write_index(entries)


def import_vnexai_files(
    state: Dict[str, Any],
    name: str,
    model_bytes: bytes,
    tokenizer_bytes: Optional[bytes] = None,
    *,
    note: str = "",
) -> Dict[str, Any]:
    slug = _slug(name)
    folder = _model_dir(slug)
    os.makedirs(folder, exist_ok=True)

    model_path = os.path.join(folder, MODEL_BIN)
    with open(model_path, "wb") as f:
        f.write(model_bytes)

    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    model_type = "Unknown"
    arch = "unknown"
    decoder_only = False
    vocab_size: Optional[int] = None

    try:
        model, model_type, arch = _load_vnexai_model(model_path)
        decoder_only = bool(getattr(model, "decoder_only", False))
    except Exception as exc:
        if tokenizer_bytes:
            raise ValueError(f"Could not load model weights: {exc}") from exc
        meta = {
            "name": name.strip(),
            "slug": slug,
            "source": "imported",
            "note": (note.strip() or "Imported weights without a VnexAI tokenizer."),
            "model_type": "External",
            "importable": False,
            "finetune_ready": False,
            "tokenizer_missing": True,
            "created_at": now,
            "updated_at": now,
        }
        with open(os.path.join(folder, META_FILE), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        entries = [e for e in _read_index() if e.get("slug") != slug]
        entries.append({"slug": slug, "name": name.strip(), "updated_at": now, "source": "imported"})
        _write_index(entries)
        return meta

    if not tokenizer_bytes:
        meta = {
            "name": name.strip(),
            "slug": slug,
            "source": "imported",
            "note": (note.strip() or "Weights saved without tokenizer (e.g. Llama/GGUF). Add a VnexAI tokenizer .bin to fine-tune."),
            "model_type": model_type,
            "model_config": {"model_type": arch},
            "is_trained": False,
            "decoder_only": decoder_only,
            "importable": True,
            "finetune_ready": False,
            "tokenizer_missing": True,
            "created_at": now,
            "updated_at": now,
        }
        with open(os.path.join(folder, META_FILE), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        entries = [e for e in _read_index() if e.get("slug") != slug]
        entries.append({"slug": slug, "name": name.strip(), "updated_at": now, "source": "imported"})
        _write_index(entries)
        return meta

    tok_path = os.path.join(folder, TOKENIZER_BIN)
    with open(tok_path, "wb") as f:
        f.write(tokenizer_bytes)

    tokenizer = ChatbotTokenizer()
    tokenizer.load(tok_path)
    vocab_size = tokenizer.vocab_size

    meta = {
        "name": name.strip(),
        "slug": slug,
        "source": "imported",
        "note": note.strip(),
        "model_type": model_type,
        "model_config": {"model_type": arch},
        "is_trained": True,
        "decoder_only": decoder_only,
        "vocab_size": vocab_size,
        "importable": True,
        "finetune_ready": True,
        "tokenizer_missing": False,
        "created_at": now,
        "updated_at": now,
    }
    with open(os.path.join(folder, META_FILE), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    entries = [e for e in _read_index() if e.get("slug") != slug]
    entries.append({"slug": slug, "name": name.strip(), "updated_at": now, "source": "imported"})
    _write_index(entries)

    state["model"] = model
    state["tokenizer"] = tokenizer
    state["model_type"] = model_type
    state["model_config"] = meta["model_config"]
    state["is_trained"] = True
    state["active_model_slug"] = slug
    state["active_model_name"] = name.strip()
    state["training_mode"] = "finetune"
    state["chat_history"] = []

    return meta


def register_external_model(
    name: str,
    *,
    format_label: str = "external",
    reference: str = "",
    note: str = "",
) -> Dict[str, Any]:
    """Register metadata for an external model (Llama, GGUF, etc.) for library tracking."""
    slug = _slug(name)
    folder = _model_dir(slug)
    os.makedirs(folder, exist_ok=True)

    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    meta = {
        "name": name.strip(),
        "slug": slug,
        "source": "external",
        "format": format_label.strip() or "external",
        "reference": reference.strip(),
        "note": note.strip(),
        "importable": False,
        "finetune_ready": False,
        "created_at": now,
        "updated_at": now,
    }
    with open(os.path.join(folder, META_FILE), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    entries = [e for e in _read_index() if e.get("slug") != slug]
    entries.append({"slug": slug, "name": name.strip(), "updated_at": now, "source": "external"})
    _write_index(entries)
    return meta
