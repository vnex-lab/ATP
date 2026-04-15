"""
GGUF (GPT-Generated Unified Format) writer for VnexAI models.

GGUF is the file format used by llama.cpp and Ollama.
Spec: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md

NOTE: Ollama can only RUN models with architectures it understands (llama, mistral,
gemma, etc.). A VnexAI RNN or custom Transformer written here will be stored in
valid GGUF binary format, but Ollama will refuse to load it because the architecture
key won't match a built-in backend. The GGUF file is still useful for:
  - Archiving model weights in a standardised binary format
  - Loading with custom llama.cpp forks or your own C++ inference code
  - Inspecting with tools like `gguf-dump` from the llama.cpp repo
"""

import struct
import io
import numpy as np

# ── GGUF constants ──────────────────────────────────────────────────────────
GGUF_MAGIC   = b"GGUF"
GGUF_VERSION = 3
ALIGNMENT    = 32          # tensor data is aligned to 32 bytes

# Metadata value types
GGUF_TYPE_UINT8   = 0
GGUF_TYPE_INT8    = 1
GGUF_TYPE_UINT16  = 2
GGUF_TYPE_INT16   = 3
GGUF_TYPE_UINT32  = 4
GGUF_TYPE_INT32   = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL    = 7
GGUF_TYPE_STRING  = 8
GGUF_TYPE_ARRAY   = 9
GGUF_TYPE_UINT64  = 10
GGUF_TYPE_INT64   = 11
GGUF_TYPE_FLOAT64 = 12

# Tensor (GGML) dtype ids  –  we use F32 for all weights
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1


# ── Low-level helpers ───────────────────────────────────────────────────────

def _write_str(buf: io.BytesIO, s: str):
    """Write a GGUF length-prefixed UTF-8 string."""
    enc = s.encode("utf-8")
    buf.write(struct.pack("<Q", len(enc)))
    buf.write(enc)


def _write_kv(buf: io.BytesIO, key: str, vtype: int, value):
    """Write one metadata key-value pair."""
    _write_str(buf, key)
    buf.write(struct.pack("<I", vtype))

    if vtype == GGUF_TYPE_STRING:
        _write_str(buf, value)
    elif vtype == GGUF_TYPE_UINT8:
        buf.write(struct.pack("<B", value))
    elif vtype == GGUF_TYPE_INT8:
        buf.write(struct.pack("<b", value))
    elif vtype == GGUF_TYPE_UINT16:
        buf.write(struct.pack("<H", value))
    elif vtype == GGUF_TYPE_INT16:
        buf.write(struct.pack("<h", value))
    elif vtype == GGUF_TYPE_UINT32:
        buf.write(struct.pack("<I", value))
    elif vtype == GGUF_TYPE_INT32:
        buf.write(struct.pack("<i", value))
    elif vtype == GGUF_TYPE_FLOAT32:
        buf.write(struct.pack("<f", value))
    elif vtype == GGUF_TYPE_UINT64:
        buf.write(struct.pack("<Q", value))
    elif vtype == GGUF_TYPE_INT64:
        buf.write(struct.pack("<q", value))
    elif vtype == GGUF_TYPE_FLOAT64:
        buf.write(struct.pack("<d", value))
    elif vtype == GGUF_TYPE_BOOL:
        buf.write(struct.pack("<?", bool(value)))
    elif vtype == GGUF_TYPE_ARRAY:
        # value = (element_type, list_of_values)
        elem_type, items = value
        buf.write(struct.pack("<I", elem_type))
        buf.write(struct.pack("<Q", len(items)))
        for item in items:
            _write_kv_value(buf, elem_type, item)
    else:
        raise ValueError(f"Unknown GGUF value type: {vtype}")


def _write_kv_value(buf: io.BytesIO, vtype: int, value):
    """Write just the value bytes (used inside arrays)."""
    if vtype == GGUF_TYPE_STRING:
        _write_str(buf, value)
    elif vtype == GGUF_TYPE_UINT32:
        buf.write(struct.pack("<I", value))
    elif vtype == GGUF_TYPE_FLOAT32:
        buf.write(struct.pack("<f", value))
    else:
        raise ValueError(f"Unsupported array element type: {vtype}")


def _align_up(x: int, alignment: int) -> int:
    return (x + alignment - 1) & ~(alignment - 1)


# ── Main export function ─────────────────────────────────────────────────────

def export_rnn_to_gguf(model, model_name: str) -> bytes:
    """
    Serialise a VnexAI RNN model (VnexAIChatbot) to GGUF bytes.

    Tensors stored:
        token_embd.weight       [vocab_size, embedding_dim]
        enc.Wxh                 [embedding_dim, hidden_dim]
        enc.Whh                 [hidden_dim, hidden_dim]
        enc.bh                  [hidden_dim]
        dec.Wxh                 [embedding_dim, hidden_dim]
        dec.Whh                 [hidden_dim, hidden_dim]
        dec.bh                  [hidden_dim]
        output.weight           [hidden_dim, vocab_size]
        output.bias             [vocab_size]
    """
    arch = "vnexai-rnn"
    tensors = _collect_rnn_tensors(model)
    metadata = _rnn_metadata(model, arch, model_name)
    return _build_gguf(arch, metadata, tensors)


def export_transformer_to_gguf(model, model_name: str) -> bytes:
    """
    Serialise a VnexAI Transformer model (TransformerChatbot) to GGUF bytes.

    Collects all weight matrices from encoder/decoder layers plus embeddings.
    """
    arch = "vnexai-transformer"
    tensors = _collect_transformer_tensors(model)
    metadata = _transformer_metadata(model, arch, model_name)
    return _build_gguf(arch, metadata, tensors)


# ── Tensor collectors ────────────────────────────────────────────────────────

def _to_f32_np(arr) -> np.ndarray:
    """Convert a CuPy or NumPy array to a CPU float32 NumPy array."""
    if hasattr(arr, "get"):          # CuPy → NumPy
        arr = arr.get()
    return np.asarray(arr, dtype=np.float32)


def _collect_rnn_tensors(model) -> list:
    """Return list of (name, numpy_f32_array) tuples."""
    return [
        ("token_embd.weight", _to_f32_np(model.embedding)),
        ("enc.Wxh",           _to_f32_np(model.Wxh_enc)),
        ("enc.Whh",           _to_f32_np(model.Whh_enc)),
        ("enc.bh",            _to_f32_np(model.bh_enc).flatten()),
        ("dec.Wxh",           _to_f32_np(model.Wxh_dec)),
        ("dec.Whh",           _to_f32_np(model.Whh_dec)),
        ("dec.bh",            _to_f32_np(model.bh_dec).flatten()),
        ("output.weight",     _to_f32_np(model.Why)),
        ("output.bias",       _to_f32_np(model.by).flatten()),
    ]


def _collect_transformer_tensors(model) -> list:
    """Collect all Transformer weight matrices."""
    tensors = [
        ("token_embd.weight", _to_f32_np(model.embedding)),
    ]

    for i, layer in enumerate(model.encoder_layers):
        prefix = f"enc.layer.{i}"
        for attr_name, tensor_name in [
            ("Wq", f"{prefix}.attn.Wq"),
            ("Wk", f"{prefix}.attn.Wk"),
            ("Wv", f"{prefix}.attn.Wv"),
            ("Wo", f"{prefix}.attn.Wo"),
            ("W1", f"{prefix}.ff.W1"),
            ("b1", f"{prefix}.ff.b1"),
            ("W2", f"{prefix}.ff.W2"),
            ("b2", f"{prefix}.ff.b2"),
            ("gamma1", f"{prefix}.ln1.gamma"),
            ("beta1",  f"{prefix}.ln1.beta"),
            ("gamma2", f"{prefix}.ln2.gamma"),
            ("beta2",  f"{prefix}.ln2.beta"),
        ]:
            if hasattr(layer, attr_name):
                arr = _to_f32_np(getattr(layer, attr_name)).flatten()
                tensors.append((tensor_name, arr))

    for i, layer in enumerate(model.decoder_layers):
        prefix = f"dec.layer.{i}"
        for attr_name, tensor_name in [
            ("Wq", f"{prefix}.self_attn.Wq"),
            ("Wk", f"{prefix}.self_attn.Wk"),
            ("Wv", f"{prefix}.self_attn.Wv"),
            ("Wo", f"{prefix}.self_attn.Wo"),
            ("cross_Wq", f"{prefix}.cross_attn.Wq"),
            ("cross_Wk", f"{prefix}.cross_attn.Wk"),
            ("cross_Wv", f"{prefix}.cross_attn.Wv"),
            ("cross_Wo", f"{prefix}.cross_attn.Wo"),
            ("W1", f"{prefix}.ff.W1"),
            ("b1", f"{prefix}.ff.b1"),
            ("W2", f"{prefix}.ff.W2"),
            ("b2", f"{prefix}.ff.b2"),
        ]:
            if hasattr(layer, attr_name):
                arr = _to_f32_np(getattr(layer, attr_name)).flatten()
                tensors.append((tensor_name, arr))

    tensors.append(("output.weight", _to_f32_np(model.output_proj).flatten()
                    if hasattr(model, 'output_proj') else np.zeros(1, dtype=np.float32)))
    tensors.append(("output.bias",   _to_f32_np(model.output_bias).flatten()
                    if hasattr(model, 'output_bias') else np.zeros(1, dtype=np.float32)))

    return tensors


# ── Metadata builders ────────────────────────────────────────────────────────

def _rnn_metadata(model, arch: str, model_name: str) -> list:
    """Return list of (key, vtype, value) triples for RNN model metadata."""
    return [
        ("general.architecture",        GGUF_TYPE_STRING,  arch),
        ("general.name",                GGUF_TYPE_STRING,  model_name),
        ("general.file_type",           GGUF_TYPE_UINT32,  0),   # F32
        (f"{arch}.vocab_size",          GGUF_TYPE_UINT32,  int(model.vocab_size)),
        (f"{arch}.embedding_dim",       GGUF_TYPE_UINT32,  int(model.embedding_dim)),
        (f"{arch}.hidden_dim",          GGUF_TYPE_UINT32,  int(model.hidden_dim)),
        (f"{arch}.max_length",          GGUF_TYPE_UINT32,  int(model.max_length)),
        (f"{arch}.learning_rate",       GGUF_TYPE_FLOAT32, float(model.learning_rate)),
        ("tokenizer.ggml.model",        GGUF_TYPE_STRING,  "vnexai"),
        ("tokenizer.ggml.bos_token_id", GGUF_TYPE_UINT32,  1),  # <START>
        ("tokenizer.ggml.eos_token_id", GGUF_TYPE_UINT32,  2),  # <END>
        ("tokenizer.ggml.padding_token_id", GGUF_TYPE_UINT32, 0),
        ("tokenizer.ggml.unknown_token_id", GGUF_TYPE_UINT32, 3),
    ]


def _transformer_metadata(model, arch: str, model_name: str) -> list:
    return [
        ("general.architecture",        GGUF_TYPE_STRING,  arch),
        ("general.name",                GGUF_TYPE_STRING,  model_name),
        ("general.file_type",           GGUF_TYPE_UINT32,  0),
        (f"{arch}.vocab_size",          GGUF_TYPE_UINT32,  int(model.vocab_size)),
        (f"{arch}.embedding_dim",       GGUF_TYPE_UINT32,  int(model.embed_dim)),
        (f"{arch}.num_heads",           GGUF_TYPE_UINT32,  int(model.num_heads)),
        (f"{arch}.num_layers",          GGUF_TYPE_UINT32,  int(model.num_layers)),
        (f"{arch}.ff_dim",              GGUF_TYPE_UINT32,  int(model.ff_dim)),
        (f"{arch}.max_seq_len",         GGUF_TYPE_UINT32,  int(model.max_seq_len)),
        (f"{arch}.learning_rate",       GGUF_TYPE_FLOAT32, float(model.learning_rate)),
        ("tokenizer.ggml.model",        GGUF_TYPE_STRING,  "vnexai"),
        ("tokenizer.ggml.bos_token_id", GGUF_TYPE_UINT32,  1),
        ("tokenizer.ggml.eos_token_id", GGUF_TYPE_UINT32,  2),
        ("tokenizer.ggml.padding_token_id", GGUF_TYPE_UINT32, 0),
        ("tokenizer.ggml.unknown_token_id", GGUF_TYPE_UINT32, 3),
    ]


# ── GGUF binary builder ──────────────────────────────────────────────────────

def _build_gguf(arch: str, metadata: list, tensors: list) -> bytes:
    """
    Assemble the complete GGUF binary.

    Layout:
        [magic] [version] [n_tensors] [n_kv]
        [metadata k/v pairs …]
        [tensor info section …]
        [padding to ALIGNMENT]
        [tensor data …]   ← each tensor also padded to ALIGNMENT
    """
    header_buf  = io.BytesIO()
    meta_buf    = io.BytesIO()
    tinfo_buf   = io.BytesIO()
    tdata_buf   = io.BytesIO()

    # 1. Write metadata
    for key, vtype, value in metadata:
        _write_kv(meta_buf, key, vtype, value)

    # 2. Write tensor info and collect tensor data
    current_offset = 0
    prepared = []
    for name, arr in tensors:
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        shape = arr.shape
        n_dims = len(shape)
        nbytes = arr.nbytes

        # tensor info
        _write_str(tinfo_buf, name)
        tinfo_buf.write(struct.pack("<I", n_dims))
        for d in shape:
            tinfo_buf.write(struct.pack("<Q", d))
        tinfo_buf.write(struct.pack("<I", GGML_TYPE_F32))
        tinfo_buf.write(struct.pack("<Q", current_offset))

        # align offset for next tensor
        aligned_nbytes = _align_up(nbytes, ALIGNMENT)
        prepared.append((arr, nbytes, aligned_nbytes))
        current_offset += aligned_nbytes

    # 3. Header  (written last because we need counts)
    header_buf.write(GGUF_MAGIC)
    header_buf.write(struct.pack("<I", GGUF_VERSION))
    header_buf.write(struct.pack("<Q", len(tensors)))
    header_buf.write(struct.pack("<Q", len(metadata)))

    # 4. Assemble header + meta + tinfo, then pad the whole thing to ALIGNMENT
    pre_data = header_buf.getvalue() + meta_buf.getvalue() + tinfo_buf.getvalue()
    pad_needed = _align_up(len(pre_data), ALIGNMENT) - len(pre_data)
    pre_data += b"\x00" * pad_needed

    # 5. Write tensor data (each tensor padded to ALIGNMENT)
    for arr, nbytes, aligned_nbytes in prepared:
        tdata_buf.write(arr.tobytes())
        pad = aligned_nbytes - nbytes
        if pad:
            tdata_buf.write(b"\x00" * pad)

    return pre_data + tdata_buf.getvalue()
