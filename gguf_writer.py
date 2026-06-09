"""
GGUF (GPT-Generated Unified Format) writer for VnexAI models.

GGUF is the file format used by llama.cpp and Ollama.
Spec: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md

VnexAI Transformer models are exported using the "llama" architecture so that
Ollama (and llama.cpp) can load and run them directly.  The decoder layers are
mapped to llama block format; the encoder is omitted since Ollama only supports
decoder-only inference.
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
    elif vtype == GGUF_TYPE_INT32:
        buf.write(struct.pack("<i", value))
    elif vtype == GGUF_TYPE_FLOAT32:
        buf.write(struct.pack("<f", value))
    else:
        raise ValueError(f"Unsupported array element type: {vtype}")


def _align_up(x: int, alignment: int) -> int:
    return (x + alignment - 1) & ~(alignment - 1)


def _to_f32_np(arr) -> np.ndarray:
    """Convert a CuPy or NumPy array to a CPU float32 NumPy array."""
    if hasattr(arr, "get"):          # CuPy → NumPy
        arr = arr.get()
    return np.asarray(arr, dtype=np.float32)


# ── Main export functions ─────────────────────────────────────────────────────

def export_rnn_to_gguf(model, model_name: str) -> bytes:
    """
    Serialise a VnexAI RNN model (VnexAIChatbot) to GGUF bytes.
    Uses custom 'vnexai-rnn' architecture (inspection/archive use).
    """
    arch = "vnexai-rnn"
    tensors = _collect_rnn_tensors(model)
    metadata = _rnn_metadata(model, arch, model_name)
    return _build_gguf(arch, metadata, tensors)


def export_transformer_to_gguf(model, tokenizer, model_name: str) -> bytes:
    """
    Serialise a VnexAI Transformer model to GGUF using the 'llama' architecture
    so that Ollama and llama.cpp can load and run it directly.

    Mapping strategy
    ----------------
    - Decoder layers → blk.{i}.* (llama block format)
    - Encoder layers are omitted (Ollama is decoder-only)
    - Attention weights are transposed to match llama.cpp's matmul convention
    - FFN gate = FFN up = W1.T  (we have ReLU, llama has SiLU+gate;
      setting gate = up is the closest approximation)
    - FFN down = W2.T
    - Layer-norm gamma → attn_norm / ffn_norm
    - Full vocabulary embedded in tokenizer section
    """
    _validate_transformer_for_llama_export(model, tokenizer)
    arch = "llama"
    tensors = _collect_transformer_tensors_llama(model)
    metadata = _transformer_metadata_llama(model, tokenizer, model_name)
    return _build_gguf(arch, metadata, tensors)


# keep old name as alias for any existing callers
export_transformer_to_gguf_ollama = export_transformer_to_gguf


def _validate_transformer_for_llama_export(model, tokenizer):
    """
    Strict preflight checks so we fail early with a clear message instead of
    emitting a malformed GGUF that crashes the Ollama runner at generation time.
    """
    if not getattr(model, "decoder_only", False):
        raise ValueError(
            "Ollama export requires a Decoder-Only Transformer model. "
            "Create/train with 'Transformer Decoder-Only (Ollama Recommended)'."
        )

    if int(model.embed_dim) % int(model.num_heads) != 0:
        raise ValueError("embed_dim must be divisible by num_heads for llama export.")

    if not hasattr(tokenizer, "vocab_size") or int(tokenizer.vocab_size) <= 4:
        raise ValueError("Tokenizer vocabulary is too small. Build vocabulary before export.")

    required = (
        "Wq_self", "Wk_self", "Wv_self", "Wo_self",
        "ff_w1", "ff_w2", "ln1_gamma", "ln2_gamma"
    )

    for i, layer in enumerate(model.decoder_layers):
        for key in required:
            if key not in layer:
                raise ValueError(f"Decoder layer {i} missing required weight: {key}")
            arr = _to_f32_np(layer[key])
            if not np.isfinite(arr).all():
                raise ValueError(f"Decoder layer {i} has NaN/Inf in {key}; aborting export.")

    for name, arr in (
        ("embedding", model.embedding),
        ("output_weights", model.output_weights),
    ):
        np_arr = _to_f32_np(arr)
        if not np.isfinite(np_arr).all():
            raise ValueError(f"Model has NaN/Inf in {name}; aborting export.")


# ── Tensor collectors ─────────────────────────────────────────────────────────

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


def _collect_transformer_tensors_llama(model) -> list:
    """
    Collect decoder layer weights mapped to llama tensor naming.

    llama.cpp matmul convention: ggml_mul_mat(W, x) computes W @ x where
    W has shape [out, in].  Our forward pass does x @ W where W is [in, out].
    Therefore we transpose each weight matrix before storing.
    """
    tensors = []

    # Token embeddings in llama GGUF are expected by Ollama as [embed_dim, vocab_size]
    # for this exporter path, so transpose from our [vocab_size, embed_dim].
    tensors.append(("token_embd.weight", _to_f32_np(model.embedding).T))

    num_dec = len(model.decoder_layers)
    for i, layer in enumerate(model.decoder_layers):
        # Self-attention weights (transpose: [in, out] → [out, in])
        tensors.append((f"blk.{i}.attn_q.weight",
                        _to_f32_np(layer['Wq_self']).T))
        tensors.append((f"blk.{i}.attn_k.weight",
                        _to_f32_np(layer['Wk_self']).T))
        tensors.append((f"blk.{i}.attn_v.weight",
                        _to_f32_np(layer['Wv_self']).T))
        tensors.append((f"blk.{i}.attn_output.weight",
                        _to_f32_np(layer['Wo_self']).T))

        # Feed-forward weights
        # Ollama expects:
        #   ffn_gate/up: [embed_dim, ff_dim]
        #   ffn_down:    [ff_dim, embed_dim]
        # which already matches our internal shapes.
        w1 = _to_f32_np(layer['ff_w1'])        # [embed_dim, ff_dim]
        w2 = _to_f32_np(layer['ff_w2'])        # [ff_dim, embed_dim]
        tensors.append((f"blk.{i}.ffn_gate.weight", w1))
        tensors.append((f"blk.{i}.ffn_up.weight",   w1))
        tensors.append((f"blk.{i}.ffn_down.weight", w2))

        # Llama checkpoints are bias-free in the FFN path.
        # Do not export FFN biases, otherwise Ollama can reject the file with
        # unknown/unexpected tensor names.

        # Layer-norm weights (gamma → llama norm weight)
        tensors.append((f"blk.{i}.attn_norm.weight",
                        _to_f32_np(layer['ln1_gamma']).flatten()))
        tensors.append((f"blk.{i}.ffn_norm.weight",
                        _to_f32_np(layer['ln2_gamma']).flatten()))

    # output.weight: llama.cpp expects [vocab_size, embed_dim] (it does output = x @ W.T)
    # Our model stores [embed_dim, vocab_size], so we must transpose.
    tensors.append(("output.weight",
                    _to_f32_np(model.output_weights).T))

    # Final norm — use last decoder block's ln3 gamma if present, else ln2, else ones
    if num_dec > 0:
        last = model.decoder_layers[-1]
        final_norm_arr = last.get('ln3_gamma', last.get('ln2_gamma', None))
        if final_norm_arr is not None:
            final_norm = _to_f32_np(final_norm_arr).flatten()
        else:
            final_norm = np.ones(model.embed_dim, dtype=np.float32)
    else:
        final_norm = np.ones(model.embed_dim, dtype=np.float32)
    tensors.append(("output_norm.weight", final_norm))

    return tensors


# ── Metadata builders ─────────────────────────────────────────────────────────

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
        ("tokenizer.ggml.model",        GGUF_TYPE_STRING,  "vnexai"),
        ("tokenizer.ggml.bos_token_id", GGUF_TYPE_UINT32,  1),  # <START>
        ("tokenizer.ggml.eos_token_id", GGUF_TYPE_UINT32,  2),  # <END>
        ("tokenizer.ggml.padding_token_id", GGUF_TYPE_UINT32, 0),
        ("tokenizer.ggml.unknown_token_id", GGUF_TYPE_UINT32, 3),
    ]


def _transformer_metadata_llama(model, tokenizer, model_name: str) -> list:
    """
    Build the full llama-compatible metadata block, including the embedded
    vocabulary so that Ollama can tokenise user input without an external file.
    """
    num_layers = len(model.decoder_layers)
    head_dim   = model.embed_dim // model.num_heads

    # ── Build vocabulary list in index order ──────────────────────────────
    # tokenizer.idx2word is {int: str}; pad any missing indices with <UNK>
    vocab_size = tokenizer.vocab_size
    vocab_list = []
    for idx in range(vocab_size):
        token = tokenizer.idx2word.get(idx, "<UNK>")
        # Special tokens stay as-is; regular words get a leading space so that
        # llama.cpp reconstructs "hello world" instead of "helloworld" when it
        # concatenates adjacent tokens.  We use the "none" tokenizer model
        # below so llama.cpp treats each entry as a literal string — the space
        # prefix is the only thing that adds word boundaries.
        if token in ("<PAD>", "<START>", "<END>", "<UNK>"):
            vocab_list.append(token)
        else:
            vocab_list.append(" " + token)  # leading space = word boundary

    # Token scores — llama expects one float per token (0.0 is fine for us)
    token_scores = [0.0] * vocab_size

    # Token types — 1 = normal, 3 = BOS, 4 = EOS, 6 = PAD
    token_types = [1] * vocab_size
    if vocab_size > 0:
        token_types[0] = 6   # <PAD>
    if vocab_size > 1:
        token_types[1] = 3   # <START> / BOS
    if vocab_size > 2:
        token_types[2] = 4   # <END>  / EOS
    if vocab_size > 3:
        token_types[3] = 1   # <UNK>

    metadata = [
        # ── general ──────────────────────────────────────────────────────
        ("general.architecture",              GGUF_TYPE_STRING,  "llama"),
        ("general.name",                      GGUF_TYPE_STRING,  model_name),
        ("general.file_type",                 GGUF_TYPE_UINT32,  0),   # F32

        # ── llama architecture params ────────────────────────────────────
        ("llama.context_length",              GGUF_TYPE_UINT32,  int(model.max_seq_len)),
        ("llama.embedding_length",            GGUF_TYPE_UINT32,  int(model.embed_dim)),
        ("llama.block_count",                 GGUF_TYPE_UINT32,  int(num_layers)),
        ("llama.feed_forward_length",         GGUF_TYPE_UINT32,  int(model.ff_dim)),
        ("llama.rope.dimension_count",        GGUF_TYPE_UINT32,  int(head_dim)),
        ("llama.attention.head_count",        GGUF_TYPE_UINT32,  int(model.num_heads)),
        ("llama.attention.head_count_kv",     GGUF_TYPE_UINT32,  int(model.num_heads)),
        ("llama.attention.layer_norm_rms_epsilon", GGUF_TYPE_FLOAT32, 1e-5),
        ("llama.vocab_size",                  GGUF_TYPE_UINT32,  int(vocab_size)),

        # ── tokenizer ─────────────────────────────────────────────────────
        # "none" = raw token-ID pass-through; llama.cpp will not attempt BPE
        # merging or byte-level fallback — it concatenates token strings as-is.
        # This matches our word-level tokenizer perfectly.  Each regular token
        # has a leading space (added in vocab_list above) so word boundaries
        # are preserved without any merge rules.
        ("tokenizer.ggml.model",              GGUF_TYPE_STRING,  "none"),
        ("tokenizer.ggml.pre",                GGUF_TYPE_STRING,  "default"),
        ("tokenizer.ggml.tokens",             GGUF_TYPE_ARRAY,
            (GGUF_TYPE_STRING, vocab_list)),
        ("tokenizer.ggml.merges",             GGUF_TYPE_ARRAY,
            (GGUF_TYPE_STRING, [])),
        ("tokenizer.ggml.token_type",         GGUF_TYPE_ARRAY,
            (GGUF_TYPE_INT32, token_types)),
        ("tokenizer.ggml.scores",             GGUF_TYPE_ARRAY,
            (GGUF_TYPE_FLOAT32, token_scores)),
        ("tokenizer.ggml.bos_token_id",       GGUF_TYPE_UINT32,  1 if vocab_size > 1 else 0),
        ("tokenizer.ggml.eos_token_id",       GGUF_TYPE_UINT32,  2 if vocab_size > 2 else 0),
        ("tokenizer.ggml.padding_token_id",   GGUF_TYPE_UINT32,  0),
        ("tokenizer.ggml.unknown_token_id",   GGUF_TYPE_UINT32,  3 if vocab_size > 3 else 0),
        ("tokenizer.ggml.add_bos_token",      GGUF_TYPE_BOOL,    True),
        ("tokenizer.ggml.add_eos_token",      GGUF_TYPE_BOOL,    False),
    ]

    return metadata


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
