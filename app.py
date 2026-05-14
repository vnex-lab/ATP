import streamlit as st
import numpy as np
import json
import io
import re
from chatbot_model import VnexAIChatbot
from transformer_model import TransformerChatbot
from chatbot_tokenizer import ChatbotTokenizer


def parse_code_parquet(df, max_pairs=2000):
    """
    Extract user/bot training pairs from code-file Parquet datasets like
    The Stack (bigcode/the-stack) or any dataset with a 'content' column.
    Strategy 1: Extract function-signature + docstring → function body pairs.
    Strategy 2 (fallback): Split each file in thirds as code-completion pairs.
    """
    col_map = {c.lower(): c for c in df.columns}
    content_col = col_map.get('content')
    lang_col = col_map.get('lang')

    if not content_col:
        return []

    # Prefer Python files if a language column exists
    if lang_col:
        df = df[df[lang_col].str.lower().isin(['python', 'py'])].copy()

    data = []

    # Regex: match Python functions that have a docstring immediately after the signature
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
                user_msg = f'Write a Python function that: {doc}'
                bot_msg = f'{sig}\n    """{doc}"""\n    {body}'
                data.append({'user': user_msg, 'bot': bot_msg})
                if len(data) >= max_pairs:
                    break

    # Fallback: code-split style if docstring extraction yielded too few pairs
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

# Set page configuration
st.set_page_config(
    page_title="VnexAI Chatbot - Train Your Own AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .user-message {
        background-color: #e3f2fd;
        text-align: right;
        color: black !important;
    }
    .bot-message {
        background-color: #f5f5f5;
        text-align: left;
        color: black !important;
    }
    .chat-message div, .chat-message p, .chat-message span {
        color: black !important;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chatbot_model' not in st.session_state:
    st.session_state.chatbot_model = None
if 'tokenizer' not in st.session_state:
    st.session_state.tokenizer = None
if 'training_data' not in st.session_state:
    st.session_state.training_data = None
if 'is_trained' not in st.session_state:
    st.session_state.is_trained = False
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'model_bytes' not in st.session_state:
    st.session_state.model_bytes = None
if 'tokenizer_bytes' not in st.session_state:
    st.session_state.tokenizer_bytes = None
if 'gguf_bytes' not in st.session_state:
    st.session_state.gguf_bytes = None
if 'training_data_profile' not in st.session_state:
    st.session_state.training_data_profile = None  # "sft_pack" | "assistant_pack" | None

def main():
    # Main title
    st.markdown('<h1 class="main-header">🤖 VnexAI Chatbot</h1>', unsafe_allow_html=True)
    st.markdown("Train your own AI chatbot from scratch - supports conversation & code debugging! Export as .bin!")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a section:",
        ["Data Upload", "Model Setup", "Training", "Chat Interface", "Export Model"]
    )
    
    if page == "Data Upload":
        data_upload_section()
    elif page == "Model Setup":
        model_setup_section()
    elif page == "Training":
        training_section()
    elif page == "Chat Interface":
        chat_interface_section()
    elif page == "Export Model":
        export_model_section()

def data_upload_section():
    st.header("📊 Training Data Upload")
    
    st.write("""
    Upload your conversation training data. Supports multiple formats:
    - **Built-in Assistant Dialog**: Large, general assistant-style pairs (default ~90 MB JSON; adjustable).
    - **Built-in SFT Pack**: Instruction → response pairs built so **SFT mode** (decoder-only) has a clear user/assistant boundary.
    - **JSON**: Conversation format or code debugging format
    - **CSV**: Two columns (user, bot) or (question, answer) or (input, output)
    - **Text**: Line-by-line with separators (|, →, -, :, tab)
    - **Numbered JSON**: Format like `0:{...} 1:{...}`
    """)
    
    upload_type = st.radio(
        "Choose upload method:",
        [
            "Upload File",
            "Enter Text Manually",
            "Use Built-in Assistant Dialog",
            "Use Built-in SFT Pack (decoder-only + SFT)",
        ],
    )

    if upload_type == "Use Built-in Assistant Dialog":
        st.subheader("Built-in Assistant Dialog")
        st.write(
            "Professional assistant-style replies (e.g. **Hello** → **Hello! How can I help you today?**), "
            "plus many help-desk and learning topics. Size is **JSON in RAM/disk** — your GTX 1650 **VRAM** "
            "is mostly from **model size**, not this slider. If RAM is tight, choose a smaller MB."
        )
        target_mb = st.slider(
            "Approximate dataset size (MB)",
            min_value=15,
            max_value=100,
            value=90,
            help="Target serialized JSON size. Actual size can differ by a few MB.",
        )
        if st.button("Generate & load built-in dataset"):
            with st.spinner("Generating conversation pairs (may take a few seconds)..."):
                from builtin_assistant_dataset import (
                    build_builtin_assistant_dataset,
                    dataset_json_size_bytes,
                )

                samples = build_builtin_assistant_dataset(
                    target_bytes=int(target_mb * 1024 * 1024),
                )
                st.session_state.training_data = samples
                st.session_state.training_data_profile = "assistant_pack"
                mb = dataset_json_size_bytes(samples) / (1024 * 1024)
                st.success(
                    f"Loaded **{len(samples):,}** pairs (~**{mb:.1f} MB** JSON). "
                    "Head to **Model Setup**, then **Training**."
                )

    elif upload_type == "Use Built-in SFT Pack (decoder-only + SFT)":
        st.subheader("Built-in SFT Pack")
        st.write(
            "This pack is **only** meant for **decoder-only** training with **SFT turned ON** on the Training tab. "
            "Every **user** row reads like an **instruction, task, or question**; every **bot** row is the **assistant completion**. "
            "That way masked loss focuses on learning *how to answer*, not on copying chit-chat in the user half."
        )
        sft_rows = st.slider(
            "Number of training pairs",
            min_value=3_000,
            max_value=100_000,
            value=25_000,
            step=1_000,
            help="Unique templates are repeated with shuffled cycles to reach this count. "
            "Smaller = faster epochs; larger = more repetition of the same patterns.",
        )
        if st.button("Generate & load SFT pack"):
            with st.spinner("Building SFT-oriented pairs..."):
                from builtin_sft_dataset import build_builtin_sft_dataset, dataset_json_size_bytes

                samples = build_builtin_sft_dataset(target_rows=int(sft_rows))
                st.session_state.training_data = samples
                st.session_state.training_data_profile = "sft_pack"
                mb = dataset_json_size_bytes(samples) / (1024 * 1024)
                st.success(
                    f"Loaded **{len(samples):,}** SFT-style pairs (~**{mb:.1f} MB** JSON). "
                    "Create a **decoder-only** model, then in **Training** enable **SFT**."
                )

    elif upload_type == "Upload File":
        uploaded_file = st.file_uploader("Upload conversation data", type=['json', 'txt', 'csv', 'tsv', 'jsonl', 'parquet'])
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.json') or uploaded_file.name.endswith('.jsonl'):
                    # Handle both JSON and JSONL (JSON Lines)
                    content = uploaded_file.read().decode('utf-8')
                    
                    # Try regular JSON first
                    try:
                        raw_data = json.loads(content)
                    except json.JSONDecodeError:
                        # Try JSONL (one JSON object per line)
                        raw_data = []
                        for line in content.strip().split('\n'):
                            if line.strip():
                                try:
                                    raw_data.append(json.loads(line))
                                except:
                                    continue
                    
                    # Check if it's code debugging format
                    if isinstance(raw_data, list) and len(raw_data) > 0:
                        first_item = raw_data[0]
                        
                        # Code debugging format detection
                        if 'original_src' in first_item and 'changed_src' in first_item:
                            data = []
                            for item in raw_data:
                                # Extract fields
                                language = item.get('language', 'code')
                                status = item.get('original_status', 'error')
                                original = item.get('original_src', '')
                                fixed = item.get('changed_src', '')
                                
                                if original and fixed:
                                    # Create user-bot pair
                                    user_msg = f"Fix this {language} code with {status}: {original}"
                                    bot_msg = fixed
                                    data.append({'user': user_msg, 'bot': bot_msg})
                            
                            st.session_state.training_data = data
                            st.info(f"Detected code debugging format. Converted to {len(data)} training pairs.")
                        
                        # Standard conversation format
                        elif 'user' in first_item and 'bot' in first_item:
                            st.session_state.training_data = raw_data
                        
                        else:
                            st.error("Unknown JSON format. Please use conversation format or code debugging format.")
                            return
                    else:
                        st.session_state.training_data = raw_data
                
                elif uploaded_file.name.endswith('.txt'):  # txt file
                    content = uploaded_file.read().decode('utf-8')
                    data = []
                    
                    # Try to parse as numbered JSON entries (0:{...}, 1:{...})
                    import re
                    entries = re.split(r'\n\d+:\{', content)
                    
                    parsed_json = False
                    for i, entry in enumerate(entries):
                        if not entry.strip():
                            continue
                        
                        # Add back the opening brace if it's not the first entry
                        if i > 0:
                            entry = '{' + entry
                        else:
                            if not entry.strip().startswith('{'):
                                continue
                        
                        # Try to parse as JSON
                        try:
                            last_brace = entry.rfind('}')
                            if last_brace != -1:
                                json_str = entry[:last_brace + 1]
                                obj = json.loads(json_str)
                                
                                # Check if it's code debugging format
                                if 'original_src' in obj and 'changed_src' in obj:
                                    language = obj.get('language', 'code')
                                    status = obj.get('original_status', 'error')
                                    original = obj.get('original_src', '')
                                    fixed = obj.get('changed_src', '')
                                    
                                    if original and fixed:
                                        user_msg = f"Fix this {language} code with {status}: {original}"
                                        bot_msg = fixed
                                        data.append({'user': user_msg, 'bot': bot_msg})
                                        parsed_json = True
                        except:
                            continue
                    
                    # If no JSON was parsed, try line-by-line format
                    if not parsed_json:
                        lines = content.split('\n')
                        for line in lines:
                            # Try different separators: |, →, -, tab
                            for sep in ['|', '→', '\t', ' - ']:
                                if sep in line:
                                    parts = line.split(sep, 1)
                                    if len(parts) == 2:
                                        user_text = parts[0].strip()
                                        bot_text = parts[1].strip()
                                        
                                        # Remove common labels
                                        for label in ['user:', 'question:', 'input:', 'q:', 'human:']:
                                            if user_text.lower().startswith(label):
                                                user_text = user_text[len(label):].strip()
                                                break
                                        
                                        for label in ['bot:', 'answer:', 'output:', 'a:', 'assistant:', 'response:']:
                                            if bot_text.lower().startswith(label):
                                                bot_text = bot_text[len(label):].strip()
                                                break
                                        
                                        data.append({'user': user_text, 'bot': bot_text})
                                        break
                    
                    st.session_state.training_data = data
                    
                    if parsed_json:
                        st.info(f"Detected code debugging format in text file. Converted to {len(data)} training pairs.")
                
                elif uploaded_file.name.endswith('.csv') or uploaded_file.name.endswith('.tsv'):
                    # Handle CSV/TSV files
                    import csv
                    import io
                    
                    content = uploaded_file.read().decode('utf-8')
                    delimiter = '\t' if uploaded_file.name.endswith('.tsv') else ','
                    
                    reader = csv.DictReader(io.StringIO(content), delimiter=delimiter)
                    data = []
                    
                    for row in reader:
                        # Try to find user/bot columns with various names
                        user_text = None
                        bot_text = None
                        
                        # Check for user column
                        for key in row.keys():
                            key_lower = key.lower().strip()
                            if key_lower in ['user', 'question', 'input', 'q', 'prompt', 'human']:
                                if row[key]:
                                    user_text = row[key].strip()
                                break
                        
                        # Check for bot column
                        for key in row.keys():
                            key_lower = key.lower().strip()
                            if key_lower in ['bot', 'answer', 'output', 'a', 'response', 'assistant', 'reply']:
                                if row[key]:
                                    bot_text = row[key].strip()
                                break
                        
                        # If no headers matched, use first two columns
                        if user_text is None or bot_text is None:
                            cols = list(row.values())
                            if len(cols) >= 2:
                                if cols[0]:
                                    user_text = cols[0].strip()
                                if cols[1]:
                                    bot_text = cols[1].strip()
                        
                        if user_text and bot_text:
                            data.append({'user': user_text, 'bot': bot_text})
                    
                    st.session_state.training_data = data

                elif uploaded_file.name.endswith('.parquet'):
                    import pandas as pd
                    df = pd.read_parquet(uploaded_file)
                    col_lower_set = {c.lower() for c in df.columns}
                    
                    # --- Detect code-file datasets (The Stack, GitHub Code, etc.) ---
                    if 'content' in col_lower_set:
                        with st.spinner("Extracting function/docstring pairs from code files..."):
                            data = parse_code_parquet(df, max_pairs=2000)
                        if data:
                            st.session_state.training_data = data
                            method = "function docstrings" if len(data) >= 10 else "code-split fallback"
                            st.info(f"Detected code dataset (The Stack / GitHub format). Extracted **{len(data)}** training pairs using {method}.")
                        else:
                            st.error("Found a 'content' column but could not extract any training pairs. The code files may have no docstrings and be too short.")
                            return
                    else:
                        # --- Standard conversation dataset ---
                        user_cols = ['user', 'question', 'input', 'human', 'original_src', 'prompt', 'query', 'instruction', 'text']
                        bot_cols = ['bot', 'answer', 'output', 'assistant', 'changed_src', 'response', 'reply', 'completion', 'target']
                        
                        user_col = next((c for c in df.columns if c.lower() in user_cols), None)
                        bot_col = next((c for c in df.columns if c.lower() in bot_cols), None)
                        
                        if user_col and bot_col:
                            data = []
                            for _, row in df.iterrows():
                                u = str(row[user_col]).strip()
                                b = str(row[bot_col]).strip()
                                if u and b:
                                    data.append({'user': u, 'bot': b})
                            st.session_state.training_data = data
                        elif len(df.columns) == 2:
                            data = []
                            for _, row in df.iterrows():
                                u = str(row.iloc[0]).strip()
                                b = str(row.iloc[1]).strip()
                                if u and b:
                                    data.append({'user': u, 'bot': b})
                            st.session_state.training_data = data
                            st.info(f"Auto-mapped columns: **'{df.columns[0]}'** → User, **'{df.columns[1]}'** → Bot")
                        else:
                            st.error(
                                f"Could not auto-detect columns in this Parquet file.\n\n"
                                f"**Found columns:** {list(df.columns)}\n\n"
                                f"**Fix:** Rename your columns to `user` and `bot`, or make sure a `content` column exists for code datasets."
                            )
                            return
                
                st.success(f"Loaded {len(st.session_state.training_data)} conversation pairs!")
                
                # Show preview
                st.subheader("Data Preview")
                for i, conv in enumerate(st.session_state.training_data[:5]):
                    st.write(f"**Pair {i+1}:**")
                    st.write(f"👤 User: {conv['user']}")
                    st.write(f"🤖 Bot: {conv['bot']}")
                    st.write("---")
                
            except Exception as e:
                st.error(f"Error loading data: {str(e)}")
    
    else:  # Manual entry
        st.subheader("Enter Conversation Pairs")
        
        if 'manual_data' not in st.session_state:
            st.session_state.manual_data = []
        
        col1, col2 = st.columns(2)
        with col1:
            user_input = st.text_area("User message:", key="user_manual")
        with col2:
            bot_input = st.text_area("Bot response:", key="bot_manual")
        
        if st.button("Add Pair"):
            if user_input and bot_input:
                st.session_state.manual_data.append({'user': user_input, 'bot': bot_input})
                st.success("Pair added!")
                st.rerun()
        
        if st.session_state.manual_data:
            st.write(f"**Current pairs: {len(st.session_state.manual_data)}**")
            for i, conv in enumerate(st.session_state.manual_data):
                col1, col2, col3 = st.columns([4, 4, 1])
                with col1:
                    st.write(f"👤 {conv['user']}")
                with col2:
                    st.write(f"🤖 {conv['bot']}")
                with col3:
                    if st.button("❌", key=f"del_{i}"):
                        st.session_state.manual_data.pop(i)
                        st.rerun()
            
            if st.button("Use This Data"):
                st.session_state.training_data = st.session_state.manual_data
                st.success("Training data set!")

def model_setup_section():
    st.header("🏗️ Model Configuration")
    
    if st.session_state.training_data is None:
        st.warning("Please upload training data first!")
        return
    
    st.write(f"**Training pairs available:** {len(st.session_state.training_data)}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Tokenizer Settings")
        st.caption(
            "Vocabulary size is **at most** your *Maximum vocabulary size*, but usually it is the "
            "number of **unique word tokens** in your training text (plus padding if enabled). "
            "Seeing **~955** means your corpus used ~951 distinct words — not a bug. Padding adds "
            "common English so chat uses fewer `<UNK>` tokens."
        )
        max_vocab_size = st.number_input(
            "Maximum vocabulary size:",
            1000,
            50000,
            12000,
            500,
            help="Hard cap on rows in the embedding table. Corpus + padding cannot exceed this.",
        )
        pad_vocab = st.checkbox(
            "Pad with common English words (recommended if corpus vocab is small)",
            value=True,
            help="Merges ~4k frequent English tokens (from bundled data/common_words_en.txt) until "
                 "your vocab reaches the target below (capped by maximum vocabulary size).",
        )
        pad_target = st.number_input(
            "Target vocab size after padding",
            2000,
            50000,
            6000,
            500,
            disabled=not pad_vocab,
            help="Stops adding padding words once this size is reached (or max vocabulary size).",
        )

        if st.button("Estimate unique words in training text"):
            tok = ChatbotTokenizer(max_vocab_size=max_vocab_size)
            sample = st.session_state.training_data[:8000]
            if len(st.session_state.training_data) > 8000:
                st.caption("(Using first 8,000 pairs for speed — close estimate for huge sets.)")
            uq: set[str] = set()
            for conv in sample:
                uq.update(tok.tokenize(conv.get("user", "")))
                uq.update(tok.tokenize(conv.get("bot", "")))
            st.info(
                f"~**{len(uq):,}** unique word tokens in sample (specials not included). "
                f"Full **Build Vocabulary** may differ slightly if you pad or use a different slice."
            )

        if st.button("Build Vocabulary"):
            with st.spinner("Building vocabulary..."):
                all_texts = []
                for conv in st.session_state.training_data:
                    all_texts.append(conv["user"])
                    all_texts.append(conv["bot"])

                tokenizer = ChatbotTokenizer(max_vocab_size=max_vocab_size)
                pad_list = None
                pad_until = 0
                if pad_vocab:
                    from english_vocab_padding import get_padding_word_list

                    pad_list = get_padding_word_list()
                    pad_until = int(min(pad_target, max_vocab_size))

                tokenizer.build_vocabulary(
                    all_texts,
                    pad_with_words=pad_list if pad_vocab else None,
                    pad_until=pad_until if pad_vocab else 0,
                )
                st.session_state.tokenizer = tokenizer

                vocab_info = tokenizer.get_vocab_info()
                st.success(f"Vocabulary built! Size: {vocab_info['vocab_size']}")
                if pad_vocab and tokenizer.vocab_size < min(pad_target, max_vocab_size):
                    st.warning(
                        f"Vocabulary stopped at **{tokenizer.vocab_size}** before reaching your padding "
                        f"target **{min(pad_target, max_vocab_size)}** — the bundled word list ran out or "
                        f"overlapped heavily with your corpus. You can still raise **Maximum vocabulary size** "
                        f"and append more lines to **data/common_words_en.txt** if you need a larger table."
                    )
                if vocab_info["vocab_size"] < 2500:
                    st.warning(
                        "Vocab is still under **2,500** types — training may be easier (lower CE floor) "
                        "but generation can be brittle. Increase **Target vocab size after padding** or "
                        "**Maximum vocabulary size**, or add more diverse training text."
                    )

                st.write("**Vocabulary Info:**")
                st.json(vocab_info)
    
    with col2:
        st.subheader("Model Architecture")
        
        if st.session_state.tokenizer is None:
            st.info("Build vocabulary first")
        else:
            vocab_size = st.session_state.tokenizer.vocab_size
            st.write(f"**Vocabulary Size:** {vocab_size}")
            
            # Architecture selection
            st.write("### 🧠 Choose Architecture:")
            model_type = st.radio(
                "Model Type:",
                ["Transformer Decoder-Only (Ollama Recommended)", "Transformer (Encoder-Decoder, Legacy)", "RNN (Legacy, Basic)"],
                index=0,
                help="Use Decoder-Only Transformer for Ollama-compatible GGUF models. Older modes are kept for compatibility."
            )

            if "RNN" in model_type:
                st.warning("RNN is a legacy option. Transformer trains better, generates smarter responses, and scales to any size. Consider switching unless you have a specific reason to use RNN.")

            # GPU size guide — Transformer-first labels
            st.write("**🔥 GPU Size Guide (Transformer):**")
            st.write("- **CPU / No GPU**: embed=128, heads=4, layers=2, ff=512")
            st.write("- **GTX 1650 (4GB)**: embed=256, heads=8, layers=4, ff=1024")
            st.write("- **RTX 3060 (12GB)**: embed=512, heads=8, layers=6, ff=2048")
            st.write("- **RTX 4090 (24GB)**: embed=1024, heads=16, layers=12, ff=4096")
            st.write("- **RTX 5090 (32GB+)**: embed=2048, heads=16, layers=24, ff=8192")

            embedding_dim = st.number_input("Embedding dimension:", 32, 16384, 256, 32,
                                           help="Token embedding size. Must be divisible by number of heads for Transformer.")

            if "RNN" in model_type:
                # RNN parameters
                hidden_dim = st.number_input("Hidden dimension:", 64, 32768, 256, 64,
                                            help="RNN hidden state size.")
                max_length = st.number_input("Max sequence length:", 10, 500, 50, 10)
                learning_rate = st.number_input("Learning rate:", 0.001, 1.0, 0.05, 0.001, format="%.3f")

                # Calculate approximate parameters for RNN
                approx_params = (
                    vocab_size * embedding_dim +
                    embedding_dim * hidden_dim * 2 +
                    hidden_dim * hidden_dim * 2 +
                    hidden_dim * 4 +
                    hidden_dim * vocab_size +
                    vocab_size
                )
            else:
                # Transformer parameters
                num_heads = st.number_input("Attention heads:", 1, 32, 8, 1,
                                           help="More heads = richer attention. Must divide embedding_dim evenly.")
                num_layers = st.number_input("Layers:", 1, 24, 4, 1,
                                            help="Deeper = smarter. GPT-2 small uses 12 layers.")
                ff_dim = st.number_input("Feed-forward dimension:", 128, 65536, 1024, 128,
                                        help="Set this to 4x your embedding_dim for best results.")
                max_length = st.number_input(
                    "Max sequence length:",
                    10,
                    500,
                    128,
                    10,
                    help="Max positions in the decoder. Longer = more context and longer replies allowed, "
                         "but more VRAM. With a large dataset of long answers, use at least 128–256.",
                )
                learning_rate = st.number_input("Learning rate:", 0.000001, 0.1, 0.003, 0.000001, format="%.6f",
                                               help="0.003 is a good starting point. Go up to 0.01 if loss barely moves. Lower to 0.001 if loss jumps around.")

                st.write("**⚙️ Training Parameters**")
                tr_col1, tr_col2 = st.columns(2)
                with tr_col1:
                    t_optimizer = st.selectbox("Optimizer:", ["Adam", "AdamW", "SGD"],
                        help="Adam: best for most cases. AdamW: Adam + weight decay (less overfitting). SGD: simple gradient descent.")
                    t_scheduler = st.selectbox("LR Scheduler:", ["Warmup + Cosine", "Cosine", "Linear", "Warmup + Linear", "Constant"],
                        help="Controls how the learning rate changes over training. Warmup+Cosine is the gold standard.")
                    t_dropout = st.slider("Dropout rate:", 0.0, 0.5, 0.1, 0.05,
                        help="Randomly drops neurons during training to prevent overfitting. 0.1 is a good default. Use 0.0 for tiny datasets.")
                with tr_col2:
                    t_weight_decay = st.number_input("Weight decay:", 0.0, 1.0, 0.01, 0.001, format="%.4f",
                        help="L2 regularization. Used by AdamW. Higher = stronger regularization. Default 0.01.")
                    t_warmup_epochs = st.number_input("Warmup epochs:", 0, 50, 5, 1,
                        help="Gradually ramp up LR from 0 to full value over this many epochs. Prevents unstable early training.")
                    t_grad_clip = st.number_input("Gradient clip:", 0.1, 50.0, 5.0, 0.5,
                        help="Max gradient magnitude. Prevents exploding gradients. 5.0 is standard.")

                # map display names to internal names
                _opt_map   = {"Adam": "adam", "AdamW": "adamw", "SGD": "sgd"}
                _sched_map = {"Warmup + Cosine": "warmup_cosine", "Cosine": "cosine",
                              "Linear": "linear", "Warmup + Linear": "warmup_linear",
                              "Constant": "constant"}
                t_optimizer_key = _opt_map[t_optimizer]
                t_scheduler_key = _sched_map[t_scheduler]
                
                # Calculate approximate parameters for Transformer
                # Each attention layer has Wq, Wk, Wv, Wo (4 * embed_dim^2)
                # Each FF layer has 2 weight matrices (embed_dim * ff_dim * 2)
                # Layer norms, etc.
                attn_params_per_layer = 4 * (embedding_dim * embedding_dim)
                ff_params_per_layer = (embedding_dim * ff_dim) + (ff_dim * embedding_dim)
                encoder_params = num_layers * (attn_params_per_layer + ff_params_per_layer)
                decoder_params = num_layers * (2 * attn_params_per_layer + ff_params_per_layer)  # Self + cross attention
                
                approx_params = (
                    vocab_size * embedding_dim +  # Embedding
                    encoder_params +
                    decoder_params +
                    embedding_dim * vocab_size +  # Output layer
                    vocab_size  # Output bias
                )
            
            # Estimate VRAM usage (4 bytes per float32)
            vram_mb = (approx_params * 4) / (1024 * 1024)
            
            st.write(f"**📊 Estimated Parameters:** {approx_params:,} ({approx_params/1e6:.1f}M)")
            st.write(f"**💾 Estimated VRAM:** ~{vram_mb:.0f}MB (model only)")
            
            # Warnings for big models
            if approx_params > 100_000_000:
                st.warning(f"⚠️ {approx_params/1e6:.0f}M params - BIG model! Make sure you have enough VRAM!")
            if approx_params > 1_000_000_000:
                st.error(f"🔥 {approx_params/1e9:.1f}B params - MASSIVE! RTX 4090/5090 recommended!")
            
            if st.button("Create Model"):
                if "RNN" in model_type:
                    model = VnexAIChatbot(
                        vocab_size=vocab_size,
                        embedding_dim=embedding_dim,
                        hidden_dim=hidden_dim,
                        max_length=max_length,
                        learning_rate=learning_rate
                    )
                    st.session_state.model_type = "RNN"
                    
                    # Show actual model info
                    total_params = (
                        model.embedding.size +
                        model.Wxh_enc.size + model.Whh_enc.size + model.bh_enc.size +
                        model.Wxh_dec.size + model.Whh_dec.size + model.bh_dec.size +
                        model.Why.size + model.by.size
                    )
                else:
                    # Check if heads divide evenly
                    if embedding_dim % num_heads != 0:
                        st.error(f"Embedding dim ({embedding_dim}) must be divisible by num heads ({num_heads})!")
                        return
                    
                    decoder_only_mode = "Decoder-Only" in model_type
                    model = TransformerChatbot(
                        vocab_size=vocab_size,
                        embed_dim=embedding_dim,
                        num_heads=num_heads,
                        num_layers=num_layers,
                        ff_dim=ff_dim,
                        max_seq_len=max_length,
                        learning_rate=learning_rate,
                        optimizer=t_optimizer_key,
                        weight_decay=float(t_weight_decay),
                        scheduler=t_scheduler_key,
                        warmup_epochs=int(t_warmup_epochs),
                        dropout_rate=float(t_dropout),
                        grad_clip=float(t_grad_clip),
                        decoder_only=decoder_only_mode
                    )
                    st.session_state.model_type = "TransformerDecoderOnly" if decoder_only_mode else "Transformer"
                    
                    # Calculate actual params (done in model init)
                    total_params = model._count_parameters()
                
                st.session_state.chatbot_model = model
                st.session_state.is_trained = False
                arch_name = "RNN" if "RNN" in model_type else "Transformer"
                st.success(f"{arch_name} model created! Head to the Training tab to start training.")
                
                # Show device info (RNN has this method, Transformer has gpu_available)
                if hasattr(model, 'get_device_info'):
                    st.info(model.get_device_info())
                else:
                    st.info(f"GPU: {'✅ Enabled (CuPy)' if model.gpu_available else '❌ CPU Only (NumPy)'}")
                
                st.write(f"**✅ Actual Total Parameters:** {total_params:,} ({total_params/1e6:.1f}M)")
                
                # Show billion parameter milestone
                if total_params >= 1_000_000_000:
                    st.balloons()
                    st.success(f"🎉 CONGRATS! You built a {total_params/1e9:.2f}B parameter model!")

def training_section():
    st.header("Train your model")
    st.caption(
        "You are optimizing the **neural network weights** (the \"brain\"). The chat page is only one way to run inference."
    )
    
    if st.session_state.chatbot_model is None:
        st.warning("Please create a model first!")
        return

    td = st.session_state.training_data
    if td is None:
        st.warning("Load training data first (Data Upload page).")
        return

    model = st.session_state.chatbot_model
    if (
        st.session_state.get("training_data_profile") == "sft_pack"
        and getattr(model, "decoder_only", False)
    ):
        st.info(
            "You loaded the **SFT Pack**. Turn **SFT (Supervised Fine-Tuning)** on below so training "
            "loss applies mainly to **assistant (bot) tokens** — that is what this dataset is shaped for."
        )
    elif st.session_state.get("training_data_profile") == "sft_pack" and not getattr(
        model, "decoder_only", False
    ):
        st.warning(
            "The **SFT Pack** is meant for **decoder-only** models with **SFT** enabled. "
            "Your model is encoder–decoder or not decoder-only; consider switching architecture or use full LM loss."
        )

    # Data stats
    total_pairs = len(td)
    
    # Parquet support
    uploaded_file = st.sidebar.file_uploader("Upload Parquet Data", type=["parquet"])
    if uploaded_file:
        import pandas as pd
        try:
            df = pd.read_parquet(uploaded_file)
            col_lower_set = {c.lower() for c in df.columns}
            new_data = []

            if 'content' in col_lower_set:
                # The Stack / code-file format
                new_data = parse_code_parquet(df, max_pairs=2000)
                if new_data:
                    st.sidebar.success(f"Code dataset detected! Extracted {len(new_data)} pairs.")
                else:
                    st.sidebar.error("Found 'content' column but could not extract training pairs.")
            else:
                # Standard conversation columns
                user_cols = ['user', 'question', 'input', 'human', 'original_src', 'prompt', 'query', 'instruction', 'text']
                bot_cols = ['bot', 'answer', 'output', 'assistant', 'changed_src', 'response', 'reply', 'completion', 'target']
                user_col = next((c for c in df.columns if c.lower() in user_cols), None)
                bot_col = next((c for c in df.columns if c.lower() in bot_cols), None)

                if user_col and bot_col:
                    for _, row in df.iterrows():
                        u = str(row[user_col]).strip()
                        b = str(row[bot_col]).strip()
                        if u and b:
                            new_data.append({'user': u, 'bot': b})
                    st.sidebar.success(f"Loaded {len(new_data)} rows from Parquet!")
                elif len(df.columns) == 2:
                    for _, row in df.iterrows():
                        u = str(row.iloc[0]).strip()
                        b = str(row.iloc[1]).strip()
                        if u and b:
                            new_data.append({'user': u, 'bot': b})
                    st.sidebar.info(f"Auto-mapped 2 columns. {len(new_data)} pairs loaded.")
                else:
                    st.sidebar.error(f"Could not detect columns. Found: {list(df.columns)}")

            if new_data:
                if st.sidebar.button("Add Parquet Data to Training"):
                    if st.session_state.training_data is None:
                        st.session_state.training_data = []
                    st.session_state.training_data.extend(new_data)
                    st.sidebar.success(f"Added {len(new_data)} pairs!")
                    st.rerun()

        except Exception as e:
            st.sidebar.error(f"Error reading parquet: {e}")
    if total_pairs < 500:
        st.warning(f"⚠️ Small Dataset ({total_pairs} pairs). Transformers need at least 1,000–5,000 pairs to learn well. More data = smarter responses.")
    elif total_pairs < 1000:
        st.info(f"📊 {total_pairs} pairs — decent start! 1,000+ pairs will give noticeably better results.")
    else:
        st.success(f"📊 {total_pairs} conversation pairs loaded. Good dataset size!")

    st.write("### 💡 Transformer Training Tips:")
    with st.expander("How this compares to generic “lower loss” advice (e.g. normalize inputs, batch norm)"):
        st.markdown(
            """
            **What already matches “generic ML” tips here**
            - **Adam / AdamW / schedules / dropout / weight decay / grad clip** — available in Model Setup for Transformers.
            - **Train longer** — still one of the biggest levers if loss is slowly decreasing.

            **What is different for text token models**
            - **“Normalize inputs”** in vision means rescale pixels. Here, inputs are **token IDs** into an **embedding table**; there is no separate per-feature z-score on tokens.
            - **Batch Normalization** is common in CNNs; Transformers usually use **LayerNorm** inside blocks (not the same knob as “add BatchNorm layers”).
            - **Vocabulary size ~950** usually means **only ~950 distinct word types appeared in your training strings** (capped by max vocab), not that the slider failed. Use **Pad with common English** to grow the table for rarer chat words.

            **Loss scale**
            - Random guessing over \(V\) tokens has cross-entropy about **ln(V)**. A vocab of 6k has a higher “random” floor than a vocab of 1k — **lower vocab can make loss look better** while **chat still looks dumb** if everything maps to `<UNK>`.
            """
        )
    st.info("""
    - **Recommended starter settings:** embed=256, heads=8, layers=4, ff=1024, LR=0.003, Optimizer=Adam, Scheduler=Warmup+Cosine
    - **Adam vs SGD:** Adam typically drops loss 4–5× faster than SGD. Use Adam unless you have a specific reason not to.
    - **Loss stuck near starting value?** LR is too low — try 0.005 or 0.01.
    - **Loss bouncing wildly?** LR is too high — try 0.001. Or increase Dropout to 0.2.
    - **Loss plateaus early?** Train longer (100–200 epochs). Cosine scheduler keeps LR productive the whole time.
    - **Overfitting (loss drops but chat is bad)?** Increase dropout rate (0.2–0.3) and use AdamW with weight decay 0.01.
    - **Still word salad after 100 epochs?** You need more data — aim for 1,000+ conversation pairs.
    """)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        epochs = st.number_input("Number of epochs:", 1, 10000, 100, 5)
    with col2:
        batch_size = st.number_input("Batch size:", 1, 512, 16, 1, 
                                      help="GTX 1650: 32-64 | RTX 3060: 64-128 | RTX 4090: 128-256 | RTX 5090: 256-512")
    with col3:
        shuffle_data = st.checkbox("Shuffle data", value=True)

    use_sft = st.checkbox(
        "SFT (Supervised Fine-Tuning)",
        value=False,
        help="Decoder-only only: train loss only on the assistant (bot) reply tokens, not on "
             "predicting the user prompt. Matches common instruction-tuning / chat SFT. "
             "Encoder–decoder mode ignores this (loss is already on the bot sequence only).",
        disabled=not getattr(model, "decoder_only", False),
    )
    if use_sft and getattr(model, "decoder_only", False):
        st.caption("SFT on: gradients focus on learning to complete after the user segment.")
    elif not getattr(model, "decoder_only", False):
        st.caption("SFT applies to Decoder-Only models; your model is encoder–decoder (loss is on bot only).")
    
    # Show batch info
    if model.gpu_available:
        st.info(f"🚀 GPU Mode: Batch size {batch_size} will process {batch_size} samples simultaneously for maximum speed!")
    else:
        st.info(f"💻 CPU Mode: Using batch size {batch_size}")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    batch_status_text = st.empty()
    loss_chart_placeholder = st.empty()
    
    if st.button("Start Training"):
        tokenizer = st.session_state.tokenizer
        data = st.session_state.training_data
        
        losses = []
        
        # Prepare all batches first to get accurate count
        input_seqs_all = []
        target_seqs_all = []
        sft_starts_all = []

        if hasattr(model, "clear_update_coverage"):
            model.clear_update_coverage()

        max_len = int(model.max_length if hasattr(model, "max_length") else model.max_seq_len)

        for conv in data:
            if getattr(model, 'decoder_only', False):
                # Decoder-only training sequence:
                # <START> user <END> <START> bot <END>
                user_seq = tokenizer.encode(conv['user'], add_special_tokens=True)
                bot_seq = tokenizer.encode(conv['bot'], add_special_tokens=True)
                full_seq = np.array(user_seq + bot_seq, dtype=np.int32)
                input_seq = full_seq
                target_seq = full_seq
                # SFT: first loss timestep predicts first bot token (index len(user_seq) in full_seq)
                sft_t = (len(user_seq) - 1) if use_sft else None
            else:
                # Encoder-decoder path
                input_seq = np.array(tokenizer.encode(conv['user'], add_special_tokens=True))
                target_seq = np.array(tokenizer.encode(conv['bot'], add_special_tokens=True))
                sft_t = None

            # Decoder/encoder forward uses pos enc length max_len; target_input has len(seq)-1.
            # Require len(seq)-1 <= max_len  =>  len(seq) <= max_len + 1 (strict '< max_len' dropped valid rows).
            if len(input_seq) > 0 and len(target_seq) > 0 and len(target_seq) <= max_len + 1:
                input_seqs_all.append(input_seq)
                target_seqs_all.append(target_seq)
                sft_starts_all.append(sft_t)
        
        # Calculate batch info
        total_samples = len(input_seqs_all)
        skipped = len(data) - total_samples
        if skipped > 0:
            st.warning(
                f"**{skipped:,}** / **{len(data):,}** conversations were skipped (too long for "
                f"max sequence length **{max_len}**). Raise **Max sequence length** in Model Setup "
                "or shorten replies — otherwise the model barely sees your large dataset."
            )
        num_batches = (total_samples + batch_size - 1) // batch_size  # Ceiling division
        
        # Print training info to console
        print("\n" + "="*60)
        print("TRAINING STARTED")
        print("="*60)
        print(f"Total samples: {total_samples}")
        print(f"Batch size: {batch_size}")
        print(f"Batches per epoch: {num_batches}")
        print(f"Total epochs: {epochs}")
        print(f"Total batches to process: {num_batches * epochs}")
        if model.gpu_available:
            print("GPU Mode: ACTIVE (CuPy)")
        else:
            print("CPU Mode: ACTIVE (NumPy)")
        print("="*60 + "\n")
        
        # Training loop - removed spinner so progress bar works!
        for epoch in range(epochs):
            epoch_losses = []
            
            # Shuffle data
            if shuffle_data:
                indices = np.random.permutation(total_samples)
                input_seqs_shuffled = [input_seqs_all[i] for i in indices]
                target_seqs_shuffled = [target_seqs_all[i] for i in indices]
                sft_shuffled = [sft_starts_all[i] for i in indices]
            else:
                input_seqs_shuffled = input_seqs_all
                target_seqs_shuffled = target_seqs_all
                sft_shuffled = sft_starts_all
            
            print(f"\nEpoch {epoch + 1}/{epochs}")
            print("-" * 60)
            
            # Train in batches
            batch_num = 0
            for i in range(0, len(input_seqs_shuffled), batch_size):
                batch_inputs = input_seqs_shuffled[i:i+batch_size]
                batch_targets = target_seqs_shuffled[i:i+batch_size]
                batch_sft = sft_shuffled[i:i+batch_size] if use_sft and getattr(model, "decoder_only", False) else None
                
                if len(batch_inputs) > 0:
                    batch_num += 1
                    if batch_sft is not None:
                        loss = model.train_batch(
                            batch_inputs, batch_targets, sft_loss_starts=batch_sft
                        )
                    else:
                        loss = model.train_batch(batch_inputs, batch_targets)
                    epoch_losses.append(loss)
                    
                    # Print batch progress
                    print(f"  Batch {batch_num}/{num_batches} | Samples: {len(batch_inputs)} | Loss: {loss:.4f}")
                    batch_status_text.text(
                        f"Batch {batch_num}/{num_batches} (Epoch {epoch + 1}/{epochs}) - Batch Loss: {loss:.4f}"
                    )
            
            avg_loss = np.mean(epoch_losses) if epoch_losses else 0
            losses.append(avg_loss)

            # Decay LR once per epoch (both RNN and Transformer have step_lr)
            if hasattr(model, 'step_lr'):
                model.step_lr(total_epochs=epochs)

            # Print epoch summary
            current_lr = getattr(model, 'learning_rate', '?')
            print(f"  Epoch {epoch + 1} complete | Avg Loss: {avg_loss:.4f} | LR: {current_lr:.6f}")
            print("-" * 60)
            
            # Update progress
            progress = (epoch + 1) / epochs
            progress_bar.progress(progress)
            status_text.text(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f} | LR: {current_lr:.6f}")
            
            # Update chart every 10 epochs
            if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
                import plotly.graph_objects as go
                fig = go.Figure()
                fig.add_trace(go.Scatter(y=losses, mode='lines', name='Training Loss'))
                fig.update_layout(title="Training Loss", xaxis_title="Epoch", yaxis_title="Loss")
                loss_chart_placeholder.plotly_chart(fig, use_container_width=True, key=f"training_loss_{epoch}")
        
        model.training_history['loss'] = losses
        st.session_state.is_trained = True

        if hasattr(model, "get_update_coverage_report"):
            report = model.get_update_coverage_report()
            st.write("### Parameter Update Coverage")
            st.write(
                f"Updated {report['updated_count']}/{report['expected_count']} expected parameter groups "
                f"({report['coverage_percent']:.1f}%)."
            )
            if report["missing"]:
                st.warning(f"Missing updates: {', '.join(report['missing'][:12])}" +
                           (" ..." if len(report["missing"]) > 12 else ""))
        
        # Print final summary
        print("\n" + "="*60)
        print("TRAINING COMPLETE!")
        print("="*60)
        print(f"Total epochs completed: {epochs}")
        print(f"Total batches processed: {num_batches * epochs}")
        print(f"Final loss: {losses[-1]:.4f}")
        print(f"Starting loss: {losses[0]:.4f}")
        print(f"Loss improvement: {((losses[0] - losses[-1]) / losses[0] * 100):.1f}%")
        print("="*60 + "\n")
        
        st.success("Training complete!")

def chat_interface_section():
    st.header("💬 Chat with Your AI")
    
    if not st.session_state.is_trained:
        st.warning("Please train the model first!")
        return
    
    model = st.session_state.chatbot_model
    tokenizer = st.session_state.tokenizer
    
    # Temperature slider for creativity control
    st.subheader("⚙️ Generation Settings")
    temperature = st.slider(
        "Temperature (creativity):", 
        min_value=0.1, 
        max_value=2.0, 
        value=0.8, 
        step=0.1,
        help="Lower = more focused/repetitive, Higher = more creative/random. Try 0.5-1.0 for best results!"
    )
    
    if temperature < 0.5:
        st.info("🎯 Low temperature: Very focused, may repeat phrases")
    elif temperature > 1.5:
        st.info("🎲 High temperature: Very creative, may be random")
    else:
        st.info("✅ Good temperature: Balanced creativity")
    
    # Chat history display
    st.subheader("Conversation")
    chat_container = st.container()
    
    with chat_container:
        for msg in st.session_state.chat_history:
            if msg['role'] == 'user':
                st.markdown(f'<div class="chat-message user-message">👤 {msg["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-message bot-message">🤖 {msg["content"]}</div>', unsafe_allow_html=True)
    
    # Chat input
    col1, col2 = st.columns([5, 1])
    with col1:
        user_message = st.text_input("Your message:", key="chat_input")
    with col2:
        send_button = st.button("Send")
    
    if send_button and user_message:
        # Add user message to history
        st.session_state.chat_history.append({'role': 'user', 'content': user_message})
        
        # Generate response with anti-comma spam protection
        # Try generating up to 3 times if comma spam detected
        max_retries = 3
        response_text = ""
        
        for attempt in range(max_retries):
            # Check model type and use appropriate generate method
            if hasattr(model, 'generate_response'):
                # RNN model - use special tokens to match training
                input_seq = np.array(tokenizer.encode(user_message, add_special_tokens=True))
                response_indices = model.generate_response(input_seq, temperature=temperature)
                response_text = tokenizer.decode(response_indices.tolist())
            else:
                # Transformer model
                response_text = model.generate(user_message, tokenizer, temperature=temperature)
            
            # Check for 3+ consecutive commas or dots (spam detection)
            if ',,,' not in response_text and '...' not in response_text:
                # Good response, no spam!
                break
            else:
                # Spam detected! Increase temperature and retry
                spam_type = "comma" if ',,,' in response_text else "dot"
                if attempt < max_retries - 1:
                    print(f"Warning: {spam_type.capitalize()} spam detected (attempt {attempt + 1}), regenerating with higher temperature...")
                    temperature = min(temperature + 0.3, 2.0)  # Increase temperature
                else:
                    print(f"Warning: Still spam after {max_retries} attempts, using last response")
        
        # Add bot response to history
        st.session_state.chat_history.append({'role': 'bot', 'content': response_text})
        
        st.rerun()
    
    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

def export_model_section():
    st.header("💾 Export Your Chatbot")

    if st.session_state.chatbot_model is None:
        st.warning("No model to export!")
        return

    model = st.session_state.chatbot_model
    is_rnn = hasattr(model, 'hidden_dim')

    # ── Model name (used for ALL export filenames) ───────────────────────────
    model_name = st.text_input(
        "Model name:",
        value="vnexai_chatbot",
        help="This name is used for all downloaded files. Change it to whatever you like!"
    )
    st.caption(f"Files will be saved as: `{model_name}.bin`, `{model_name}.gguf`, `{model_name}_tokenizer.bin`")

    st.divider()

    # ── Three export columns ─────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)

    # ── 1. .bin export (original pickle format) ──────────────────────────────
    with col1:
        st.subheader("📦 .bin Format")
        st.caption("VnexAI native format — load it back with the VnexAI Python API.")
        if st.button("Prepare Model (.bin)", key="export_model_btn"):
            import tempfile, os
            with tempfile.NamedTemporaryFile(delete=False, suffix='.bin') as tmp:
                tmp_path = tmp.name
            try:
                if hasattr(model, 'save_model'):
                    model.save_model(tmp_path)
                else:
                    model.save(tmp_path)
                with open(tmp_path, 'rb') as f:
                    st.session_state.model_bytes = f.read()
                st.success("Ready!")
            except Exception as e:
                st.error(f"Export failed: {e}")
            finally:
                try:
                    os.unlink(tmp_path)
                except:
                    pass

        if st.session_state.get('model_bytes'):
            st.download_button(
                label="📥 Download .bin",
                data=st.session_state.model_bytes,
                file_name=f"{model_name}.bin",
                mime="application/octet-stream",
                key="download_model_btn"
            )

    # ── 2. .gguf export ───────────────────────────────────────────────────────
    with col2:
        st.subheader("🦙 .gguf Format")
        if is_rnn:
            st.caption("GGUF binary for archiving/llama.cpp tooling (RNN archive format).")
        else:
            if getattr(model, 'decoder_only', False):
                st.caption("Ollama-compatible GGUF — decoder-only `llama` architecture with embedded vocabulary.")
            else:
                st.caption("Best-effort GGUF export from encoder-decoder Transformer. For stable Ollama runtime, train Decoder-Only mode.")
            st.info("✅ This GGUF can be loaded directly by Ollama using the Modelfile below.", icon="✅")

        if st.button("Prepare Model (.gguf)", key="export_gguf_btn"):
            try:
                from gguf_writer import export_rnn_to_gguf, export_transformer_to_gguf
                tokenizer = st.session_state.get('tokenizer')
                with st.spinner("Building GGUF file…"):
                    if is_rnn:
                        gguf_bytes = export_rnn_to_gguf(model, model_name)
                    else:
                        if tokenizer is None:
                            st.error("Tokenizer not found in session. Please train a model first.")
                            return
                        if not getattr(model, 'decoder_only', False):
                            st.error("For reliable Ollama runtime, export only Decoder-Only Transformer models.")
                            return
                        gguf_bytes = export_transformer_to_gguf(model, tokenizer, model_name)
                st.session_state.gguf_bytes = gguf_bytes
                size_mb = len(gguf_bytes) / (1024 * 1024)
                st.success(f"Ready! ({size_mb:.1f} MB)")
            except Exception as e:
                st.error(f"GGUF export failed: {e}")

        if st.session_state.get('gguf_bytes'):
            st.download_button(
                label="📥 Download .gguf",
                data=st.session_state.gguf_bytes,
                file_name=f"{model_name}.gguf",
                mime="application/octet-stream",
                key="download_gguf_btn"
            )

    # ── 3. Tokenizer .bin export ──────────────────────────────────────────────
    with col3:
        st.subheader("📝 Tokenizer (.bin)")
        st.caption("Save the vocabulary so you can decode outputs later.")
        if st.button("Prepare Tokenizer (.bin)", key="export_tokenizer_btn"):
            if st.session_state.tokenizer:
                import tempfile, os
                with tempfile.NamedTemporaryFile(delete=False, suffix='.bin') as tmp:
                    tmp_path = tmp.name
                try:
                    st.session_state.tokenizer.save(tmp_path)
                    with open(tmp_path, 'rb') as f:
                        st.session_state.tokenizer_bytes = f.read()
                    st.success("Ready!")
                except Exception as e:
                    st.error(f"Export failed: {e}")
                finally:
                    try:
                        os.unlink(tmp_path)
                    except:
                        pass
            else:
                st.error("No tokenizer found — build the vocabulary first.")

        if st.session_state.get('tokenizer_bytes'):
            st.download_button(
                label="📥 Download Tokenizer.bin",
                data=st.session_state.tokenizer_bytes,
                file_name=f"{model_name}_tokenizer.bin",
                mime="application/octet-stream",
                key="download_tokenizer_btn"
            )

    st.divider()

    # ── Ollama Modelfile & Push Guide ─────────────────────────────────────────
    st.subheader("🦙 Publish to Ollama Registry")
    st.write(
        "Follow these steps to share your model on **ollama.com** so anyone can "
        "`ollama pull` and run it."
    )

    with st.expander("⚙️ Modelfile Settings", expanded=True):
        ollama_username = st.text_input(
            "Your Ollama username:",
            value="yourusername",
            help="Your username on ollama.com — needed for the push command."
        )
        system_prompt = st.text_area(
            "System prompt (what your bot says about itself):",
            value=f"You are {model_name}, a custom AI assistant trained with VnexAI. Be helpful, friendly, and concise.",
            height=80
        )
        num_ctx = st.number_input(
            "Context length (num_ctx):",
            min_value=64, max_value=4096, value=512, step=64,
            help="How many tokens of context the model uses. Keep it ≤ your max_seq_len."
        )
        temperature_val = st.slider(
            "Default temperature:", 0.1, 2.0, 0.8, 0.1,
            help="Same temperature slider as the Chat Interface."
        )
        extra_stop = st.text_input(
            "Extra stop token (optional):",
            value="",
            help="e.g. </s> or [END]. <END> is always included automatically."
        )

    # Build the Modelfile text
    stop_lines = 'PARAMETER stop "<END>"'
    if extra_stop.strip():
        stop_lines += f'\nPARAMETER stop "{extra_stop.strip()}"'

    arch_note = (
        f"# Architecture : {'RNN encoder-decoder' if is_rnn else 'Transformer encoder-decoder'}\n"
        f"# Trained with : VnexAI (NumPy/CuPy from scratch)\n"
        f"# Vocab size   : {model.vocab_size}\n"
    )
    if is_rnn:
        arch_note += f"# Hidden dim   : {model.hidden_dim}\n"
    else:
        arch_note += f"# Layers/Heads : {model.num_layers} / {model.num_heads}\n"

    modelfile_text = f"""FROM ./{model_name}.gguf

{arch_note}
SYSTEM \"\"\"{system_prompt}\"\"\"

PARAMETER temperature {temperature_val}
PARAMETER num_ctx {num_ctx}
{stop_lines}

TEMPLATE \"\"\"{{{{ .Prompt }}}}\"\"\"
"""

    st.subheader("📄 Generated Modelfile")
    st.code(modelfile_text, language="dockerfile")

    modelfile_bytes = modelfile_text.encode("utf-8")
    st.download_button(
        label="📥 Download Modelfile",
        data=modelfile_bytes,
        file_name="Modelfile",
        mime="text/plain",
        key="download_modelfile_btn"
    )

    st.subheader("🚀 Step-by-Step: Push to Ollama")

    full_model_tag = f"{ollama_username}/{model_name}"

    st.markdown("**1. Make sure Ollama is installed on your computer**")
    st.code("curl -fsSL https://ollama.com/install.sh | sh", language="bash")

    st.markdown("**2. Put these two files in the same folder:**")
    st.code(f"{model_name}.gguf\nModelfile", language="text")

    st.markdown("**3. Create the model locally from your GGUF + Modelfile:**")
    st.code(f"ollama create {full_model_tag} -f Modelfile", language="bash")

    st.markdown("**4. Test it runs on your machine:**")
    st.code(f"ollama run {full_model_tag}", language="bash")

    st.markdown("**5. Log in to Ollama (first time only):**")
    st.code("ollama login", language="bash")

    st.markdown("**6. Push to the Ollama registry so others can download it:**")
    st.code(f"ollama push {full_model_tag}", language="bash")

    st.markdown("**7. Share this command with others:**")
    st.code(f"ollama run {full_model_tag}", language="bash")

    st.info(
        f"Once pushed, your model will be visible at: **https://ollama.com/{full_model_tag}**\n\n"
        "Anyone with Ollama installed can then run it with the command above — "
        "no Python or VnexAI required on their end."
    )

    st.divider()

    # ── Model info panel ──────────────────────────────────────────────────────
    st.subheader("Model Information")
    if is_rnn:
        info = {
            "Model Type": "RNN (Recurrent Neural Network)",
            "Model Name": model_name,
            "Vocabulary Size": model.vocab_size,
            "Embedding Dimension": model.embedding_dim,
            "Hidden Dimension": model.hidden_dim,
            "Max Sequence Length": model.max_length,
            "Learning Rate": model.learning_rate,
            "Trained": "Yes" if st.session_state.is_trained else "No",
        }
    else:
        transformer_mode = "Decoder-Only (Ollama-ready)" if getattr(model, 'decoder_only', False) else "Encoder-Decoder"
        info = {
            "Model Type": f"Transformer ({transformer_mode})",
            "Model Name": model_name,
            "Vocabulary Size": model.vocab_size,
            "Embedding Dimension": model.embed_dim,
            "Number of Attention Heads": model.num_heads,
            "Number of Layers": model.num_layers,
            "Feed-Forward Dimension": model.ff_dim,
            "Max Sequence Length": model.max_seq_len,
            "Learning Rate": model.learning_rate,
            "Trained": "Yes" if st.session_state.is_trained else "No",
        }
    st.json(info)

    if st.session_state.is_trained and model.training_history['loss']:
        st.write(f"**Final Training Loss:** {model.training_history['loss'][-1]:.4f}")

if __name__ == "__main__":
    main()
