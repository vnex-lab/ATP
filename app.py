import streamlit as st
import numpy as np
import json
import io
from chatbot_model import VnexAIChatbot
from chatbot_tokenizer import ChatbotTokenizer

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
    }
    .bot-message {
        background-color: #f5f5f5;
        text-align: left;
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
    - **JSON**: Conversation format or code debugging format
    - **CSV**: Two columns (user, bot) or (question, answer) or (input, output)
    - **Text**: Line-by-line with separators (|, →, -, :, tab)
    - **Numbered JSON**: Format like `0:{...} 1:{...}`
    """)
    
    upload_type = st.radio("Choose upload method:", ["Upload File", "Enter Text Manually"])
    
    if upload_type == "Upload File":
        uploaded_file = st.file_uploader("Upload conversation data", type=['json', 'txt', 'csv', 'tsv', 'jsonl'])
        
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
        max_vocab_size = st.number_input("Maximum vocabulary size:", 1000, 50000, 10000, 1000)
        
        if st.button("Build Vocabulary"):
            with st.spinner("Building vocabulary..."):
                # Collect all texts
                all_texts = []
                for conv in st.session_state.training_data:
                    all_texts.append(conv['user'])
                    all_texts.append(conv['bot'])
                
                # Build tokenizer
                tokenizer = ChatbotTokenizer(max_vocab_size=max_vocab_size)
                tokenizer.build_vocabulary(all_texts)
                st.session_state.tokenizer = tokenizer
                
                vocab_info = tokenizer.get_vocab_info()
                st.success(f"Vocabulary built! Size: {vocab_info['vocab_size']}")
                
                # Show vocabulary info
                st.write("**Vocabulary Info:**")
                st.json(vocab_info)
    
    with col2:
        st.subheader("Model Architecture")
        
        if st.session_state.tokenizer is None:
            st.info("Build vocabulary first")
        else:
            vocab_size = st.session_state.tokenizer.vocab_size
            st.write(f"**Vocabulary Size:** {vocab_size}")
            
            embedding_dim = st.number_input("Embedding dimension:", 32, 512, 128, 32)
            hidden_dim = st.number_input("Hidden dimension:", 64, 1024, 256, 64)
            max_length = st.number_input("Max sequence length:", 10, 100, 50, 10)
            learning_rate = st.number_input("Learning rate:", 0.001, 0.1, 0.01, 0.001, format="%.3f")
            
            if st.button("Create Model"):
                model = VnexAIChatbot(
                    vocab_size=vocab_size,
                    embedding_dim=embedding_dim,
                    hidden_dim=hidden_dim,
                    max_length=max_length,
                    learning_rate=learning_rate
                )
                st.session_state.chatbot_model = model
                st.session_state.is_trained = False
                st.success("Chatbot model created!")
                
                # Show device info
                st.info(model.get_device_info())
                
                # Show model info
                total_params = (
                    model.embedding.size +
                    model.Wxh_enc.size + model.Whh_enc.size + model.bh_enc.size +
                    model.Wxh_dec.size + model.Whh_dec.size + model.bh_dec.size +
                    model.Why.size + model.by.size
                )
                st.write(f"**Total parameters:** {total_params:,}")

def training_section():
    st.header("🎯 Train Your Chatbot")
    
    if st.session_state.chatbot_model is None:
        st.warning("Please create a model first!")
        return
    
    if st.session_state.tokenizer is None or st.session_state.training_data is None:
        st.warning("Please set up tokenizer and training data first!")
        return
    
    col1, col2, col3 = st.columns(3)
    with col1:
        epochs = st.number_input("Number of epochs:", 1, 1000, 100, 10)
    with col2:
        batch_size = st.number_input("Batch size:", 1, 128, 32, 1, 
                                      help="Higher batch size = faster GPU training! Try 32-64 for your GTX 1650")
    with col3:
        shuffle_data = st.checkbox("Shuffle data", value=True)
    
    # Show batch info
    model = st.session_state.chatbot_model
    if model.gpu_available:
        st.info(f"🚀 GPU Mode: Batch size {batch_size} will process {batch_size} samples simultaneously for maximum speed!")
    else:
        st.info(f"💻 CPU Mode: Using batch size {batch_size}")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    loss_chart_placeholder = st.empty()
    
    if st.button("Start Training"):
        tokenizer = st.session_state.tokenizer
        data = st.session_state.training_data
        
        losses = []
        
        # Prepare all batches first to get accurate count
        input_seqs_all = []
        target_seqs_all = []
        
        for conv in data:
            input_seq = np.array(tokenizer.encode(conv['user'], add_special_tokens=False))
            target_seq = np.array(tokenizer.encode(conv['bot'], add_special_tokens=False))
            
            if len(input_seq) > 0 and len(target_seq) > 0 and len(target_seq) < model.max_length:
                input_seqs_all.append(input_seq)
                target_seqs_all.append(target_seq)
        
        # Calculate batch info
        total_samples = len(input_seqs_all)
        num_batches = (total_samples + batch_size - 1) // batch_size  # Ceiling division
        
        # Print training info to console
        print("\n" + "="*60)
        print("🚀 TRAINING STARTED")
        print("="*60)
        print(f"📊 Total samples: {total_samples}")
        print(f"📦 Batch size: {batch_size}")
        print(f"🔢 Batches per epoch: {num_batches}")
        print(f"🔄 Total epochs: {epochs}")
        print(f"💪 Total batches to process: {num_batches * epochs}")
        if model.gpu_available:
            print(f"🚀 GPU Mode: ACTIVE (CuPy)")
        else:
            print(f"💻 CPU Mode: ACTIVE (NumPy)")
        print("="*60 + "\n")
        
        # Training loop - removed spinner so progress bar works!
        for epoch in range(epochs):
            epoch_losses = []
            
            # Shuffle data
            if shuffle_data:
                indices = np.random.permutation(total_samples)
                input_seqs_shuffled = [input_seqs_all[i] for i in indices]
                target_seqs_shuffled = [target_seqs_all[i] for i in indices]
            else:
                input_seqs_shuffled = input_seqs_all
                target_seqs_shuffled = target_seqs_all
            
            print(f"\n📍 Epoch {epoch + 1}/{epochs}")
            print("-" * 60)
            
            # Train in batches
            batch_num = 0
            for i in range(0, len(input_seqs_shuffled), batch_size):
                batch_inputs = input_seqs_shuffled[i:i+batch_size]
                batch_targets = target_seqs_shuffled[i:i+batch_size]
                
                if len(batch_inputs) > 0:
                    batch_num += 1
                    loss = model.train_batch(batch_inputs, batch_targets)
                    epoch_losses.append(loss)
                    
                    # Print batch progress
                    print(f"  Batch {batch_num}/{num_batches} | Samples: {len(batch_inputs)} | Loss: {loss:.4f}")
            
            avg_loss = np.mean(epoch_losses) if epoch_losses else 0
            losses.append(avg_loss)
            
            # Print epoch summary
            print(f"  ✅ Epoch {epoch + 1} complete | Avg Loss: {avg_loss:.4f}")
            print("-" * 60)
            
            # Update progress
            progress = (epoch + 1) / epochs
            progress_bar.progress(progress)
            status_text.text(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}")
            
            # Update chart every 10 epochs
            if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
                import plotly.graph_objects as go
                fig = go.Figure()
                fig.add_trace(go.Scatter(y=losses, mode='lines', name='Training Loss'))
                fig.update_layout(title="Training Loss", xaxis_title="Epoch", yaxis_title="Loss")
                loss_chart_placeholder.plotly_chart(fig, use_container_width=True, key=f"training_loss_{epoch}")
        
        model.training_history['loss'] = losses
        st.session_state.is_trained = True
        
        # Print final summary
        print("\n" + "="*60)
        print("✅ TRAINING COMPLETE!")
        print("="*60)
        print(f"📊 Total epochs completed: {epochs}")
        print(f"📦 Total batches processed: {num_batches * epochs}")
        print(f"📉 Final loss: {losses[-1]:.4f}")
        print(f"📈 Starting loss: {losses[0]:.4f}")
        print(f"💪 Loss improvement: {((losses[0] - losses[-1]) / losses[0] * 100):.1f}%")
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
        
        # Generate response with temperature
        input_seq = np.array(tokenizer.encode(user_message, add_special_tokens=False))
        response_indices = model.generate_response(input_seq, temperature=temperature)
        response_text = tokenizer.decode(response_indices.tolist())
        
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
    
    st.write("""
    Export your trained VnexAI chatbot model as a .bin file. This file contains:
    - All trained weights and biases
    - Model architecture parameters
    - Training history
    """)
    
    model_name = st.text_input("Model name:", value="vnexai_chatbot")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Export Model (.bin)"):
            model = st.session_state.chatbot_model
            
            # Save to bytes
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix='.bin') as tmp:
                model.save_model(tmp.name)
                with open(tmp.name, 'rb') as f:
                    model_bytes = f.read()
            
            st.download_button(
                label="Download Model.bin",
                data=model_bytes,
                file_name=f"{model_name}.bin",
                mime="application/octet-stream"
            )
            st.success("Model ready for download!")
    
    with col2:
        if st.button("Export Tokenizer (.bin)"):
            if st.session_state.tokenizer:
                tokenizer = st.session_state.tokenizer
                
                # Save to bytes
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix='.bin') as tmp:
                    tokenizer.save(tmp.name)
                    with open(tmp.name, 'rb') as f:
                        tokenizer_bytes = f.read()
                
                st.download_button(
                    label="Download Tokenizer.bin",
                    data=tokenizer_bytes,
                    file_name=f"{model_name}_tokenizer.bin",
                    mime="application/octet-stream"
                )
                st.success("Tokenizer ready for download!")
    
    # Show model info
    if st.session_state.chatbot_model:
        st.subheader("Model Information")
        model = st.session_state.chatbot_model
        
        info = {
            "Vocabulary Size": model.vocab_size,
            "Embedding Dimension": model.embedding_dim,
            "Hidden Dimension": model.hidden_dim,
            "Max Sequence Length": model.max_length,
            "Learning Rate": model.learning_rate,
            "Trained": "Yes" if st.session_state.is_trained else "No"
        }
        
        st.json(info)
        
        if st.session_state.is_trained and model.training_history['loss']:
            st.write(f"**Final Training Loss:** {model.training_history['loss'][-1]:.4f}")

if __name__ == "__main__":
    main()
