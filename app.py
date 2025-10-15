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
    st.markdown("Train your own conversational AI from scratch and export as .bin!")
    
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
    Upload your conversation training data. Supported formats:
    - **JSON**: `[{"user": "Hello", "bot": "Hi there!"}, ...]`
    - **Text**: Each line as `user: Hello | bot: Hi there!`
    """)
    
    upload_type = st.radio("Choose upload method:", ["Upload File", "Enter Text Manually"])
    
    if upload_type == "Upload File":
        uploaded_file = st.file_uploader("Upload conversation data", type=['json', 'txt'])
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.json'):
                    data = json.load(uploaded_file)
                    st.session_state.training_data = data
                else:  # txt file
                    lines = uploaded_file.read().decode('utf-8').split('\n')
                    data = []
                    for line in lines:
                        if '|' in line:
                            parts = line.split('|')
                            if len(parts) == 2:
                                user_text = parts[0].replace('user:', '').strip()
                                bot_text = parts[1].replace('bot:', '').strip()
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
    
    col1, col2 = st.columns(2)
    with col1:
        epochs = st.number_input("Number of epochs:", 1, 1000, 100, 10)
    with col2:
        shuffle_data = st.checkbox("Shuffle data", value=True)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    loss_chart_placeholder = st.empty()
    
    if st.button("Start Training"):
        model = st.session_state.chatbot_model
        tokenizer = st.session_state.tokenizer
        data = st.session_state.training_data
        
        losses = []
        
        with st.spinner("Training in progress..."):
            for epoch in range(epochs):
                epoch_losses = []
                
                # Shuffle data
                if shuffle_data:
                    indices = np.random.permutation(len(data))
                    data_shuffled = [data[i] for i in indices]
                else:
                    data_shuffled = data
                
                # Train on each pair
                for conv in data_shuffled:
                    # Encode sequences
                    input_seq = np.array(tokenizer.encode(conv['user'], add_special_tokens=False))
                    target_seq = np.array(tokenizer.encode(conv['bot'], add_special_tokens=False))
                    
                    # Skip if too long
                    if len(input_seq) > 0 and len(target_seq) > 0 and len(target_seq) < model.max_length:
                        loss = model.train_step(input_seq, target_seq)
                        epoch_losses.append(loss)
                
                avg_loss = np.mean(epoch_losses) if epoch_losses else 0
                losses.append(avg_loss)
                
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
            st.success("Training complete!")

def chat_interface_section():
    st.header("💬 Chat with Your AI")
    
    if not st.session_state.is_trained:
        st.warning("Please train the model first!")
        return
    
    model = st.session_state.chatbot_model
    tokenizer = st.session_state.tokenizer
    
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
        
        # Generate response
        input_seq = np.array(tokenizer.encode(user_message, add_special_tokens=False))
        response_indices = model.generate_response(input_seq)
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
