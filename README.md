# VnexAI Chatbot - Train Your Own AI from Scratch

Train custom chatbot models from scratch using NumPy and export as .bin files!

## Features

✅ **Train from scratch** - Build your own AI chatbot with custom neural networks  
✅ **GPU Acceleration** - Automatically uses GPU (CuPy) if available, falls back to CPU (NumPy)  
✅ **Multiple data formats** - JSON, CSV, TSV, JSONL, text files with various separators  
✅ **Export as .bin** - Download trained model and tokenizer as binary files  
✅ **Chat interface** - Test your chatbot in real-time  
✅ **Flexible architecture** - Pure NumPy/CuPy implementation (educational & transparent)

## Supported Data Formats

### 1. Conversation Format (JSON)
```json
[
  {"user": "Hello", "bot": "Hi there!"},
  {"user": "How are you?", "bot": "I'm doing great!"}
]
```

### 2. Code Debugging Format (JSON)
```json
[
  {
    "language": "Python",
    "original_status": "Runtime Error",
    "original_src": "buggy code here...",
    "changed_src": "fixed code here...",
    "problem_id": "p123",
    "error": "0"
  }
]
```

### 3. Numbered JSON Format (Text File)
```
0:{"original_src": "...", "changed_src": "...", "language": "Python"}
1:{"original_src": "...", "changed_src": "...", "language": "C++"}
```

### 4. Line-by-Line Format (Text File)
Supports multiple separators: `|`, `→`, `-`, tab
```
user: Hello | bot: Hi there!
question: How are you? → answer: I'm great!
input: What's your name? - output: I'm VnexAI!
Q: Tell me a joke       A: Why did the chicken cross the road?
```

### 5. CSV/TSV Format
Headers can be: user/bot, question/answer, input/output, q/a, prompt/response, etc.
```csv
user,bot
Hello,Hi there!
How are you?,I'm great!
```

### 6. JSONL (JSON Lines) Format
```jsonl
{"user": "Hello", "bot": "Hi there!"}
{"user": "How are you?", "bot": "I'm great!"}
```

## Quick Start

### 1. Upload Your Data
- Navigate to **Data Upload**
- Choose your file format (JSON or text)
- Upload your training data
- The app will auto-detect the format and convert it

### 2. Setup Model
- Navigate to **Model Setup**
- Click **Build Vocabulary** to create word dictionary
- Configure model parameters:
  - Embedding dimension (32 to **16,384** - supports RTX 5090!)
  - Hidden dimension (64 to **32,768** - massive models!)
  - Learning rate (default: 0.01)
- **GPU Size Guide:**
  - GTX 1650 (4GB): embed=512, hidden=1024
  - RTX 3060 (12GB): embed=1024, hidden=2048
  - RTX 4090 (24GB): embed=4096, hidden=8192
  - RTX 5090 (32GB+): embed=8192, hidden=16384+
- Click **Create Model**
- App shows **parameter count** and **VRAM estimate**!

### 3. Train
- Navigate to **Training**
- Set number of epochs (start with 50-100)
- **Set batch size** (32-64 recommended for GPU, higher = faster!)
- Click **Start Training**
- Watch the loss decrease over time

### 4. Test Your Chatbot
- Navigate to **Chat Interface**
- Type messages and see responses
- Test how well your model learned

### 5. Export Model
- Navigate to **Export Model**
- Download **Model.bin** (contains all weights and architecture)
- Download **Tokenizer.bin** (contains vocabulary)
- Use these files with other platforms or applications

## Using Your Code Debugging Data

If you have code debugging data (like the provided example), the app will:

1. **Auto-detect** the format (original_src → changed_src)
2. **Convert** to conversational format:
   - User: "Fix this [language] code with [error]: [buggy code]"
   - Bot: "[fixed code]"
3. **Train** the chatbot to learn code fixing patterns
4. **Export** as .bin for deployment

## GPU Acceleration

VnexAI automatically detects and uses GPU when available:

- **GPU Mode**: Uses CuPy for GPU-accelerated training (much faster!)
- **CPU Mode**: Falls back to NumPy if no GPU is available
- **Automatic**: No configuration needed - just run the app!

When you create a model, the app will show:
- 🚀 **GPU Acceleration ACTIVE** - Training on GPU
- 💻 **CPU Mode** - Training on CPU

Training on GPU can be **10-100x faster** depending on your hardware!

### Batch Training for Maximum Speed

**What is Batch Training?**
Instead of training one conversation at a time, batch training processes multiple samples simultaneously on your GPU!

**Batch Size Recommendations:**
- **GTX 1650 (4GB)**: Use batch size 32-64
- **RTX 3060 (12GB)**: Use batch size 64-128
- **RTX 4090 (24GB)**: Use batch size 128-256
- **RTX 5090 (32GB+)**: Use batch size 256-512
- **CPU Mode**: Use batch size 8-16

**Why Batch Training Matters:**
- ⚡ **10-50x faster** on GPU vs one-by-one training
- 🔥 **Maxes out GPU usage** - keeps your GPU busy 100% of the time
- 💪 **Better gradient estimates** - more stable training

**Example Speed Improvement:**
- Without batches: 200MB VRAM, slow training (GPU idle 90% of time)
- With batch size 32: 2GB VRAM, **30x faster** (GPU at 100%)!

### Fixing GPU Issues on Windows

If you have a GPU but the app shows "CPU Mode", you might be missing CUDA components:

**Error**: `DLL load failed while importing curand`

**Solution**:
1. **Install CUDA Toolkit**: Download from [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads)
   - Choose CUDA 12.x version (matches cupy-cuda12x)
   - Make sure to install the full toolkit, not just drivers

2. **Verify Installation**:
   - Open Command Prompt and run: `nvcc --version`
   - Should show CUDA version 12.x

3. **Reinstall CuPy** (if needed):
   ```bash
   pip uninstall cupy-cuda12x
   pip install cupy-cuda12x
   ```

4. **Restart the app** - it should now detect and use your GPU!

**Note**: Your GPU must be NVIDIA with CUDA support. AMD/Intel GPUs won't work with CuPy.

## Model Architecture

- **Type**: Sequence-to-sequence RNN (Encoder-Decoder)
- **Encoder**: Processes input text into hidden representation
- **Decoder**: Generates response token by token
- **Implementation**: Pure NumPy/CuPy (GPU support included)
- **Export Format**: Pickle binary (.bin files)

## File Outputs

After training, you can download:

1. **vnexai_chatbot.bin**
   - Model weights (embedding, encoder, decoder, output layers)
   - Architecture parameters
   - Training history

2. **vnexai_chatbot_tokenizer.bin**
   - Vocabulary (word ↔ index mapping)
   - Special tokens
   - Vocabulary size

## Tips for Better Results

1. **More data = better results** - Aim for 100+ training pairs
2. **Consistent format** - Make sure your data is clean and consistent
3. **Longer training** - More epochs (100-500) for complex patterns
4. **Adjust hidden dimension** - Larger for complex tasks (256-512)
5. **Test regularly** - Use chat interface to monitor progress

## Technical Details

- Built with: Streamlit, NumPy
- Model: Custom RNN encoder-decoder
- Training: Backpropagation through time
- Loss: Cross-entropy
- Optimization: Gradient descent

## Deployment

The exported .bin files can be loaded in:
- Other Python applications (using pickle)
- Custom inference servers
- Chatbot platforms that support custom models

To load the model in Python:
```python
import pickle
from chatbot_model import VnexAIChatbot
from chatbot_tokenizer import ChatbotTokenizer

# Load model
model = VnexAIChatbot(vocab_size=0, embedding_dim=128, hidden_dim=256)
model.load_model('vnexai_chatbot.bin')

# Load tokenizer
tokenizer = ChatbotTokenizer()
tokenizer.load('vnexai_chatbot_tokenizer.bin')

# Generate response
input_text = "Hello"
input_seq = tokenizer.encode(input_text, add_special_tokens=False)
response_indices = model.generate_response(np.array(input_seq))
response = tokenizer.decode(response_indices.tolist())
print(response)
```

## Need Help?

- Check the Data Preview to ensure your data loaded correctly
- Monitor training loss - it should decrease over epochs
- If responses are poor, try more training epochs or more data
- For code debugging, ensure original_src and changed_src are present

Happy training! 🤖
