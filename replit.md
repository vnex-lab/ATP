# VnexAI - Custom Chatbot Training Platform

## Overview

VnexAI is a custom chatbot training platform built from scratch using NumPy/CuPy, with a Streamlit-based web interface. The application enables users to train conversational AI models using either RNN or Transformer architectures. Users can upload conversation data or code debugging data, choose between RNN (fast, basic) or Transformer (smart, attention-based like ChatGPT), configure the neural network architecture, train the model from scratch with GPU acceleration, test it in a chat interface with temperature control, and export the trained model as a .bin file for deployment.

## User Preferences

Preferred communication style: Simple, everyday language.

## Recent Changes

### April 2026 - Bug Fixes & The Stack Parquet Support

**Critical RNN Generation Bug Fixed** (April 15, 2026)
- Fixed: RNN was starting generation from `<PAD>` token (index 0) instead of `<START>` (index 1) — every response was broken from step 1
- Fixed: Chat interface was encoding user input without special tokens, but training used them — now consistent
- Fixed: Tokenizer decode only cleaned up `.,!?` spacing; now also cleans `()[]{}` and contractions like `i ' m` → `i'm`

**The Stack / Code Parquet Support** (April 15, 2026)
- Added `parse_code_parquet()` helper that auto-detects code-file Parquet datasets (any file with a `content` column)
- Strategy 1: Extracts Python function + docstring pairs as user/bot training data (`"Write a Python function that: <docstring>"` → `<code>`)
- Strategy 2 (fallback): Splits code files in thirds as code-completion pairs when no docstrings found
- Filters to Python-only rows when a `lang` column exists
- Both the Data Upload section and the Training sidebar now handle The Stack format
- Existing conversation Parquet files are unaffected — detection is based on column presence

### October 2025 - Transformer Architecture Added! 🚀

**Transformer Model Implementation** (October 16, 2025)
- Added full Transformer architecture with multi-head attention mechanism
- Encoder-decoder structure with self-attention and cross-attention
- Positional encoding for word order understanding
- Layer normalization for stable deep network training
- Configurable: attention heads (1-32), layers (1-24), feed-forward dim (128-65536)
- Same GPU/CPU compatibility as RNN (CuPy/NumPy)
- Users can now choose: RNN (fast, basic) or Transformer (smart, like ChatGPT!)
- Transformer benefits: better context, attention mechanism, scalable to 1B+ params

### October 2025 - Major Performance & Quality Updates

**Batch Training Implementation** (October 16, 2025)
- Added batch processing for GPU optimization (batch sizes 1-128)
- Implemented gradient accumulation across batches for stable training
- 30-50x speed improvement on GPU vs single-sample training
- Batch size recommendations: GTX 1650 (32-64), RTX 3060 (64-128), CPU (8-16)
- Console logging shows batches per epoch and samples per batch

**Temperature Sampling for Generation** (October 16, 2025)
- Replaced greedy sampling (argmax) with temperature-based probability sampling
- Added temperature slider in chat interface (0.1-2.0, default 0.8)
- Fixed repetitive output issue (comma/character spam) with probabilistic token selection
- Lower temperature = focused/repetitive, Higher temperature = creative/random
- CuPy/NumPy compatibility for random sampling across CPU and GPU

**Training Progress & Logging** (October 16, 2025)
- Removed spinner blocking to enable real-time progress bar updates
- Added detailed console logging: batch progress, loss per batch, epoch summaries
- Training start info: total samples, batches per epoch, GPU/CPU mode indicator
- Training end summary: total batches processed, loss improvement percentage
- Visual feedback for training progression in both UI and console

**GPU Compatibility Fixes** (October 16, 2025)
- Fixed CuPy-NumPy implicit conversion errors in loss calculation
- All loss values converted to Python floats for compatibility
- Fixed CuPy random.choice() size parameter requirement
- Automatic GPU array conversion to CPU for operations requiring NumPy
- Ensures seamless operation on both GPU and CPU environments

## System Architecture

### Frontend Architecture

**Technology**: Streamlit web framework with interactive widgets and real-time updates

**Design Pattern**: Session state management for persisting model, tokenizer, and training data across user interactions

**Key Features**:
- Wide layout configuration with expandable sidebar for navigation
- Custom CSS styling for chat interface and visual presentation
- Interactive plotly-based visualizations for training loss curves
- Multi-step workflow: data upload → model setup → training → chat interface → export

**Supported Data Formats**:
- **JSON**: Standard conversation format `[{"user": "...", "bot": "..."}, ...]`
- **JSONL**: JSON Lines format (one object per line)
- **CSV/TSV**: Two-column format with flexible headers (user/bot, question/answer, input/output, etc.)
- **Code debugging JSON**: Objects with `original_src` and `changed_src` fields
- **Numbered JSON text**: Format like `0:{...} 1:{...}` style entries
- **Line-by-line text**: Multiple separators supported (`|`, `→`, `-`, tab) with flexible labels

**Rationale**: Streamlit provides rapid prototyping for chatbot applications with minimal frontend code while maintaining interactivity. Session state prevents data loss during user navigation and interactions.

### Backend Architecture

**Core Components**:

1. **VnexAI Chatbot Model** (`chatbot_model.py`)
   - Pure NumPy implementation of sequence-to-sequence RNN architecture
   - Encoder-decoder RNN for conversation modeling
   - Custom gradient descent optimization with backpropagation through time
   - **Batch training** with gradient accumulation (1-512 samples per batch)
   - **Temperature sampling** for diverse text generation (0.1-2.0 temperature range)
   - Configurable embedding and hidden dimensions
   - Supports text generation with autoregressive decoding
   - Binary (.bin) model export using pickle serialization

2. **Transformer Chatbot Model** (`transformer_model.py`)
   - Pure NumPy/CuPy implementation of Transformer architecture
   - Multi-head attention mechanism (1-32 heads) for context understanding
   - Encoder-decoder structure with self-attention and cross-attention
   - Positional encoding for word order without recurrence
   - Layer normalization for stable deep network training
   - Configurable: layers (1-24), heads, feed-forward dim (128-65536)
   - **Same batch training** with gradient accumulation (1-512 samples per batch)
   - **Temperature sampling** for creative generation
   - Scales to 1B+ parameters for RTX 4090/5090
   - Binary (.bin) model export using pickle serialization

3. **Chatbot Tokenizer** (`chatbot_tokenizer.py`)
   - Vocabulary building from training text
   - Text tokenization with special tokens (<PAD>, <START>, <END>, <UNK>)
   - Encode/decode functionality for text ↔ token indices conversion
   - Configurable vocabulary size with frequency-based selection
   - Binary (.bin) tokenizer export for deployment

3. **Data Processing** (`app.py`)
   - Multi-format data loading (JSON, text files)
   - Code debugging format converter (original_src → changed_src)
   - Numbered JSON entry parser
   - Automatic format detection and conversion
   - Training data preview and validation

**Design Decisions**:

- **GPU Acceleration Support**: Automatically detects and uses CuPy for GPU acceleration when available, falls back to NumPy for CPU. Provides 10-100x speedup for training on GPU-enabled systems without code changes.

- **NumPy/CuPy implementation**: Provides full control and transparency over the model architecture without framework dependencies. Educational value for understanding chatbot mechanics from scratch. GPU support via CuPy maintains identical API.

- **Sequence-to-sequence architecture**: Simple RNN-based encoder-decoder enables conversation modeling and code transformation tasks.

- **Flexible data formats**: Supports multiple file formats (JSON, JSONL, CSV, TSV, text) and conversational styles to accommodate various training data sources including code debugging datasets.

- **Binary export**: .bin file format (pickle) enables easy model deployment and sharing for use with other platforms. Arrays are converted to CPU (NumPy) before saving for compatibility.

### Data Processing Pipeline

**Flow**:
1. CSV data upload through Streamlit file uploader
2. Automatic feature type detection and classification/regression determination
3. Configurable preprocessing with sklearn transformers
4. Train/validation/test splitting with user-defined ratios
5. Feature scaling and encoding persistence for inference

**Design Choice**: Sklearn preprocessing maintains compatibility with industry standards and provides robust, tested implementations. Preprocessor state is stored in session to enable consistent transformations during prediction.

### Visualization & Metrics

**Technology**: Plotly for interactive, publication-quality visualizations

**Metrics System**:
- Classification: Accuracy, confusion matrix, classification report
- Regression: MSE, MAE, R² score
- Training monitoring: Loss and accuracy curves with validation metrics

**Rationale**: Plotly enables interactive exploration of results, zoom capabilities, and professional aesthetics. Subplot layouts allow simultaneous comparison of training and validation performance.

## External Dependencies

### Python Libraries

**Core ML/Data Processing**:
- `numpy`: Numerical computing for neural network implementation
- `pandas`: Data manipulation and CSV handling
- `scikit-learn`: Preprocessing utilities (StandardScaler, LabelEncoder, OneHotEncoder, train_test_split), evaluation metrics

**Visualization**:
- `plotly`: Interactive plotting library for training history and performance visualization
- `matplotlib`: Secondary plotting support
- `seaborn`: Statistical visualization enhancement

**Web Framework**:
- `streamlit`: Web application framework for the user interface

### Data Storage

**Format**: In-memory storage using Streamlit session state

**Storage Objects**:
- Trained model (VnexAI instance with weights and biases)
- Preprocessor state (scalers, encoders, feature metadata)
- Dataset splits (X_train, X_val, X_test, y_train, y_val, y_test)

**Rationale**: Session-based storage eliminates need for external database while maintaining user context during application lifecycle. For production deployment, this could be extended to persistent storage (database or file system) for model versioning and retrieval.

### Model Persistence

**Format**: JSON serialization for model weights, biases, and configuration

**Export Capability**: ZIP file generation containing model artifacts and preprocessor state

**Design Decision**: JSON provides human-readable model inspection and cross-platform compatibility. Future enhancement could include pickle serialization or ONNX export for production deployment scenarios.