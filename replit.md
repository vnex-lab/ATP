# VnexAI - Custom Chatbot Training Platform

## Overview

VnexAI is a custom chatbot training platform built from scratch using NumPy, with a Streamlit-based web interface. The application enables users to train conversational AI models and code debugging assistants on custom datasets. Users can upload conversation data or code debugging data, configure the neural network architecture, train the model from scratch, test it in a chat interface, and export the trained model as a .bin file for deployment.

## User Preferences

Preferred communication style: Simple, everyday language.

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
- Standard conversation JSON: `[{"user": "...", "bot": "..."}, ...]`
- Code debugging JSON: Objects with `original_src` and `changed_src` fields
- Numbered JSON text format: `0:{...} 1:{...}` style entries
- Line-by-line text: `user: ... | bot: ...`

**Rationale**: Streamlit provides rapid prototyping for chatbot applications with minimal frontend code while maintaining interactivity. Session state prevents data loss during user navigation and interactions.

### Backend Architecture

**Core Components**:

1. **VnexAI Chatbot Model** (`chatbot_model.py`)
   - Pure NumPy implementation of sequence-to-sequence architecture
   - Encoder-decoder RNN for conversation modeling
   - Custom gradient descent optimization with backpropagation through time
   - Configurable embedding and hidden dimensions
   - Supports text generation with autoregressive decoding
   - Binary (.bin) model export using pickle serialization

2. **Chatbot Tokenizer** (`chatbot_tokenizer.py`)
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

- **NumPy-only implementation**: Provides full control and transparency over the model architecture without framework dependencies. Educational value for understanding chatbot mechanics from scratch.

- **Sequence-to-sequence architecture**: Simple RNN-based encoder-decoder enables conversation modeling and code transformation tasks.

- **Flexible data formats**: Supports both conversational chatbots and code debugging assistants by auto-converting different data formats to user-bot pairs.

- **Binary export**: .bin file format (pickle) enables easy model deployment and sharing for use with other platforms.

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