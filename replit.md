# VnexAI - Custom Chatbot Training Platform

## Overview

VnexAI is a custom chatbot training platform enabling users to build conversational AI models using either RNN or Transformer architectures. It allows users to upload conversation or code debugging data, configure neural network parameters, train models with GPU acceleration, test them in a chat interface, and export trained models for deployment. The platform aims to provide a comprehensive, from-scratch solution for custom AI model creation, focusing on transparency and control over the model architecture.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture

**Technology**: Streamlit for the web interface, providing interactive widgets and real-time updates.

**Design Pattern**: Session state management to maintain model, tokenizer, and training data across user interactions.

**Key Features**:
- Wide layout with an expandable sidebar.
- Custom CSS for styling.
- Interactive Plotly visualizations for training loss.
- Multi-step workflow: data upload, model setup, training, chat interface, and export.
- Supports various data formats: JSON, JSONL, CSV/TSV (two-column), code debugging JSON (`original_src`, `changed_src`), numbered JSON text, and line-by-line text with multiple separators. Automatically detects and converts "The Stack" Parquet datasets.

### Backend Architecture

**Core Components**:

1.  **VnexAI Chatbot Model (`chatbot_model.py`)**:
    -   Pure NumPy (with CuPy for GPU) implementation of a sequence-to-sequence RNN.
    -   Encoder-decoder architecture with custom gradient descent and backpropagation through time.
    -   Supports batch training with gradient accumulation and temperature sampling for generation.
    -   Configurable embedding and hidden dimensions.
    -   Exports as `.bin` files via pickle.

2.  **Transformer Chatbot Model (`transformer_model.py`)**:
    -   Pure NumPy/CuPy implementation of a Transformer architecture.
    -   Multi-head attention mechanism with encoder-decoder structure (self-attention and cross-attention).
    -   Includes positional encoding and layer normalization.
    -   Configurable layers, attention heads, and feed-forward dimensions.
    -   Supports batch training with gradient accumulation and temperature sampling.
    -   Exports as `.bin` files via pickle.

3.  **Chatbot Tokenizer (`chatbot_tokenizer.py`)**:
    -   Builds vocabulary from training text.
    -   Tokenizes text with special tokens (`<PAD>`, `<START>`, `<END>`, `<UNK>`).
    -   Configurable vocabulary size based on frequency.
    -   Exports as `.bin` files.

4.  **Data Processing (`app.py`)**:
    -   Handles multi-format data loading and automatic format detection.
    -   Includes specific parsers for code debugging formats and numbered JSON.

**Design Decisions**:
-   **GPU Acceleration**: Automatic detection and utilization of CuPy for GPU acceleration, falling back to NumPy for CPU, offering significant speedups.
-   **NumPy/CuPy Implementation**: Provides granular control and transparency over the neural network mechanics without external deep learning frameworks.
-   **Sequence-to-sequence**: Enables various tasks like conversation modeling and code transformation.
-   **Flexible Data Formats**: Accommodates diverse training data, including code debugging datasets.
-   **Binary Export**: Uses pickle for `.bin` file export, facilitating easy model deployment.

### Data Processing Pipeline

**Flow**:
1.  CSV data upload via Streamlit.
2.  Automatic feature type detection.
3.  Configurable preprocessing using scikit-learn transformers.
4.  Train/validation/test splitting.
5.  Feature scaling and encoding persistence for inference.

### Visualization & Metrics

**Technology**: Plotly for interactive visualizations.

**Metrics System**:
-   Classification: Accuracy, confusion matrix, classification report.
-   Regression: MSE, MAE, R² score.
-   Training monitoring: Loss and accuracy curves.

## External Dependencies

### Python Libraries

**Core ML/Data Processing**:
-   `numpy`: Numerical computing.
-   `cupy`: GPU acceleration.
-   `pandas`: Data manipulation.
-   `scikit-learn`: Preprocessing, metrics.

**Visualization**:
-   `plotly`: Interactive plotting.
-   `matplotlib`: Secondary plotting.
-   `seaborn`: Statistical visualization.

**Web Framework**:
-   `streamlit`: User interface framework.

### Data Storage

**Format**: In-memory using Streamlit session state.

**Storage Objects**: Trained model, preprocessor state, dataset splits.

### Model Persistence

**Format**: JSON serialization for model parameters and configuration.

**Export Capability**: ZIP file generation containing model artifacts.