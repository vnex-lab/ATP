# VnexAI - Custom Neural Network Builder

## Overview

VnexAI is a custom neural network framework built from scratch using NumPy, with a Streamlit-based web interface for building, training, and evaluating neural networks. The application enables users to upload datasets, automatically preprocess data, configure custom neural network architectures, train models, and visualize results through interactive plots and performance metrics. The framework supports both classification and regression tasks with automatic problem type detection and feature engineering.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture

**Technology**: Streamlit web framework with interactive widgets and real-time updates

**Design Pattern**: Session state management for persisting model, preprocessor, and dataset objects across user interactions

**Key Features**:
- Wide layout configuration with expandable sidebar for configuration options
- Custom CSS styling for enhanced visual presentation
- Interactive plotly-based visualizations for training metrics and model performance
- Multi-step workflow: data upload → preprocessing → architecture design → training → evaluation

**Rationale**: Streamlit provides rapid prototyping for ML applications with minimal frontend code while maintaining interactivity. Session state prevents data loss during user navigation and interactions.

### Backend Architecture

**Core Components**:

1. **VnexAI Neural Network Engine** (`vnexai.py`)
   - Pure NumPy implementation of feedforward neural networks
   - Custom gradient descent optimization
   - Configurable layer architectures and activation functions
   - Xavier weight initialization for stable training
   - Support for multiple activation functions (ReLU, sigmoid, tanh) and output types (softmax, sigmoid, linear)

2. **Data Preprocessor** (`data_preprocessor.py`)
   - Automatic feature type detection (numerical vs categorical)
   - Automatic problem type detection (classification vs regression)
   - Sklearn-based preprocessing pipelines
   - StandardScaler for numerical features
   - Label encoding and one-hot encoding for categorical features
   - Train/validation/test split functionality

3. **Utility Module** (`utils.py`)
   - Training history visualization with Plotly subplots
   - Model performance metrics and displays
   - Confusion matrices for classification
   - Regression metrics (MSE, R², MAE)
   - Architecture diagram generation
   - Dataset validation
   - Recommended architecture suggestions

**Design Decisions**:

- **NumPy-only neural network**: Provides educational value and full control over implementation without framework dependencies. Trade-off is slower training compared to optimized frameworks like TensorFlow/PyTorch.

- **Modular architecture**: Separation of concerns between data preprocessing, model training, and visualization enables independent testing and future extensibility.

- **Automatic detection**: Reduces user configuration burden by intelligently detecting problem types and feature characteristics, improving user experience for non-technical users.

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