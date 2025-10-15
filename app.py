import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import json
import io
import zipfile
from vnexai import VnexAI
from data_preprocessor import DataPreprocessor
from utils import (
    plot_training_history, display_model_performance, 
    create_architecture_diagram, validate_dataset, 
    get_recommended_architecture
)

# Set page configuration
st.set_page_config(
    page_title="VnexAI - Custom Neural Network Builder",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #555;
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'preprocessor' not in st.session_state:
    st.session_state.preprocessor = None
if 'X_train' not in st.session_state:
    st.session_state.X_train = None
if 'X_val' not in st.session_state:
    st.session_state.X_val = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_train' not in st.session_state:
    st.session_state.y_train = None
if 'y_val' not in st.session_state:
    st.session_state.y_val = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'training_history' not in st.session_state:
    st.session_state.training_history = None
if 'is_trained' not in st.session_state:
    st.session_state.is_trained = False

def main():
    # Main title
    st.markdown('<h1 class="main-header">🧠 VnexAI - Custom Neural Network Builder</h1>', unsafe_allow_html=True)
    st.markdown("Build and train custom neural networks from scratch using NumPy!")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a section:",
        ["Data Upload & Preprocessing", "Model Architecture", "Training", "Evaluation", "Model Management"]
    )
    
    if page == "Data Upload & Preprocessing":
        data_upload_section()
    elif page == "Model Architecture":
        model_architecture_section()
    elif page == "Training":
        training_section()
    elif page == "Evaluation":
        evaluation_section()
    elif page == "Model Management":
        model_management_section()

def data_upload_section():
    st.header("📊 Data Upload & Preprocessing")
    
    # File upload
    uploaded_file = st.file_uploader("Upload your CSV dataset", type=['csv'])
    
    if uploaded_file is not None:
        try:
            # Load data
            df = pd.read_csv(uploaded_file)
            st.success(f"Dataset loaded successfully! Shape: {df.shape}")
            
            # Validate dataset
            is_valid, issues = validate_dataset(df)
            if not is_valid:
                st.warning("Dataset validation issues:")
                for issue in issues:
                    st.write(f"⚠️ {issue}")
                st.write("Please fix these issues or proceed with caution.")
            else:
                st.success("✅ Dataset validation passed!")
            
            # Display basic info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rows", df.shape[0])
            with col2:
                st.metric("Columns", df.shape[1])
            with col3:
                missing_values = df.isnull().sum().sum()
                st.metric("Missing Values", missing_values)
            
            # Show data preview
            st.subheader("Data Preview")
            st.dataframe(df.head())
            
            # Data info
            st.subheader("Dataset Information")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Column Data Types:**")
                dtype_df = pd.DataFrame({
                    'Column': df.columns,
                    'Data Type': df.dtypes,
                    'Null Count': df.isnull().sum(),
                    'Unique Values': [df[col].nunique() for col in df.columns]
                })
                st.dataframe(dtype_df)
            
            with col2:
                st.write("**Missing Values Heatmap:**")
                if df.isnull().sum().sum() > 0:
                    missing_data = df.isnull().sum()
                    missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
                    fig = go.Figure([go.Bar(x=missing_data.index, y=missing_data.values)])
                    fig.update_layout(title="Missing Values by Column", 
                                    xaxis_title="Columns", yaxis_title="Missing Count")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.write("No missing values found!")
            
            # Target selection and preprocessing
            st.subheader("Preprocessing Configuration")
            
            col1, col2 = st.columns(2)
            with col1:
                target_column = st.selectbox("Select target column:", df.columns.tolist())
            
            with col2:
                test_size = st.slider("Test set size", 0.1, 0.4, 0.2, 0.05)
                val_size = st.slider("Validation set size (from training)", 0.1, 0.3, 0.1, 0.05)
            
            if st.button("Preprocess Data"):
                with st.spinner("Preprocessing data..."):
                    try:
                        # Initialize preprocessor
                        preprocessor = DataPreprocessor()
                        
                        # Preprocess data
                        X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.preprocess_data(
                            df, target_column, test_size=test_size, val_size=val_size
                        )
                        
                        # Store in session state
                        st.session_state.preprocessor = preprocessor
                        st.session_state.X_train = X_train
                        st.session_state.X_val = X_val
                        st.session_state.X_test = X_test
                        st.session_state.y_train = y_train
                        st.session_state.y_val = y_val
                        st.session_state.y_test = y_test
                        
                        st.success("Data preprocessing completed!")
                        
                        # Display preprocessing info
                        info = preprocessor.get_preprocessing_info()
                        st.write("**Preprocessing Summary:**")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Problem Type", info['problem_type'].title())
                        with col2:
                            st.metric("Features", info['num_features'])
                        with col3:
                            st.metric("Classes/Output Size", info['num_classes'])
                        
                        # Display data splits
                        st.write("**Data Splits:**")
                        split_data = pd.DataFrame({
                            'Split': ['Training', 'Validation', 'Test'],
                            'Samples': [X_train.shape[0], X_val.shape[0], X_test.shape[0]],
                            'Percentage': [
                                f"{X_train.shape[0]/df.shape[0]*100:.1f}%",
                                f"{X_val.shape[0]/df.shape[0]*100:.1f}%",
                                f"{X_test.shape[0]/df.shape[0]*100:.1f}%"
                            ]
                        })
                        st.dataframe(split_data)
                        
                    except Exception as e:
                        st.error(f"Error during preprocessing: {str(e)}")
        
        except Exception as e:
            st.error(f"Error loading dataset: {str(e)}")
    
    else:
        st.info("👆 Please upload a CSV file to get started")

def model_architecture_section():
    st.header("🏗️ Model Architecture")
    
    if st.session_state.preprocessor is None:
        st.warning("Please upload and preprocess data first!")
        return
    
    # Get preprocessing info
    info = st.session_state.preprocessor.get_preprocessing_info()
    num_features = info['num_features']
    num_classes = info['num_classes']
    problem_type = info['problem_type']
    
    st.write(f"**Input Features:** {num_features}")
    st.write(f"**Output Size:** {num_classes}")
    st.write(f"**Problem Type:** {problem_type.title()}")
    
    # Architecture configuration
    st.subheader("Network Architecture")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Architecture Options:**")
        arch_option = st.radio(
            "Choose architecture:",
            ["Recommended", "Custom"]
        )
        
        if arch_option == "Recommended":
            layers = get_recommended_architecture(num_features, num_classes, problem_type)
            st.write("**Recommended Architecture:**")
            for i, layer_size in enumerate(layers):
                if i == 0:
                    st.write(f"Input Layer: {layer_size}")
                elif i == len(layers) - 1:
                    st.write(f"Output Layer: {layer_size}")
                else:
                    st.write(f"Hidden Layer {i}: {layer_size}")
        
        else:  # Custom architecture
            st.write("**Custom Architecture:**")
            layers = [num_features]  # Input layer
            
            num_hidden = st.number_input("Number of hidden layers:", 1, 10, 2)
            
            for i in range(num_hidden):
                hidden_size = st.number_input(f"Hidden layer {i+1} size:", 1, 1000, 32)
                layers.append(hidden_size)
            
            layers.append(num_classes)  # Output layer
    
    with col2:
        st.write("**Hyperparameters:**")
        learning_rate = st.number_input("Learning rate:", 0.001, 1.0, 0.01, 0.001, format="%.3f")
        activation = st.selectbox("Hidden layers activation:", ['relu', 'sigmoid', 'tanh'])
        
        if problem_type == 'classification':
            if num_classes > 2:
                output_activation = 'softmax'
                st.write("Output activation: Softmax (multi-class)")
            else:
                output_activation = st.selectbox("Output activation:", ['sigmoid', 'softmax'])
        else:
            output_activation = st.selectbox("Output activation:", ['linear', 'relu'])
    
    # Display architecture diagram
    st.subheader("Architecture Visualization")
    diagram = create_architecture_diagram(layers)
    st.text(diagram)
    
    # Create model
    if st.button("Create Model"):
        try:
            model = VnexAI(
                layers=layers,
                learning_rate=learning_rate,
                activation=activation,
                output_activation=output_activation
            )
            
            st.session_state.model = model
            st.session_state.is_trained = False
            
            st.success("Model created successfully!")
            st.write("**Model Summary:**")
            st.text(model.get_model_summary())
            
        except Exception as e:
            st.error(f"Error creating model: {str(e)}")

def training_section():
    st.header("🎯 Model Training")
    
    if st.session_state.model is None:
        st.warning("Please create a model first!")
        return
    
    if st.session_state.X_train is None:
        st.warning("Please preprocess data first!")
        return
    
    # Training parameters
    st.subheader("Training Configuration")
    
    col1, col2 = st.columns(2)
    with col1:
        epochs = st.number_input("Number of epochs:", 1, 1000, 100)
        batch_size = st.number_input("Batch size:", 1, 512, 32)
    
    with col2:
        use_validation = st.checkbox("Use validation data", value=True)
        verbose = st.checkbox("Verbose training", value=True)
    
    # Training progress placeholders
    progress_bar = st.progress(0)
    status_text = st.empty()
    metrics_placeholder = st.empty()
    chart_placeholder = st.empty()
    
    if st.button("Start Training"):
        try:
            with st.spinner("Training in progress..."):
                # Prepare data
                X_val = st.session_state.X_val if use_validation else None
                y_val = st.session_state.y_val if use_validation else None
                
                # Custom training loop for real-time updates
                model = st.session_state.model
                X_train = st.session_state.X_train
                y_train = st.session_state.y_train
                
                # Training parameters
                n_samples = X_train.shape[0]
                n_batches = max(1, n_samples // batch_size)
                
                # Initialize history
                history = {
                    'loss': [],
                    'accuracy': [],
                    'val_loss': [],
                    'val_accuracy': []
                }
                
                # Training loop with real-time updates
                for epoch in range(epochs):
                    # Update progress
                    progress = (epoch + 1) / epochs
                    progress_bar.progress(progress)
                    status_text.text(f"Epoch {epoch + 1}/{epochs}")
                    
                    # Shuffle data
                    indices = np.random.permutation(n_samples)
                    X_shuffled = X_train[indices]
                    y_shuffled = y_train[indices]
                    
                    epoch_losses = []
                    epoch_accuracies = []
                    
                    # Mini-batch training
                    for i in range(n_batches):
                        start_idx = i * batch_size
                        end_idx = min((i + 1) * batch_size, n_samples)
                        
                        X_batch = X_shuffled[start_idx:end_idx]
                        y_batch = y_shuffled[start_idx:end_idx]
                        
                        batch_loss, batch_accuracy = model.train_batch(X_batch, y_batch)
                        epoch_losses.append(batch_loss)
                        epoch_accuracies.append(batch_accuracy)
                    
                    # Compute epoch metrics
                    avg_loss = np.mean(epoch_losses)
                    avg_accuracy = np.mean(epoch_accuracies)
                    
                    history['loss'].append(avg_loss)
                    history['accuracy'].append(avg_accuracy)
                    
                    # Validation metrics
                    if X_val is not None:
                        val_pred = model.predict(X_val)
                        val_loss = model.compute_loss(y_val, val_pred)
                        val_accuracy = model.compute_accuracy(y_val, val_pred)
                        
                        history['val_loss'].append(val_loss)
                        history['val_accuracy'].append(val_accuracy)
                    
                    # Update metrics display every 10 epochs or at the end
                    if epoch % 10 == 0 or epoch == epochs - 1:
                        # Display current metrics
                        if X_val is not None:
                            metrics_text = f"Loss: {avg_loss:.4f} | Acc: {avg_accuracy:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.4f}"
                        else:
                            metrics_text = f"Loss: {avg_loss:.4f} | Accuracy: {avg_accuracy:.4f}"
                        
                        metrics_placeholder.text(metrics_text)
                        
                        # Update training plot
                        if len(history['loss']) > 1:
                            fig = plot_training_history(history)
                            chart_placeholder.plotly_chart(fig, use_container_width=True)
                
                # Store training history
                st.session_state.training_history = history
                st.session_state.is_trained = True
                
                # Update model's training history
                model.training_history = history
                
                st.success("Training completed successfully!")
                
        except Exception as e:
            st.error(f"Error during training: {str(e)}")
            progress_bar.empty()
            status_text.empty()
    
    # Display training history if available
    if st.session_state.training_history is not None:
        st.subheader("Training History")
        fig = plot_training_history(st.session_state.training_history)
        st.plotly_chart(fig, use_container_width=True)

def evaluation_section():
    st.header("📈 Model Evaluation")
    
    if not st.session_state.is_trained:
        st.warning("Please train a model first!")
        return
    
    if st.session_state.X_test is None:
        st.warning("No test data available!")
        return
    
    st.subheader("Test Set Performance")
    
    try:
        model = st.session_state.model
        preprocessor = st.session_state.preprocessor
        X_test = st.session_state.X_test
        y_test = st.session_state.y_test
        
        # Get class names for classification
        class_names = None
        if preprocessor.is_classification and 'target' in preprocessor.encoders:
            class_names = preprocessor.encoders['target'].classes_.tolist()
        
        # Display performance metrics
        display_model_performance(
            model, X_test, y_test, 
            preprocessor.is_classification, class_names
        )
        
        # Additional analysis
        st.subheader("Prediction Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Sample predictions
            st.write("**Sample Predictions:**")
            n_samples = min(10, len(X_test))
            sample_indices = np.random.choice(len(X_test), n_samples, replace=False)
            
            y_pred = model.predict(X_test[sample_indices])
            
            if preprocessor.is_classification:
                if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
                    y_pred_classes = np.argmax(y_pred, axis=1)
                    y_pred_probs = np.max(y_pred, axis=1)
                else:
                    y_pred_classes = y_pred.astype(int)
                    y_pred_probs = None
                
                y_true_display = preprocessor.inverse_transform_target(y_test[sample_indices])
                y_pred_display = preprocessor.inverse_transform_target(y_pred_classes)
                
                sample_df = pd.DataFrame({
                    'True Label': y_true_display,
                    'Predicted Label': y_pred_display,
                })
                
                if y_pred_probs is not None:
                    sample_df['Confidence'] = y_pred_probs.round(4)
                
                st.dataframe(sample_df)
            
            else:
                sample_df = pd.DataFrame({
                    'True Value': y_test[sample_indices].flatten(),
                    'Predicted Value': y_pred.flatten().round(4),
                    'Error': (y_test[sample_indices] - y_pred).flatten().round(4)
                })
                st.dataframe(sample_df)
        
        with col2:
            # Feature importance (simple version based on weights)
            st.write("**Feature Importance (First Layer Weights):**")
            
            if len(model.weights) > 0:
                # Use absolute values of first layer weights as proxy for importance
                feature_importance = np.abs(model.weights[0]).mean(axis=1)
                feature_names = preprocessor.feature_names[:len(feature_importance)]
                
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': feature_importance
                }).sort_values('Importance', ascending=False)
                
                # Show top 10 features
                top_features = importance_df.head(10)
                
                fig = go.Figure([go.Bar(
                    x=top_features['Importance'],
                    y=top_features['Feature'],
                    orientation='h'
                )])
                
                fig.update_layout(
                    title="Top 10 Feature Importance",
                    xaxis_title="Average Absolute Weight",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    except Exception as e:
        st.error(f"Error during evaluation: {str(e)}")

def model_management_section():
    st.header("💾 Model Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Save Model")
        
        if st.session_state.model is not None:
            model_name = st.text_input("Model name:", value="vnexai_model")
            
            if st.button("Download Model"):
                try:
                    # Create a zip file with model and preprocessing info
                    zip_buffer = io.BytesIO()
                    
                    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                        # Save model
                        model_json = {
                            'layers': st.session_state.model.layers,
                            'learning_rate': st.session_state.model.learning_rate,
                            'activation': st.session_state.model.activation,
                            'output_activation': st.session_state.model.output_activation,
                            'weights': [w.tolist() for w in st.session_state.model.weights],
                            'biases': [b.tolist() for b in st.session_state.model.biases],
                            'training_history': st.session_state.model.training_history
                        }
                        
                        zip_file.writestr(f"{model_name}.json", json.dumps(model_json, indent=2))
                        
                        # Save preprocessing info
                        if st.session_state.preprocessor is not None:
                            preprocess_info = {
                                'feature_names': st.session_state.preprocessor.feature_names,
                                'target_name': st.session_state.preprocessor.target_name,
                                'is_classification': st.session_state.preprocessor.is_classification,
                                'num_classes': st.session_state.preprocessor.num_classes,
                                'scalers': {k: {'mean_': v.mean_.tolist(), 'scale_': v.scale_.tolist()} 
                                          for k, v in st.session_state.preprocessor.scalers.items()},
                                'encoders': {k: {'classes_': v.classes_.tolist() if hasattr(v, 'classes_') else None}
                                           for k, v in st.session_state.preprocessor.encoders.items() 
                                           if hasattr(v, 'classes_')}
                            }
                            zip_file.writestr("preprocessing_info.json", json.dumps(preprocess_info, indent=2))
                        
                        # Save model summary
                        summary = st.session_state.model.get_model_summary()
                        zip_file.writestr("model_summary.txt", summary)
                    
                    zip_buffer.seek(0)
                    
                    st.download_button(
                        label="📥 Download VnexAI Model",
                        data=zip_buffer.getvalue(),
                        file_name=f"{model_name}.zip",
                        mime="application/zip"
                    )
                    
                    st.success("Model package created successfully!")
                
                except Exception as e:
                    st.error(f"Error saving model: {str(e)}")
        else:
            st.info("No model available to save. Please create and train a model first.")
    
    with col2:
        st.subheader("Load Model")
        
        uploaded_model = st.file_uploader("Upload VnexAI model (.zip)", type=['zip'])
        
        if uploaded_model is not None:
            if st.button("Load Model"):
                try:
                    # Extract and load model
                    zip_buffer = io.BytesIO(uploaded_model.read())
                    
                    with zipfile.ZipFile(zip_buffer, 'r') as zip_file:
                        # Find model JSON file
                        json_files = [f for f in zip_file.namelist() if f.endswith('.json') and 'preprocessing' not in f]
                        
                        if json_files:
                            model_file = json_files[0]
                            model_data = json.loads(zip_file.read(model_file).decode('utf-8'))
                            
                            # Create and load model
                            model = VnexAI(
                                layers=model_data['layers'],
                                learning_rate=model_data['learning_rate'],
                                activation=model_data['activation'],
                                output_activation=model_data['output_activation']
                            )
                            
                            model.weights = [np.array(w) for w in model_data['weights']]
                            model.biases = [np.array(b) for b in model_data['biases']]
                            model.training_history = model_data['training_history']
                            
                            st.session_state.model = model
                            st.session_state.is_trained = True
                            st.session_state.training_history = model_data['training_history']
                            
                            st.success("Model loaded successfully!")
                            st.text(model.get_model_summary())
                        else:
                            st.error("No model file found in the uploaded zip.")
                
                except Exception as e:
                    st.error(f"Error loading model: {str(e)}")
    
    # Model Information
    if st.session_state.model is not None:
        st.subheader("Current Model Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.text(st.session_state.model.get_model_summary())
        
        with col2:
            if st.session_state.training_history is not None:
                final_metrics = {
                    'Final Training Loss': st.session_state.training_history['loss'][-1] if st.session_state.training_history['loss'] else 'N/A',
                    'Final Training Accuracy': st.session_state.training_history['accuracy'][-1] if st.session_state.training_history['accuracy'] else 'N/A',
                    'Total Epochs Trained': len(st.session_state.training_history['loss']),
                }
                
                if st.session_state.training_history['val_loss']:
                    final_metrics['Final Validation Loss'] = st.session_state.training_history['val_loss'][-1]
                    final_metrics['Final Validation Accuracy'] = st.session_state.training_history['val_accuracy'][-1]
                
                st.write("**Training Statistics:**")
                for key, value in final_metrics.items():
                    if isinstance(value, float):
                        st.write(f"- {key}: {value:.4f}")
                    else:
                        st.write(f"- {key}: {value}")

if __name__ == "__main__":
    main()
