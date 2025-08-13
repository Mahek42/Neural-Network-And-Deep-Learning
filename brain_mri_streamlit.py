import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input as effnet_preprocess
import joblib
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
IMG_H, IMG_W = 224, 224
WORK_DIR = "brain_mri_work"  # This should match your training output directory

class ModelManager:
    """Manages loading and prediction for all trained models"""
    
    def __init__(self, work_dir):
        self.work_dir = Path(work_dir)
        self.models = {}
        self.label_encoder_classes = None
        self.efficientnet_base = None
        self.load_label_mapping()
        
    def load_label_mapping(self):
        """Load the label mapping from training"""
        try:
            with open(self.work_dir / "models" / "label_index.json", 'r') as f:
                label_data = json.load(f)
                self.label_encoder_classes = label_data["classes"]
        except Exception as e:
            st.error(f"Could not load label mapping: {e}")
            self.label_encoder_classes = []
    
    @st.cache_resource
    def load_cnn_models(_self):
        """Load CNN models (custom and EfficientNet)"""
        models = {}
        
        # Load Custom CNN
        try:
            custom_path = _self.work_dir / "models" / "cnn_custom" / "saved_model.keras"
            if custom_path.exists():
                models["Custom CNN"] = tf.keras.models.load_model(str(custom_path))
                st.success("✅ Custom CNN loaded successfully")
            else:
                st.warning("⚠️ Custom CNN model not found")
        except Exception as e:
            st.error(f"❌ Error loading Custom CNN: {e}")
        
        # Load EfficientNet CNN
        try:
            effnet_path = _self.work_dir / "models" / "cnn_efficientnet" / "saved_model.keras"
            if effnet_path.exists():
                models["EfficientNet CNN"] = tf.keras.models.load_model(str(effnet_path))
                st.success("✅ EfficientNet CNN loaded successfully")
            else:
                st.warning("⚠️ EfficientNet CNN model not found")
        except Exception as e:
            st.error(f"❌ Error loading EfficientNet CNN: {e}")
            
        return models
    
    @st.cache_resource  
    def load_sklearn_models(_self):
        """Load scikit-learn models"""
        models = {}
        sklearn_dir = _self.work_dir / "models" / "sklearn"
        
        model_files = {
            "SVM (RBF)": "svm_rbf.joblib",
            "Random Forest": "random_forest.joblib", 
            "K-Nearest Neighbors": "knn.joblib",
            "Logistic Regression": "logreg.joblib"
        }
        
        for model_name, filename in model_files.items():
            try:
                model_path = sklearn_dir / filename
                if model_path.exists():
                    models[model_name] = joblib.load(model_path)
                    st.success(f"✅ {model_name} loaded successfully")
                else:
                    st.warning(f"⚠️ {model_name} model not found at {model_path}")
            except Exception as e:
                st.error(f"❌ Error loading {model_name}: {e}")
                
        return models
    
    @st.cache_resource
    def load_efficientnet_base(_self):
        """Load EfficientNet base model for feature extraction"""
        try:
            from tensorflow.keras.applications import EfficientNetB0
            base = EfficientNetB0(include_top=False, weights='imagenet', 
                                input_shape=(IMG_H, IMG_W, 3))
            st.success("✅ EfficientNet base model loaded for feature extraction")
            return base
        except Exception as e:
            st.error(f"❌ Error loading EfficientNet base: {e}")
            return None
    
    def preprocess_image_for_cnn(self, image):
        """Preprocess image for CNN models"""
        # Resize and normalize
        img_array = np.array(image.resize((IMG_W, IMG_H)))
        img_array = img_array.astype(np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        return img_array
    
    def preprocess_image_for_efficientnet(self, image):
        """Preprocess image specifically for EfficientNet"""
        img_array = np.array(image.resize((IMG_W, IMG_H)))
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array = effnet_preprocess(img_array)
        return img_array
    
    def extract_features(self, image):
        """Extract deep features using EfficientNet base"""
        if self.efficientnet_base is None:
            self.efficientnet_base = self.load_efficientnet_base()
        
        if self.efficientnet_base is None:
            return None
            
        # Preprocess for EfficientNet
        img_array = self.preprocess_image_for_efficientnet(image)
        
        # Extract features
        features = self.efficientnet_base(img_array, training=False)
        
        # Global average pooling
        gap = tf.keras.layers.GlobalAveragePooling2D()
        features = gap(features).numpy()
        
        return features
    
    def predict_cnn(self, model, image, model_name):
        """Make prediction using CNN model"""
        try:
            if "EfficientNet" in model_name:
                img_array = self.preprocess_image_for_efficientnet(image)
            else:
                img_array = self.preprocess_image_for_cnn(image)
            
            predictions = model(img_array, training=False).numpy()[0]
            
            # Get predicted class and confidence
            predicted_idx = np.argmax(predictions)
            confidence = predictions[predicted_idx]
            predicted_class = self.label_encoder_classes[predicted_idx] if self.label_encoder_classes else f"Class_{predicted_idx}"
            
            # Get top 3 predictions
            top_3_idx = np.argsort(predictions)[-3:][::-1]
            top_3_predictions = []
            for idx in top_3_idx:
                class_name = self.label_encoder_classes[idx] if self.label_encoder_classes else f"Class_{idx}"
                top_3_predictions.append((class_name, predictions[idx]))
            
            return predicted_class, confidence, top_3_predictions, predictions
            
        except Exception as e:
            st.error(f"Error during CNN prediction: {e}")
            return None, None, None, None
    
    def predict_sklearn(self, model, image):
        """Make prediction using scikit-learn model"""
        try:
            # Extract features first
            features = self.extract_features(image)
            if features is None:
                return None, None, None, None
            
            # Make prediction
            prediction = model.predict(features)[0]
            
            # Get prediction probabilities if available
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(features)[0]
                confidence = np.max(probabilities)
                
                # Get top 3 predictions
                top_3_idx = np.argsort(probabilities)[-3:][::-1]
                top_3_predictions = []
                for idx in top_3_idx:
                    class_name = self.label_encoder_classes[idx] if self.label_encoder_classes else f"Class_{idx}"
                    top_3_predictions.append((class_name, probabilities[idx]))
                    
                all_probabilities = probabilities
            else:
                confidence = 1.0  # For models without probability estimation
                top_3_predictions = [(self.label_encoder_classes[prediction] if self.label_encoder_classes else f"Class_{prediction}", 1.0)]
                all_probabilities = None
            
            predicted_class = self.label_encoder_classes[prediction] if self.label_encoder_classes else f"Class_{prediction}"
            
            return predicted_class, confidence, top_3_predictions, all_probabilities
            
        except Exception as e:
            st.error(f"Error during sklearn prediction: {e}")
            return None, None, None, None

def plot_prediction_confidence(predictions, class_names, title):
    """Plot prediction confidence as a bar chart"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if predictions is not None and len(predictions) > 0:
        # Sort by confidence for better visualization
        sorted_indices = np.argsort(predictions)[::-1]
        sorted_predictions = predictions[sorted_indices]
        sorted_classes = [class_names[i] for i in sorted_indices]
        
        # Only show top 10 for readability
        top_n = min(10, len(sorted_predictions))
        
        bars = ax.bar(range(top_n), sorted_predictions[:top_n])
        ax.set_xlabel('Classes')
        ax.set_ylabel('Confidence')
        ax.set_title(title)
        ax.set_xticks(range(top_n))
        ax.set_xticklabels(sorted_classes[:top_n], rotation=45, ha='right')
        
        # Color the highest confidence bar differently
        bars[0].set_color('red')
        for i in range(1, len(bars)):
            bars[i].set_color('skyblue')
        
        plt.tight_layout()
        return fig
    else:
        ax.text(0.5, 0.5, 'No prediction data available', 
                horizontalalignment='center', verticalalignment='center', 
                transform=ax.transAxes, fontsize=16)
        ax.set_title(title)
        return fig

def main():
    st.set_page_config(
        page_title="Brain MRI Classification",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Title and description
    st.title("🧠 Brain MRI Classification System")
    st.markdown("""
    Upload a brain MRI image and select a model to get predictions for brain tumor classification.
    This system uses multiple machine learning approaches including deep learning and classical ML models.
    """)
    
    # Initialize model manager
    @st.cache_resource
    def load_model_manager():
        return ModelManager(WORK_DIR)
    
    try:
        model_manager = load_model_manager()
        
        # Check if models directory exists
        if not Path(WORK_DIR).exists():
            st.error(f"❌ Models directory '{WORK_DIR}' not found! Please ensure you've run the training script and the output folder is in the same directory as this app.")
            st.stop()
            
    except Exception as e:
        st.error(f"❌ Error initializing model manager: {e}")
        st.stop()
    
    # Sidebar for model selection
    with st.sidebar:
        st.header("🔧 Model Selection")
        
        # Load all models
        cnn_models = model_manager.load_cnn_models()
        sklearn_models = model_manager.load_sklearn_models()
        
        # Combine all available models
        all_models = {**cnn_models, **sklearn_models}
        
        if not all_models:
            st.error("❌ No models found! Please check your model files.")
            st.stop()
        
        selected_model_name = st.selectbox(
            "Choose a model:",
            options=list(all_models.keys()),
            index=0
        )
        
        st.markdown("---")
        st.header("ℹ️ Model Information")
        
        # Display model info
        if "CNN" in selected_model_name:
            st.info("🤖 **Deep Learning Model**\n\nThis model uses Convolutional Neural Networks to analyze image patterns directly.")
        else:
            st.info("🔢 **Classical ML Model**\n\nThis model uses deep features extracted from images and applies traditional machine learning algorithms.")
        
        # Show available classes
        if model_manager.label_encoder_classes:
            st.markdown("**📋 Detectable Classes:**")
            for i, class_name in enumerate(model_manager.label_encoder_classes):
                st.write(f"• {class_name}")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Image")
        
        uploaded_file = st.file_uploader(
            "Choose a brain MRI image...",
            type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
            help="Upload a brain MRI image for classification"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption='Uploaded MRI Image', use_column_width=True)
            
            # Image info
            st.write(f"**Image size:** {image.size[0]} x {image.size[1]} pixels")
            st.write(f"**File size:** {uploaded_file.size / 1024:.1f} KB")
    
    with col2:
        st.header("🎯 Prediction Results")
        
        if uploaded_file is not None:
            # Make prediction button
            if st.button("🔍 Analyze Image", type="primary"):
                with st.spinner(f"Analyzing with {selected_model_name}..."):
                    selected_model = all_models[selected_model_name]
                    
                    # Make prediction based on model type
                    if "CNN" in selected_model_name:
                        predicted_class, confidence, top_3, all_predictions = model_manager.predict_cnn(
                            selected_model, image, selected_model_name
                        )
                    else:
                        predicted_class, confidence, top_3, all_predictions = model_manager.predict_sklearn(
                            selected_model, image
                        )
                    
                    # Display results
                    if predicted_class is not None:
                        st.success("✅ Analysis Complete!")
                        
                        # Main prediction
                        st.markdown("### 🏆 Primary Prediction")
                        col_pred1, col_pred2 = st.columns([2, 1])
                        with col_pred1:
                            st.markdown(f"**Class:** `{predicted_class}`")
                        with col_pred2:
                            st.markdown(f"**Confidence:** `{confidence:.1%}`")
                        
                        # Confidence indicator
                        if confidence > 0.8:
                            st.success("🟢 High confidence prediction")
                        elif confidence > 0.6:
                            st.warning("🟡 Medium confidence prediction")
                        else:
                            st.error("🔴 Low confidence prediction")
                        
                        # Top 3 predictions
                        st.markdown("### 📊 Top 3 Predictions")
                        for i, (class_name, conf) in enumerate(top_3[:3]):
                            emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉"
                            st.write(f"{emoji} **{class_name}**: {conf:.1%}")
                        
                        # Confidence visualization
                        if all_predictions is not None and model_manager.label_encoder_classes:
                            st.markdown("### 📈 Confidence Distribution")
                            fig = plot_prediction_confidence(
                                all_predictions, 
                                model_manager.label_encoder_classes,
                                f"Prediction Confidence - {selected_model_name}"
                            )
                            st.pyplot(fig)
                            plt.close(fig)
                    
                    else:
                        st.error("❌ Prediction failed. Please try again with a different image or model.")
        else:
            st.info("👆 Please upload an image to get started")
    
    # Footer with additional information
    st.markdown("---")
    with st.expander("ℹ️ About this Application"):
        st.markdown("""
        ### Brain MRI Classification System
        
        This application provides automated analysis of brain MRI images using multiple machine learning approaches:
        
        **🤖 Deep Learning Models:**
        - **Custom CNN**: A custom-built Convolutional Neural Network
        - **EfficientNet**: Pre-trained EfficientNetB0 with transfer learning
        
        **🔢 Classical ML Models:**
        - **SVM**: Support Vector Machine with RBF kernel
        - **Random Forest**: Ensemble of decision trees
        - **K-Nearest Neighbors**: Instance-based learning
        - **Logistic Regression**: Linear classification model
        
        **📝 Usage Notes:**
        - Upload brain MRI images in common formats (PNG, JPG, etc.)
        - Images are automatically resized to 224x224 pixels
        - Classical ML models use deep features extracted from EfficientNet
        - Confidence scores indicate prediction reliability
        
        **⚠️ Disclaimer:** This is for educational/research purposes only. Not for medical diagnosis.
        """)

if __name__ == "__main__":
    main()