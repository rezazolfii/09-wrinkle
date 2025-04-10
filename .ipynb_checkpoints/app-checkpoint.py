import streamlit as st
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import os

# Set page configuration and custom CSS
st.set_page_config(page_title="Skin Analysis App", layout="wide")

# Custom CSS for styling
st.markdown("""
<style>
    .sidebar-container {
        background-color: white !important;
        color: #262730 !important;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        height: 100%;
    }
    
    .results-container {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        margin-top: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        height: 100%;
    }
    
    .stButton button {
        width: 100%;
        background-color: #4e8cff;
        color: white;
        font-weight: bold;
    }
    
    .title {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 20px;
    }
    
    .subtitle {
        font-size: 1.5rem;
        font-weight: bold;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

# Custom loss functions and metrics
def dice_loss(y_true, y_pred, smooth=1e-5):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def iou_loss(y_true, y_pred, smooth=1e-5):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
    return 1 - (intersection + smooth) / (union + smooth)

def weighted_bce_improved(y_true, y_pred, weight_0=0.05, weight_1=0.95):
    """More aggressive weighting to focus on the wrinkle class"""
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    loss = -weight_1 * y_true * K.log(y_pred + 1e-7) - weight_0 * (1 - y_true) * K.log(1 - y_pred + 1e-7)
    return K.mean(loss)

def iou_focused_loss(y_true, y_pred):
    return (2 * iou_loss(y_true, y_pred)) + dice_loss(y_true, y_pred) + weighted_bce_improved(y_true, y_pred)

def custom_iou_metric(threshold=0.5):
    def iou(y_true, y_pred):
        # Apply threshold to predictions
        y_pred = tf.cast(y_pred > threshold, tf.float32)
        y_true = tf.cast(y_true > 0.5, tf.float32)
        
        intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
        union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3]) - intersection
        iou = (intersection + 1e-7) / (union + 1e-7)
        return K.mean(iou)
    iou.__name__ = 'iou'  # Set the name explicitly
    return iou

# Function to load the model - using st.cache for older Streamlit versions
@st.cache_resource
def load_wrinkle_model():
    custom_objects = {
        'iou_focused_loss': iou_focused_loss,
        'iou': custom_iou_metric(threshold=0.4),
        'dice_loss': dice_loss,
        'iou_loss': iou_loss,
        'weighted_bce_improved': weighted_bce_improved
    }
    
    model_path = 'best_model_deeplabv3plus_iou.keras'
    if os.path.exists(model_path):
        return load_model(model_path, custom_objects=custom_objects)
    else:
        st.error(f"Model file not found at {model_path}. Please make sure the model file exists.")
        return None

# Function to preprocess the image
def preprocess_image(img):
    img = tf.image.resize(img, (512, 512))
    img = tf.cast(img, tf.float32) / 255.0
    return img

# Function to create visualization - modified to only show prediction and overlay
def create_visualization(img, prediction, threshold=0.7):
    # Create a figure with 2 subplots (removing the original image display)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # Prediction
    axes[0].imshow(prediction[0, :, :, 0], cmap='gray')
    axes[0].set_title('Predicted Wrinkles')
    axes[0].axis('off')
    
    # Overlay
    pred_binary = prediction[0, :, :, 0] > threshold
    overlay = img.numpy().copy()
    overlay[:, :, 1][pred_binary] = 1.0  # Set green channel to max for wrinkles
    alpha = 0.5
    blended = img.numpy() * (1 - alpha) + overlay * alpha
    axes[1].imshow(blended)
    axes[1].set_title('Wrinkles Overlay')
    axes[1].axis('off')
    
    plt.tight_layout()
    return fig

# Main app
def main():
    # Title
    st.markdown('<p class="title">Skin Analysis App</p>', unsafe_allow_html=True)
    
    # Create a layout with three columns
    col1, col2, col3 = st.columns([1, 1, 1])
    
    # Left column - Upload section with white background
    with col1:
        st.markdown('<div class="sidebar-container">', unsafe_allow_html=True)
        st.markdown('<p class="subtitle">Upload Image</p>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        analyze_button = st.button("Analyze")
        
        # Add some information about the app
        st.markdown("---")
        st.markdown("### About")
        st.markdown("This app uses a deep learning model to detect wrinkles in facial images.")
        st.markdown("Upload a clear, well-lit photo of a face for best results.")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Middle column - Results section
    with col2:
        # Results container
        st.markdown('<div class="results-container">', unsafe_allow_html=True)
        st.markdown('<p class="subtitle">Analysis Results</p>', unsafe_allow_html=True)
        
        if uploaded_file is not None:
            # Don't display the original uploaded image as requested
            
            # Convert PIL Image to TensorFlow tensor
            image = Image.open(uploaded_file)
            img_array = np.array(image)
            img_tensor = tf.convert_to_tensor(img_array)
            
            if analyze_button:
                with st.spinner("Analyzing image..."):
                    # Load the model
                    model = load_wrinkle_model()
                    
                    if model is not None:
                        # Preprocess the image
                        processed_img = preprocess_image(img_tensor)
                        img_batch = tf.expand_dims(processed_img, axis=0)
                        
                        # Make prediction
                        prediction = model.predict(img_batch)
                        
                        # Create visualization
                        fig = create_visualization(processed_img, prediction)
                        
                        # Display the results
                        st.pyplot(fig)
                        
                        # Add some metrics
                        st.markdown("### Analysis Metrics")
                        coverage = np.mean(prediction[0, :, :, 0] > 0.7) * 100
                        st.metric("Wrinkle Coverage", f"{coverage:.2f}%")
                        
                        # Add download button for the result
                        buf = io.BytesIO()
                        fig.savefig(buf, format="png")
                        buf.seek(0)
                        st.download_button(
                            label="Download Analysis",
                            data=buf,
                            file_name="wrinkle_analysis.png",
                            mime="image/png"
                        )
        else:
            st.info("Please upload an image to see analysis results")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Right column - For the image the user will add
    with col3:
        # Replace this line in the right column section:
        st.info("This column is reserved for your reference image. You can add it by modifying the code.")

# With this:
        st.image("Picture1.png", caption="Your Reference Image", use_container_width=True)
        st.markdown('<div class="results-container">', unsafe_allow_html=True)
        st.markdown('<p class="subtitle">Reference Image</p>', unsafe_allow_html=True)
        st.info("This column is reserved for your reference image. You can add it by modifying the code.")
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()