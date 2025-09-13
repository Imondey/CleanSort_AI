import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
import os

# --- Configuration ---
# Update the model path to use absolute path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(os.path.dirname(CURRENT_DIR), 'models', 'waste_classifier.h5')
IMG_WIDTH, IMG_HEIGHT = 224, 224
CLASS_NAMES = ['metal', 'organic', 'paper', 'plastic']

# Custom CSS to make it more attractive
st.markdown("""
<style>
.main {
    padding: 2rem;
}
.title {
    color: #2c3e50;
    text-align: center;
}
.prediction {
    padding: 1rem;
    border-radius: 5px;
    margin: 1rem 0;
}
.prediction-high {
    background-color: #d4edda;
    border: 1px solid #c3e6cb;
    color: #155724;
}
.prediction-low {
    background-color: #fff3cd;
    border: 1px solid #ffeeba;
    color: #856404;
}
</style>
""", unsafe_allow_html=True)

# --- Functions ---
@st.cache_resource
def load_model():
    """Load and cache the model"""
    try:
        if not os.path.exists(MODEL_PATH):
            st.error(f"Model file not found at: {MODEL_PATH}")
            st.info("Please ensure you have run train_model.py first and the model file exists in the models directory.")
            return None
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.info(f"Current working directory: {os.getcwd()}")
        st.info(f"Looking for model at: {MODEL_PATH}")
        return None

def classify_image(image, model):
    """Classify the uploaded image"""
    try:
        # Preprocess the image
        img = image.resize((IMG_WIDTH, IMG_HEIGHT))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        # Make prediction
        prediction = model.predict(img_array)
        predicted_class_index = np.argmax(prediction[0])
        predicted_class = CLASS_NAMES[predicted_class_index]
        confidence = float(np.max(prediction[0]) * 100)

        return predicted_class, confidence
    except Exception as e:
        st.error(f"Error during classification: {e}")
        return None, None

# --- Main App ---
def main():
    # Title and Description
    st.title("🌍 CleanSort AI")
    st.markdown("""
    <p style='text-align: center; color: #666;'>
    Upload an image of waste material and let AI help you classify it for proper recycling!
    </p>
    """, unsafe_allow_html=True)

    # Load model
    model = load_model()
    if model is None:
        st.error("⚠️ Please ensure the model file exists and run train_model.py first.")
        return

    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image...", 
        type=["jpg", "jpeg", "png"],
        help="Upload a clear image of the waste item you want to classify"
    )

    # Create two columns for layout
    col1, col2 = st.columns(2)

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        col1.image(image, caption="Uploaded Image", use_container_width=True)  # Updated parameter

        # Add a classify button
        if col2.button("🔍 Classify Waste"):
            with st.spinner("Analyzing image..."):
                predicted_class, confidence = classify_image(image, model)

            if predicted_class and confidence:
                # Display results with custom styling
                if confidence >= 75:
                    col2.markdown(f"""
                    <div class='prediction prediction-high'>
                        <h3>Classification Result:</h3>
                        <p>Type: <strong>{predicted_class.upper()}</strong></p>
                        <p>Confidence: <strong>{confidence:.2f}%</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    col2.markdown(f"""
                    <div class='prediction prediction-low'>
                        <h3>Classification Result:</h3>
                        <p>Type: <strong>{predicted_class.upper()}</strong></p>
                        <p>Confidence: <strong>{confidence:.2f}%</strong></p>
                        <p><em>Note: Low confidence prediction</em></p>
                    </div>
                    """, unsafe_allow_html=True)

                # Display recycling instructions
                st.markdown("### ♻️ Recycling Instructions")
                instructions = {
                    'metal': "Clean and separate from other materials. Most metal items are highly recyclable!",
                    'organic': "Compost if possible. Keep separate from non-biodegradable waste.",
                    'paper': "Keep dry and flatten. Remove any plastic or metal attachments.",
                    'plastic': "Check the recycling number. Clean and dry before recycling."
                }
                st.info(instructions.get(predicted_class, ""))

    # Add information section
    with st.expander("ℹ️ About This Classifier"):
        st.markdown("""
        This waste classifier uses a deep learning model trained on various types of waste materials.
        It can classify waste into four categories:
        - 🔧 Metal
        - 🍂 Organic
        - 📄 Paper
        - 🏷️ Plastic
        
        For best results:
        - Use well-lit, clear images
        - Center the waste item in the image
        - Use a contrasting background
        """)

if __name__ == "__main__":
    main()