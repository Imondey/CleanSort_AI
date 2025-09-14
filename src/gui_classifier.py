import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
import os
import sys

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

def cleanup_and_exit():
    """Clean up resources and exit the application"""
    try:
        # Clear TensorFlow session
        tf.keras.backend.clear_session()
        # Stop Streamlit
        try:
            st.runtime.scriptrunner.add_script_run_ctx.get().stop()
        except:
            pass
        # Force exit the Python process
        os._exit(0)
    except:
        sys.exit(0)

# --- Main App ---
def main():
    try:
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
                        'metal': """
        🔧 Metal Recycling Instructions:
        1. Clean thoroughly - remove food residue and debris
        2. Separate different types of metals (aluminum, steel, etc.)
        3. Crush cans to save space (if applicable)
        4. Remove non-metal attachments or labels
        5. Keep electronics separate - they need special recycling
        6. Check for recycling symbols on metal containers
        7. Don't put in sharp metal objects without proper wrapping
        
        ✅ Recyclable: Cans, foil, metal bottles, clean containers
        ❌ Not Recyclable: Paint cans, pressurized containers, hazardous material containers
    """,
    
    'organic': """
        🍂 Organic Waste Instructions:
        1. Separate food scraps from packaging
        2. Use dedicated organic waste bins
        3. Layer with dry materials (leaves, paper) to reduce odor
        4. Keep meat/dairy separate if required by local regulations
        5. Break down large pieces for faster composting
        6. Maintain proper moisture level
        7. Avoid contamination with non-organic materials
        
        ✅ Compostable: Food scraps, yard waste, coffee grounds, tea bags
        ❌ Not Compostable: Meat bones, oils, dairy products (in home composting)
    """,
    
    'paper': """
        📄 Paper Recycling Instructions:
        1. Keep paper clean and dry
        2. Remove plastic windows from envelopes
        3. Flatten all boxes and cartons
        4. Remove tape, staples, and metal bindings
        5. Separate glossy magazines from regular paper
        6. Don't shred unless necessary (reduces recyclability)
        7. Store in dry location until disposal
        
        ✅ Recyclable: Newspapers, cardboard, office paper, magazines
        ❌ Not Recyclable: Greasy/food-stained paper, wax paper, thermal receipts
    """,
    
    'plastic': """
        🏷️ Plastic Recycling Instructions:
        1. Check the recycling number (1-7) on the bottom
        2. Clean and rinse thoroughly
        3. Remove all labels and adhesives
        4. Compress to save space
        5. Keep caps and lids separate
        6. Group by plastic type if possible
        7. Check local recycling guidelines for accepted types
        
        ✅ Recyclable: PET bottles, HDPE containers, rigid plastics
        ❌ Not Recyclable: Plastic bags, styrofoam, contaminated containers
    """
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
    except (KeyboardInterrupt, SystemExit):
        cleanup_and_exit()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        cleanup_and_exit()
    finally:
        # Cleanup resources
        if 'model' in locals():
            del model
        tf.keras.backend.clear_session()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        cleanup_and_exit()