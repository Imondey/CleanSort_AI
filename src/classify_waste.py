import cv2
import numpy as np
import tensorflow as tf
import time
import signal
import sys
import os  # Add this import
from threading import Event

# --- Configuration ---
MODEL_PATH = '../models/waste_classifier.h5'
IMG_WIDTH, IMG_HEIGHT = 224, 224
CLASS_NAMES = ['metal', 'organic', 'paper', 'plastic']

# Window configuration
WINDOW_NAME = 'CleanSort AI - Waste Classifier'
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, 800, 600)

# Control flags
exit_event = Event()

def signal_handler(signum, frame):
    """Handle cleanup on signal"""
    print("\nShutting down gracefully...")
    exit_event.set()

def create_info_panel(frame, prediction_info=None):
    """Create an information panel on the frame"""
    height, width = frame.shape[:2]
    panel_height = 80
    panel = np.zeros((panel_height, width, 3), dtype=np.uint8)
    
    # Add basic instructions
    cv2.putText(panel, "Press 'q' to quit", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    if prediction_info:
        class_name, confidence = prediction_info
        # Color based on confidence
        color = (0, 255, 0) if confidence > 75 else (0, 165, 255)
        cv2.putText(panel, f"Detected: {class_name} ({confidence:.1f}%)", 
                    (width // 2 - 100, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, color, 2)
    
    return np.vstack([panel, frame])

def process_frame(frame, model):
    """Process a single frame and return prediction"""
    try:
        img_resized = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))
        img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
        img_array = img_array / 255.0
        img_batch = np.expand_dims(img_array, axis=0)
        
        prediction = model.predict(img_batch, verbose=0)
        predicted_class_index = np.argmax(prediction[0])
        confidence = float(np.max(prediction[0]) * 100)
        
        return CLASS_NAMES[predicted_class_index], confidence
    except Exception as e:
        print(f"Error processing frame: {e}")
        return None, 0

def main():
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    cap = None
    try:
        # Load model
        print("Loading model...")
        if not os.path.exists(MODEL_PATH):
            print(f"Error: Model not found at {MODEL_PATH}")
            return
        
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Model loaded successfully")
        
        # Initialize camera
        print("Initializing camera...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Could not open camera")
        
        # Set camera properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("\nStarting waste classification...")
        last_prediction_time = time.time()
        prediction_info = None
        
        while not exit_event.is_set():
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            # Process every 0.5 seconds to avoid overload
            current_time = time.time()
            if current_time - last_prediction_time > 0.5:
                class_name, confidence = process_frame(frame, model)
                if class_name:
                    prediction_info = (class_name, confidence)
                last_prediction_time = current_time
            
            # Create and display the enhanced frame
            display_frame = create_info_panel(frame, prediction_info)
            cv2.imshow(WINDOW_NAME, display_frame)
            
            # Check for 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nUser requested quit")
                break
                
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        tf.keras.backend.clear_session()
        print("Application closed successfully")

if __name__ == "__main__":
    main()