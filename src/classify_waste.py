import cv2
import numpy as np
import tensorflow as tf
# Uncomment the line below when running on Raspberry Pi
# import RPi.GPIO as GPIO
import time

# --- Configuration ---
MODEL_PATH = '../models/waste_classifier.h5'
IMG_WIDTH, IMG_HEIGHT = 224, 224
CLASS_NAMES = ['metal', 'organic', 'paper', 'plastic'] # IMPORTANT: Order must match training generator

# --- Raspberry Pi GPIO Setup (Placeholder) ---
# Replace with your actual GPIO pin numbers
PIN_PLASTIC = 17
PIN_PAPER = 27
PIN_METAL = 22
PIN_ORGANIC = 23

PINS = {
    'plastic': PIN_PLASTIC,
    'paper': PIN_PAPER,
    'metal': PIN_METAL,
    'organic': PIN_ORGANIC
}

def setup_gpio():
    # GPIO.setmode(GPIO.BCM)
    # for pin in PINS.values():
    #     GPIO.setup(pin, GPIO.OUT)
    #     GPIO.output(pin, GPIO.LOW)
    print("GPIO setup complete (simulation).")

def trigger_actuator(waste_type):
    """Triggers the correct actuator for the given waste type."""
    pin_to_activate = PINS.get(waste_type)
    if pin_to_activate:
        print(f"Triggering actuator for: {waste_type.upper()} on PIN {pin_to_activate}")
        # --- Real Hardware Code ---
        # GPIO.output(pin_to_activate, GPIO.HIGH)
        # time.sleep(1) # Keep actuator on for 1 second
        # GPIO.output(pin_to_activate, GPIO.LOW)
        # --------------------------
    else:
        print(f"Unknown waste type: {waste_type}")

def cleanup_gpio():
    # GPIO.cleanup()
    print("GPIO cleanup complete (simulation).")

# --- Main Application ---
if __name__ == '__main__':
    # 1. Load the trained TensorFlow model
    print("Loading model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully.")
    
    # 2. Setup GPIO
    setup_gpio()
    
    # 3. Initialize Camera
    cap = cv2.VideoCapture(0) # 0 is the default camera
    if not cap.isOpened():
        print("Error: Could not open camera.")
        exit()

    print("\nStarting waste classification... Press 'q' to quit.")
    
    try:
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            # 4. Preprocess the image for the model
            # Resize the frame
            img_resized = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))
            # Convert to array and rescale
            img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
            img_array = img_array / 255.0
            # Create a batch
            img_batch = np.expand_dims(img_array, axis=0)

            # 5. Make a prediction
            prediction = model.predict(img_batch)
            predicted_class_index = np.argmax(prediction[0])
            predicted_class_name = CLASS_NAMES[predicted_class_index]
            confidence = np.max(prediction[0]) * 100

            # 6. Take Action (if confidence is high)
            if confidence > 75: # Only trigger if confidence is over 75%
                trigger_actuator(predicted_class_name)

            # 7. Display the results on the frame
            display_text = f"{predicted_class_name}: {confidence:.2f}%"
            cv2.putText(frame, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Waste Sorter - Live Feed', frame)

            # Exit loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        # 8. Release resources
        cap.release()
        cv2.destroyAllWindows()
        cleanup_gpio()
        print("Application stopped.")