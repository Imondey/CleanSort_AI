import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np
import os

# --- Configuration ---
MODEL_PATH = '../models/waste_classifier.h5'
IMG_WIDTH, IMG_HEIGHT = 224, 224
# IMPORTANT: CLASS_NAMES order MUST match the order used during training.
# You can get this from train_generator.class_indices after training.
CLASS_NAMES = ['metal', 'organic', 'paper', 'plastic']

class WasteClassifierGUI:
    def __init__(self, master):
        self.master = master
        master.title("Waste Classification Prototype")

        self.model = self.load_model()
        self.current_image_path = None

        # --- GUI Elements ---
        self.label_title = Label(master, text="Upload Waste Image for Classification", font=("Arial", 16))
        self.label_title.pack(pady=10)

        self.btn_upload = Button(master, text="Upload Image", command=self.upload_image, font=("Arial", 12))
        self.btn_upload.pack(pady=5)

        self.image_panel = Label(master) # To display the uploaded image
        self.image_panel.pack(pady=10)

        self.label_result = Label(master, text="Prediction: N/A", font=("Arial", 14), fg="blue")
        self.label_result.pack(pady=5)

        self.label_confidence = Label(master, text="Confidence: N/A", font=("Arial", 12))
        self.label_confidence.pack(pady=5)

        # Pre-load a placeholder or explain no image is loaded
        self.display_placeholder()

    def load_model(self):
        """Loads the pre-trained TensorFlow model."""
        try:
            print("Loading model for GUI...")
            model = tf.keras.models.load_model(MODEL_PATH)
            print("Model loaded successfully.")
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            self.label_result.config(text="Error: Model not found or loaded.", fg="red")
            return None

    def display_placeholder(self):
        """Displays a simple placeholder when no image is loaded."""
        try:
            # Create a simple blank image or a text placeholder
            placeholder_img = Image.new('RGB', (IMG_WIDTH, IMG_HEIGHT), color='lightgray')
            placeholder_photo = ImageTk.PhotoImage(placeholder_img)
            self.image_panel.config(image=placeholder_photo)
            self.image_panel.image = placeholder_photo # Keep a reference
        except Exception as e:
            print(f"Error displaying placeholder: {e}")

    def upload_image(self):
        """Opens a file dialog to select an image and displays it."""
        file_path = filedialog.askopenfilename(
            initialdir=".",
            title="Select Image File",
            filetypes=(("Image files", "*.jpg *.jpeg *.png"), ("All files", "*.*"))
        )

        if file_path:
            self.current_image_path = file_path
            self.display_image(file_path)
            self.classify_image(file_path)

    def display_image(self, file_path):
        """Loads and displays the image in the GUI."""
        try:
            img = Image.open(file_path)
            img = img.resize((IMG_WIDTH, IMG_HEIGHT), Image.LANCZOS) # Use Image.LANCZOS for quality
            img_tk = ImageTk.PhotoImage(img)

            self.image_panel.config(image=img_tk)
            self.image_panel.image = img_tk # Keep a reference to prevent garbage collection
        except Exception as e:
            print(f"Error displaying image: {e}")
            self.label_result.config(text="Error: Could not display image.", fg="red")

    def classify_image(self, file_path):
        """Classifies the uploaded image using the loaded TensorFlow model."""
        if self.model is None:
            self.label_result.config(text="Model not loaded. Cannot classify.", fg="red")
            return

        try:
            # Load the image using Keras utility for consistency with training
            img = tf.keras.preprocessing.image.load_img(
                file_path, target_size=(IMG_WIDTH, IMG_HEIGHT)
            )
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) # Create a batch
            img_array /= 255.0 # Rescale to [0, 1]

            # Make prediction
            prediction = self.model.predict(img_array)
            predicted_class_index = np.argmax(prediction[0])
            predicted_class_name = CLASS_NAMES[predicted_class_index]
            confidence = np.max(prediction[0]) * 100

            # Update GUI with results
            self.label_result.config(text=f"Prediction: {predicted_class_name.upper()}", fg="green")
            self.label_confidence.config(text=f"Confidence: {confidence:.2f}%")

        except Exception as e:
            print(f"Error classifying image: {e}")
            self.label_result.config(text="Error during classification.", fg="red")
            self.label_confidence.config(text="Confidence: N/A")

if __name__ == "__main__":
    # Ensure the models directory exists
    if not os.path.exists(os.path.dirname(MODEL_PATH)):
        print(f"Error: Model directory '{os.path.dirname(MODEL_PATH)}' not found.")
        print("Please run 'train_model.py' first to create the model.")
    elif not os.path.exists(MODEL_PATH):
        print(f"Error: Model file '{MODEL_PATH}' not found.")
        print("Please run 'train_model.py' first to create the model.")
    else:
        root = tk.Tk()
        app = WasteClassifierGUI(root)
        root.mainloop()