from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog

# Load the model
model = load_model("pneumonia_model_v1.h5")

def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((224, 224))  # Match training size
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

def predict_pneumonia(image_path):
    img_array = preprocess_image(image_path)
    prediction = model.predict(img_array)[0][0]
    if prediction > 0.5:
        return "PNEUMONIA", prediction
    else:
        return "NORMAL", prediction

def browse_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        # Show image
        img = Image.open(file_path)
        img = img.resize((400, 400))
        tk_img = ImageTk.PhotoImage(img)
        img_label.configure(image=tk_img)
        img_label.image = tk_img  # Prevent garbage collection

        # Predict
        label, confidence = predict_pneumonia(file_path)
        result_text = f"Prediction: {label}\n"
        result_label.config(text=result_text, font=("Arial", 20, "bold"), fg="#333")

# Main Window
root = tk.Tk()
root.title("Pneumonia Detection")
root.geometry("1500x1000")
root.configure(bg="#F0F8FF")  # Light Blue Background

# Button
btn = tk.Button(root, text="Choose X-ray Image", command=browse_image,
                bg="#4CAF50", fg="white", font=("Arial", 20), padx=20, pady=10)
btn.place(relx=0.5, rely=0.1, anchor=tk.CENTER)

# Image Display
img_label = tk.Label(root, bg="#F0F8FF")
img_label.place(relx=0.5, rely=0.45, anchor=tk.CENTER)

# Result Display
result_label = tk.Label(root, text="", bg="#F0F8FF")
result_label.place(relx=0.5, rely=0.8, anchor=tk.CENTER)

root.mainloop()
