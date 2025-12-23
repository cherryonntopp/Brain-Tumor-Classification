import tensorflow as tf
import gradio as gr
import numpy as np
from PIL import Image
from predict import predict_image

# Load model
model = tf.keras.models.load_model("final_model.keras")

# Class names
class_names = ["glioma", "meningioma", "no_tumor", "pituitary"]

def analyze_image(image):
    if image is None:
        return "Please upload an image."
    image = image.convert("RGB")
    result, confidence, info = predict_image(image)
    return (
        f"🧠 Prediction: {result}\n\n"
        f"📊 Confidence: {confidence:.2f}%\n\n"
        f"ℹ️ Info: {info}"
    )

# Gradio UI
demo = gr.Interface(
    fn=analyze_image,
    inputs=gr.Image(type="pil", label="Upload MRI Scan"),
    outputs=gr.Textbox(label="Analysis Result"),
    title="BrainScan AI",
    description="Upload a brain MRI image to detect tumor type using a deep learning model."
)

if __name__ == "__main__":
    demo.launch()
