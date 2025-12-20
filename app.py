import os
import tensorflow as tf
import predict
from PIL import Image
from flask import Flask, render_template, send_from_directory, request, jsonify
import numpy as np
from predict import predict_image
import gradio as gr

app = Flask(__name__, static_url_path='/static', static_folder='static')

# Load model
loaded_model = tf.keras.models.load_model("final_model.keras")

# Define constants (update based on your training setup)
IMG_SIZE = 224
class_names = ["glioma", "meningioma", "no_tumor", "pituitary"]  # Example


# Set up a route to serve the main HTML page
@app.route('/')
def index():
    return render_template('index.html') # Homepage 

@app.route('/analyze')
def analyze():
    return render_template('analyze.html') # Analysis page 

@app.route("/predict", methods=["POST"])
def predict_route():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400
        
        file = request.files["image"]
        img = Image.open(file.stream).convert("RGB")
        
        result, confidence, info = predict_image(img)

        return jsonify({
            "prediction": result, 
            "confidence": f"{confidence:.2f}%",
            "info": info
            })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
# Gradio for hugging free space 
def greet(name):
    return "Hello " + name + "!!"

demo = gr.Interface(fn=greet, inputs="text", outputs="text")
demo.launch()
    

# Set up a route to serve the model files
@app.route('/model/<path:filename>')
def serve_model(filename):
    """Serves the TensorFlow.js model files from the 'model' directory."""
    return send_from_directory('model', filename)

if __name__ == '__main__':
    app.run()
