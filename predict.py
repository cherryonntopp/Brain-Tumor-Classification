import tensorflow as tf
import numpy as np

# Load model
loaded_model = tf.keras.models.load_model("final_model.keras")

# Define constants 
_, height, width, channels = loaded_model.input_shape
IMG_SIZE = height
class_names = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]  

# Define tumor information
tumor_info = {
    "Glioma": "Glioma tumors start in the glial cells (support cells in the brain and spinal cord). Common locations include the cerebrum, brain stem, and spinal cord.",
    "Meningioma": "Meningioma tumors arise from the meninges (the protective membranes covering the brain and spinal cord). Common locations include the surface of the brain, skull base, and behind the eyes. ",
    "No Tumor": "The MRI scan does not show the presence of a tumor.",
    "Pituitary": "Pituitary tumors develop in the pituitary gland (a small gland at the base of the brain that controls hormones). These are often benign, but they can affect many body systems because of hormone control. "
}

def predict_image(pil_img):
    # Convert PIL image into model input
    img = pil_img.resize((IMG_SIZE, IMG_SIZE))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    # Make prediction
    predictions = loaded_model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_names[predicted_class_index]
    confidence = float(predictions[0][predicted_class_index]) * 100

    # Get tumor info
    info = tumor_info.get(predicted_class_name, "No information available.")

    return predicted_class_name, confidence, info
