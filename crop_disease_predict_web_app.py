import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from PIL import Image
import base64

# Define the disease labels dictionary with plain English labels
disease_labels = {
    0: 'Apple Scab', 1: 'Apple Black Rot', 2: 'Apple Cedar Rust', 3: 'Healthy Apple',
    4: 'Healthy Blueberry', 5: 'Cherry Powdery Mildew', 6: 'Healthy Cherry',
    7: 'Corn Gray Leaf Spot', 8: 'Corn Common Rust',
    9: 'Corn Northern Leaf Blight', 10: 'Healthy Corn',
    11: 'Grape Black Rot', 12: 'Grape Black Measles', 13: 'Grape Leaf Blight',
    14: 'Healthy Grape', 15: 'Orange Citrus Greening', 16: 'Peach Bacterial Spot',
    17: 'Healthy Peach', 18: 'Bell Pepper Bacterial Spot', 19: 'Healthy Bell Pepper',
    20: 'Potato Early Blight', 21: 'Potato Late Blight', 22: 'Healthy Potato',
    23: 'Healthy Raspberry', 24: 'Healthy Soybean', 25: 'Squash Powdery Mildew',
    26: 'Strawberry Leaf Scorch', 27: 'Healthy Strawberry', 28: 'Tomato Bacterial Spot',
    29: 'Tomato Early Blight', 30: 'Tomato Late Blight', 31: 'Tomato Leaf Mold',
    32: 'Tomato Septoria Leaf Spot', 33: 'Tomato Spider Mites',
    34: 'Tomato Target Spot', 35: 'Tomato Yellow Leaf Curl Virus',
    36: 'Tomato Mosaic Virus', 37: 'Healthy Tomato'
}

# Load the trained model
model = tf.keras.models.load_model('trained_plant_disease_model.keras')

# Function to set a background image without affecting text visibility
def set_background(image_file):
    with open(image_file, "rb") as f:
        data = f.read()
    encoded = base64.b64encode(data).decode()
    background_style = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpeg;base64,{encoded}");
        background-size: cover;
    }}

    .stApp::before {{
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-color: rgba(255, 255, 255, 0.8); /* Light overlay to ensure text visibility */
        z-index: -1;
    }}
    </style>
    """
    st.markdown(background_style, unsafe_allow_html=True)

# Apply background to the Home tab
set_background("background.jpg")  # Replace with your background image file name

# Streamlit App with Tabs
st.title("Crop Disease Detection Web App")

# Tabs for navigation
home_tab, prediction_tab = st.tabs(["Home", "Crop Disease Identification"])

# Home Section
with home_tab:
    st.header("Welcome to the Crop Disease Detection System")
    st.markdown(
        """
        This application is a data-driven tool designed to assist farmers and agricultural experts in diagnosing and managing crop diseases. By analyzing images of crop leaves, the system uses machine learning to identify diseases and provide actionable insights.

        ### Why Use This Tool?
        - **Data-Driven Insights:** Leverages data and Machine-Learning to enhance decision-making in agriculture.
        - **Wide Crop Coverage:** Supports diagnosis for various crops including apples, tomatoes, potatoes, and more.
        - **Accessible and User-Friendly:** Simple interface that requires minimal technical knowledge.

        ### Current Crop Support:
        The system can diagnose diseases in the following crops:
        - Apples
        - Blueberries
        - Cherries
        - Corn
        - Grapes
        - Oranges
        - Peaches
        - Bell Peppers
        - Potatoes
        - Raspberries
        - Strawberries
        - Tomatoes

        More crops and disease categories will be added in future updates to expand the tool's capabilities and impact.

        ### How to Use the Application:
        1. Go to the **Crop Disease Identification** tab.
        2. Upload a clear image of a crop leaf using the file uploader.
        3. Wait for the system to process the image and display the prediction.
        4. Review the results and take appropriate action based on the diagnosis.
        """
    )

# Crop Disease Identification Section
with prediction_tab:
    st.header("Crop Disease Identification")

    # File uploader for image input
    uploaded_file = st.file_uploader("Upload an image of the crop leaf", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            # Read the image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image")

            # Preprocess the image
            image = image.resize((128, 128))
            image_array = img_to_array(image)
            image_array = np.expand_dims(image_array, axis=0) / 255.0

            # Predict using the model
            predictions = model.predict(image_array)
            predicted_index = np.argmax(predictions)
            predicted_label = disease_labels[predicted_index]

            # Display the result in bold
            st.success(f"**Prediction:** **{predicted_label}")
        except Exception as e:
            st.error(f"Error while processing the image: {e}")

