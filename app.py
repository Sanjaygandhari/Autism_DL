import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
# Load the trained model
@st.cache(allow_output_mutation=True)  # Cache the model to avoid reloading on every run
def load_model():
    model = tf.keras.models.load_model("/content/my_model.h5")  # Provide the path to your saved model
    return model

model = load_model()
# Function to preprocess and make predictions
def preprocess_and_predict(image_data, model):
    # Convert image to RGB (if needed)
    image = Image.open(image_data).convert("RGB")
    # Resize the image to (224, 224) as expected by the model
    image = image.resize((224, 224))
    # Convert the image to a numpy array
    image_array = np.asarray(image) / 255.0  # Normalize pixel values to [0, 1]
    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)
    # Make prediction
    prediction = model.predict(image_array)
    return prediction
# Streamlit UI
st.title("Prediction of Autism Spectrum Condition")
st.text("Please upload an image file:")
# File uploader
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    st.write("Classifying...")

    # Preprocess and predict
    predictions = preprocess_and_predict(uploaded_file, model)
    score = tf.nn.softmax(predictions[0])  # Apply softmax to interpret as probabilities

    # Define class names
    class_names = ['autistic', 'non_autistic']  # Replace with actual class names
    predicted_class = class_names[np.argmax(score)]
    confidence = 100 * np.max(score)

    # Display prediction
    st.success(f"This image is most likely: **{predicted_class}** with a confidence of {confidence:.2f}%")
