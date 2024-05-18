import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
from about import about

# Load the model
model = load_model('fracture_model_NN.h5', compile=False)


# Function to preprocess the image
def preprocess_image(image):
    resized_image = cv2.resize(image, (225, 225))
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    return gray_image

# Function for edge detection
def detect_edges(image):
    edges = cv2.Canny(image, 100, 200)
    return edges

# Function to make a prediction
def classify_image(uploaded_file):
    image = Image.open(uploaded_file)
    img = np.array(image)  # Convert image to numpy array
    gray_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    resized_image = cv2.resize(gray_image, (225, 225))

    # Make prediction
    prediction = model.predict(np.expand_dims(resized_image, axis=0))

    # Interpret the prediction
    if prediction[0] >= 0.5:
        st.markdown("<span style='font-weight:bold;color:#FF5733;'>The bone contains a fracture.</span>", unsafe_allow_html=True)
    else:
        st.markdown("<span style='font-weight:bold;color:#1F618D;'>The bone does not contain a fracture.</span>", unsafe_allow_html=True)

# Function to reset uploaded file
def reset_uploaded_file():
    global uploaded_file
    uploaded_file = None

# Define the main content of the app
def main():
    st.title('BONE FRACTURE CLASSIFICATION')

    # Define the buttons
    load_image_button = st.empty()
    preprocessed_button = st.button('Preprocessed Image')
    edge_detection_button = st.button('Edge Detection')
    classify_button = st.button('Bone Fracture Result')
    reset_button = st.button('RESET')

    # Placeholder for image display
    original_image_placeholder = st.empty()
    processed_image_placeholder = st.empty()
    result_placeholder = st.empty()
    uploaded_file = None  # Initialize uploaded_file variable

    # Load image
    uploaded_file = load_image_button.file_uploader("Choose an image...", type=["jpg", "png"])
    if uploaded_file is not None:
        original_image = Image.open(uploaded_file)
        original_image_placeholder.image(original_image, caption='Original Image', use_column_width=False)

    # Preprocess image
    if preprocessed_button:
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            img = np.array(image)
            processed_image = preprocess_image(img)
            processed_image_placeholder.image(processed_image, caption='Preprocessed Image', use_column_width=False)
        else:
            st.warning("Please upload an image before preprocessing.")

    # Edge detection
    if edge_detection_button:
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            img = np.array(image)
            edges = detect_edges(img)
            processed_image_placeholder.image(edges, caption='Edge Detection', use_column_width=False)
        else:
            st.warning("Please upload an image before edge detection.")

    # Classify image
    if classify_button:
        if uploaded_file is not None:
            result = classify_image(uploaded_file)
            result_placeholder.text('Result: {}'.format(result))
        else:
            st.warning("Please upload an image before classifying.")

    # Reset uploaded file
    if reset_button:
        reset_uploaded_file()

# Create navigation
page = st.sidebar.selectbox("Select a page", ["Home", "About"])

if page == "Home":
    main()
elif page == "About":
    about()

