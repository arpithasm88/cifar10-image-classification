import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

def cifar10_classification():
    st.title("CIFAR-10 Image Classification")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        st.write("Classifying...")

        # Load the saved model
        model = tf.keras.models.load_model('cifar10_cnn_model.keras')

        # Compile the model manually (needed because the model was saved without its configuration)
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        # Preprocess the uploaded image
        img = image.resize((32, 32))  # Resize to CIFAR-10 dimensions (32x32)
        img_array = np.array(img)
        img_array = img_array.astype('float32') / 255.0  # Normalize the image
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Make predictions
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)[0]
        confidence = np.max(predictions)

        cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        st.write(f"Predicted Class: {cifar10_classes[predicted_class]}")
        st.write(f"Confidence: {confidence * 100:.2f}%")

def main():
    st.set_page_config(page_title="CIFAR-10 Image Classification", page_icon=":guardsman:", layout="wide")

    # Sidebar Styling
    st.sidebar.title("Navigation")
    choice = st.sidebar.selectbox("Choose Model", ("CIFAR-10"))

    # About section with better design
    st.sidebar.write("### About")
    st.sidebar.write("""
    This web app uses a CNN model trained on the CIFAR-10 dataset to classify images.
    The model can identify objects such as airplanes, birds, cats, and more.
    """)

    if choice == "CIFAR-10":
        cifar10_classification()

    # Custom CSS for better design
    st.markdown("""
        <style>
        body {
            background-color: #F4E1D2;  /* Creamy background */
            color: #4E3629;  /* Dark brown text */
            font-family: 'Roboto', sans-serif;
        }

        .stButton button {
            background-color: #F4E1D2;  /* Cream button */
            color: #4E3629;  /* Dark brown text */
            border: none;
            padding: 10px 20px;
            border-radius: 8px;
            font-size: 16px;
            cursor: pointer;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
            transition: all 0.3s ease;
        }

        .stButton button:hover {
            background-color: #D2B89B;  /* Light brown on hover */
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.3);
        }

        .stFileUploader {
            background-color: #6A4E23;  /* Brown background for file upload */
            border: 2px dashed #D2B89B;  /* Light brown dashed border */
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 30px;
            color: #F4E1D2;  /* Cream text */
        }

        .stFileUploader:hover {
            background-color: #8B5A2B;  /* Lighter brown on hover */
            border: 2px dashed #F4E1D2;  /* Cream dashed border */
        }

        .stSelectbox, .stTextInput {
            background-color: #6A4E23;  /* Brown background for inputs */
            color: #F4E1D2;  /* Cream text */
            border-radius: 8px;
            padding: 10px;
            border: none;
        }

        .stSelectbox:focus, .stTextInput:focus {
            outline: none;
            border-color: #D2B89B;  /* Light brown border on focus */
        }

        .stSidebar {
            background-color: #6A4E23;  /* Brown background for sidebar */
            color: #F4E1D2;  /* Cream text */
            font-family: 'Roboto', sans-serif;
        }

        .stMarkdown p {
            font-size: 18px;
        }

        .footer {
            position: fixed;
            bottom: 10px;
            left: 0;
            right: 0;
            text-align: center;
            font-size: 14px;
            color: #4E3629;  /* Dark brown text */
            background-color: #F4E1D2;  /* Cream background */
            padding: 10px;
            border-radius: 5px;
        }

        h1, h2 {
            font-family: 'Arial', sans-serif;
        }

        </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="footer">Powered by Arpitha S M</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()

