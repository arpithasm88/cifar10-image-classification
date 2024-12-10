# CIFAR-10 Image Classification Web App

This project is built using **Streamlit** for image classification. It uses a **Convolutional Neural Network (CNN)** model trained on the **CIFAR-10 dataset** to classify images into 10 different categories.

## Features

- **Image Upload**: Upload your image and get the predicted class.
- **Model**: The model classifies the image into one of the 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.
- **Confidence Score**: The app shows the predicted class along with the confidence percentage.

## Technology Stack

- **Backend**: TensorFlow, Keras
- **Frontend**: Streamlit
- **Libraries**:
  - `tensorflow` for building and training the model.
  - `streamlit` for the web app interface.
  - `PIL` (Python Imaging Library) for image manipulation.
  - `numpy` for data processing.
  - `matplotlib` for visualizations.

## Prerequisites

Before running the application, ensure you have the following installed:

- Python 3.x
- TensorFlow 2.x
- Streamlit
- Pillow (PIL)
- Numpy

You can install the necessary libraries using `pip`:

```bash
pip install tensorflow streamlit pillow numpy
```
**Setup and Installation**

**1. Clone this repository:**

```bash
git clone https://github.com/arpithasm88/cifar10-image-classification.git
cd cifar10-image-classification
```
**2.Load or train the CIFAR-10 model:**

* If you already have the trained model (cifar10_cnn_model.h5), make sure it is in the project directory.
  
* If not, you can train the model using the script provided in the project or download a pre-trained model.
  
**Run the Streamlit app:**

```bash

streamlit run app.py
```

This will start the app, and you can open it in your browser.


**How It Works**

**1. Model Training**

The model is built using the CIFAR-10 dataset, which contains 60,000 32x32 images across 10 classes. The CNN model is trained using TensorFlow/Keras with the following architecture:

**Convolutional layers** for feature extraction.

**MaxPooling layers** for down-sampling.

**Dense layers** for classification.

**Dropout layer** for regularization.

**2. Image Classification**

Once the model is trained, the application allows users to upload an image, preprocess it, and use the trained model to predict the class of the image.

**3. Web App**

The web app uses Streamlit to provide a simple interface for users to upload images. The app shows the predicted class along with the confidence score.

**Project Structure**

```bash
├── app.py              # Streamlit web app
├── cifar10_cnn_model.h5 # Pre-trained model (if available)
├── README.md           # Project documentation
└── requirements.txt     # List of dependencies
```

**Usage**

**Upload an Image:** Click on "Choose an image..." and select an image (JPG, PNG, or JPEG).

**Classify:** The model will predict the class of the uploaded image and display the predicted label along with the confidence score.

**Model Details**

The CNN model has the following architecture:

* **Convolutional Layers:** Extract features from the image.

* **MaxPooling Layers:** Reduce the spatial size of the representation.

* **Flatten:** Convert the 2D matrix into a 1D vector.
  
* **Dense Layers:** Fully connected layers for classification.
  
* **Dropout Layer:** Reduces overfitting by randomly setting some neurons to zero during training.

The model is trained using Adam Optimizer and Sparse Categorical Cross-Entropy Loss for multi-class classification.

**License**

This project is licensed under the MIT License.

**Author**

Arpitha S M

