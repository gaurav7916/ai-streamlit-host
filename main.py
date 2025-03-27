import streamlit as st
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import re, os

torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)] 

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

# # model.eval()

# Load your trained PyTorch model
@st.cache_resource
def load_model():
    model = NeuralNetwork()
    try:
        model.load_state_dict(torch.load('model.pth', map_location=torch.device("cpu")))
    except RuntimeError as e:
        st.error(f"Error loading model: {e}")
        return None
    
    model.eval()
    return model

model = load_model()

if model is None:
    st.write("Model not found")
    st.stop()

# Define the class labels for Fashion MNIST
CLASS_NAMES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Image preprocessing function
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Grayscale(),  # Ensure single channel (1x28x28)
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image
    
# Function to process user input matrix
def preprocess_text_input(text):
    try:
        numbers = list(map(float, re.split(r'[ ,]+', text.strip())))  # Split by spaces or commas
        if len(numbers) != 28 * 28:
            st.error("Input must contain exactly 784 space- or comma-separated numbers.")
            return None, None
        image_array = np.array(numbers, dtype=np.float32).reshape(28, 28)
        image_tensor = torch.tensor(image_array).unsqueeze(0).unsqueeze(0)  # Add batch dimensions
        image_tensor = (image_tensor - 0.5) / 0.5  # Normalize
        return image_tensor, image_array
    except ValueError:
        st.error("Invalid input. Please enter only space- or comma-separated numbers.")
        return None, None

# Streamlit UI
st.title("Fashion MNIST Classifier")
st.write("Upload an image or enter pixel values, and the model will predict its class.")

# File upload section
uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('L')  # Convert to grayscale
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess and predict
    image_tensor = preprocess_image(image)
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)[0].numpy()
        predicted_class = np.argmax(probabilities)
        confidence = probabilities[predicted_class] * 100
    
    st.write(f"Prediction: **{CLASS_NAMES[predicted_class]}** ({confidence:.2f}% confidence)")

# Text input for manual pixel entry
st.write("Or manually enter a 28x28 pixel grid as space- or comma-separated values:")
user_input = st.text_area("Enter 784 space- or comma-separated pixel values:")

if user_input:
    image_tensor, image_array = preprocess_text_input(user_input)
    if image_tensor is not None:
        # Display grayscale preview
        fig, ax = plt.subplots()
        ax.imshow(image_array, cmap='gray')
        ax.axis('off')
        st.pyplot(fig)
        
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)[0].numpy()
            predicted_class = np.argmax(probabilities)
            confidence = probabilities[predicted_class] * 100
        
        st.write(f"Predicted class: {predicted_class}")
        st.write(f"Prediction from text input: **{CLASS_NAMES[predicted_class]}** ({confidence:.2f}% confidence)")