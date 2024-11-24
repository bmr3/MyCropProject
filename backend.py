import cv2
from flask import Flask, request, jsonify, render_template
from PIL import Image  # Import PIL Image for handling the file
import numpy as np
import torch
import torchvision.transforms as transforms

app = Flask(__name__)

# Define class labels (update based on your model's training labels)
class_labels = ['Alluvial Soil', 'Black Soil', 'Clay Soil', 'Red Soil']

# Load the pre-trained model (PyTorch version)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Cargar el modelo completo
model = torch.load('best_model.pth', map_location=device)
model.to(device)
model.eval()

def adjust_brightness_contrast(img, alpha=1.0, beta=0):
    """Ajustar brillo y contraste.
    alpha > 1 aumenta el contraste, alpha < 1 lo reduce.
    beta > 0 aumenta el brillo, beta < 0 lo disminuye.
    """
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

def preprocess_image(image, target_size=(299, 299)):
    img_array = np.array(image)
    img_resized = cv2.resize(img_array, target_size)

    # Suavizado con kernel 3x3 y filtro bilateral
    kernel = np.ones((3, 3), np.float32) / 9
    img_smoothed = cv2.filter2D(img_resized, -1, kernel)
    img_smoothed = cv2.bilateralFilter(img_smoothed, d=9, sigmaColor=75, sigmaSpace=75)

    # Ajustar brillo y contraste
    img_bright_contrast = adjust_brightness_contrast(img_smoothed, alpha=1.2, beta=0.5)

    # Normalizar y convertir a tensor
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    return transform(img_bright_contrast).unsqueeze(0)

@app.route('/')
def home():
    return render_template('index.html')  # Load HTML frontend

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if not file or not file.filename.endswith(('jpg', 'jpeg', 'png')):
        return jsonify({'error': 'Invalid file type'}), 400

    # Preprocess the image
    image = Image.open(file.stream).convert('RGB')  # Ensure the image is in RGB
    processed_image = preprocess_image(image).to(device)

    # Make prediction
    with torch.no_grad():
        outputs = model(processed_image)
        _, predicted_class_idx = torch.max(outputs, 1)
        predicted_class = class_labels[predicted_class_idx.item()]

    return jsonify({'prediction': predicted_class})

@app.route('/test')
def test():
    return "Flask is working!"

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)