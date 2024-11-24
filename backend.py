import cv2
from flask import Flask, request, jsonify, render_template
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
import sqlite3

app = Flask(__name__)

# Inicializa la base de datos al cargar el servidor
def initialize_database():
    conn = sqlite3.connect('soil_crops.db')
    cursor = conn.cursor()

    # Crear tablas si no existen
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS soils (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE
    )
    ''')

    # Crear tabla crops si no existe
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS crops (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        season TEXT,
        difficulty TEXT
    )
    ''')

    # Crear tabla soil_crops si no existe
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS soil_crops (
        soil_id INTEGER,
        crop_id INTEGER,
        FOREIGN KEY(soil_id) REFERENCES soils(id),
        FOREIGN KEY(crop_id) REFERENCES crops(id)
    )
    ''')

    # Insertar datos iniciales en la tabla `soils`
    soils = ['Alluvial Soil', 'Black Soil', 'Clay Soil', 'Red Soil']
    for soil in soils:
        cursor.execute('INSERT OR IGNORE INTO soils (name) VALUES (?)', (soil,))

    conn.commit()
    conn.close()

initialize_database()  # Llamar al inicio del servidor

def get_crops_for_soil(soil_name):
    """Obtiene los cultivos recomendados para un tipo de suelo."""
    conn = sqlite3.connect('soil_crops.db')
    cursor = conn.cursor()

    # Query para obtener cultivos relacionados con el suelo dado
    query = '''
    SELECT crops.name, crops.season, crops.difficulty
    FROM crops
    JOIN soil_crops ON crops.id = soil_crops.crop_id
    JOIN soils ON soils.id = soil_crops.soil_id
    WHERE soils.name = ?
    '''
    cursor.execute(query, (soil_name,))
    results = cursor.fetchall()

    conn.close()

    # Formatear resultados en una lista de diccionarios
    return [{'name': row[0], 'season': row[1], 'difficulty': row[2]} for row in results]

# Define class labels (actualízalos si es necesario)
class_labels = ['Alluvial Soil', 'Black Soil', 'Clay Soil', 'Red Soil']

# Configurar el dispositivo para PyTorch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Cargar el modelo preentrenado
model = torch.load('best_model.pth', map_location=device)
model.to(device)
model.eval()

# Función para ajustar brillo y contraste
def adjust_brightness_contrast(img, alpha=1.0, beta=0):
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

# Preprocesar imágenes para el modelo
def preprocess_image(image, target_size=(299, 299)):
    img_array = np.array(image)
    img_resized = cv2.resize(img_array, target_size)

    # Aplicar suavizado y ajustes
    kernel = np.ones((3, 3), np.float32) / 9
    img_smoothed = cv2.filter2D(img_resized, -1, kernel)
    img_smoothed = cv2.bilateralFilter(img_smoothed, d=9, sigmaColor=75, sigmaSpace=75)
    img_bright_contrast = adjust_brightness_contrast(img_smoothed, alpha=1.2, beta=0.5)

    # Convertir a tensor y normalizar
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    return transform(img_bright_contrast).unsqueeze(0)

# Rutas de Flask
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        if not file or not file.filename.endswith(('jpg', 'jpeg', 'png')):
            return jsonify({'error': 'Invalid file type'}), 400

        # Preprocesar la imagen
        image = Image.open(file.stream).convert('RGB')
        processed_image = preprocess_image(image).to(device)

        # Hacer predicción
        with torch.no_grad():
            outputs = model(processed_image)
            _, predicted_class_idx = torch.max(outputs, 1)
            predicted_class = class_labels[predicted_class_idx.item()]

        # Obtener recomendaciones de cultivos
        recommendations = get_crops_for_soil(predicted_class)

        return jsonify({
            'prediction': predicted_class,
            'recommendations': recommendations
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/test')
def test():
    return "Flask is working!"

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)
