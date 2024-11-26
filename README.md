# MyCropProject
# **My Crop**

**My Crop** is a computer vision-based mobile app designed to classify soil types and provide recommendations for suitable crops. By leveraging a deep learning model (ResNet18), this app aids farmers in identifying four soil types—Alluvial, Black, Clay, and Red soil—offering crucial insights to improve agricultural productivity.

---

## **Features**
- Classifies soil images into four types: Alluvial, Black, Clay, and Red soil.
- Provides crop recommendations based on soil classification.
- Supports preprocessing steps to enhance soil texture analysis, such as noise reduction and brightness adjustment.
- Uses Fast Fourier Transform (FFT) to extract soil texture information.

---

## **Requirements**
Before running **My Crop**, ensure the following libraries and dependencies are installed:

### **Python Libraries**
- `numpy`
- `matplotlib`
- `opencv-python`
- `tensorflow` (for preprocessing)
- `torch`
- `torchvision`
- `flask` (for backend API)
- `flask-cors`
- `pillow`

---

## **How to Set Up and Run My Crop**

### **1. Clone the Repository**
```bash
git clone https://github.com/your-username/MyCrop.git
cd MyCrop
```

### **2. Install Dependencies**
It is recommended to use a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # For Linux/macOS
venv\Scripts\activate     # For Windows
```

Install the required Python libraries:
```bash
pip install -r requirements.txt
```

### **3. Model Training (Optional)**
If you'd like to train the model from scratch:
1. Prepare a dataset of soil images classified into Alluvial, Black, Clay, and Red soil.
2. Use the provided `train_model` function in the `model_training.ipynb` notebook.

### **4. Pre-Trained Model**
A pre-trained ResNet18 model is included in the repository for immediate use.

### **5. Running the App**
Start the Flask backend:
```bash
python backend.py
```

This will launch a local server (by default at `http://127.0.0.1:5000`).

### **6. Using the App**
1. Use a REST client (e.g., Postman) or the provided frontend to upload soil images.
2. The app will return the predicted soil type and recommended crops.

---

## **How It Works**
1. **Preprocessing**: The app processes input images using techniques like noise reduction, resizing, brightness adjustment, and FFT-based texture extraction.
2. **Classification**: A ResNet18 model predicts the soil type based on learned features like texture and color.
3. **Recommendations**: After classification, the app suggests crops suitable for the detected soil type.

---

## **Limitations**
- Currently supports classification for only four soil types: Alluvial, Black, Clay, and Red soil.
- Requires further development for real-time soil analysis and broader regional soil diversity.

---

## **Future Work**
- Extend support for more soil types and mixed soil classifications.
- Incorporate additional environmental factors like moisture and pH.
- Enhance app UI for better accessibility.

---

## **Contributions**
Feel free to contribute by creating issues, submitting pull requests, or suggesting new features!

---

## **License**
This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## **Contact**
For questions or feedback, contact [bruno.marquezra@udlap.mx].
