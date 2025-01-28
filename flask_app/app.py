from flask import Flask, request, render_template, jsonify, send_file
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
from pathlib import Path
import io
from matplotlib.figure import Figure

app = Flask(__name__)

MODEL_PATH = '/workspaces/breast_cancer_diagnosis/flask_app/model.h5'
IMAGE_DIM = 128
CLASS_NAMES = ['Image tested Negative for IDC', 'Image tested Positive for IDC']

# Load the trained model
model = None
if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)
    print("Model loaded successfully!")
else:
    print("Error: Model file not found. Please ensure 'model.h5' is in the correct directory.")

# Helper function to preprocess images
def preprocess_image(img_path, image_dim):
    img = image.load_img(img_path, target_size=(image_dim, image_dim))
    img = image.img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded.'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected.'})

    if file:
        # Save the uploaded image
        file_path = os.path.join('uploads', file.filename)
        Path('uploads').mkdir(parents=True, exist_ok=True)
        file.save(file_path)

        # Preprocess the image and make a prediction
        img = preprocess_image(file_path, IMAGE_DIM)
        prediction = model.predict(img)[0][0]  # Single prediction value
        class_idx = int(prediction > 0.5)  # Binary classification threshold
        class_label = CLASS_NAMES[class_idx]

        # Remove the uploaded file
        os.remove(file_path)

        # Generate a bar chart dynamically
        fig = Figure()
        ax = fig.subplots()
        ax.bar(['Confidence'], [prediction], color=['blue'])
        ax.set_ylim(0, 1)
        ax.set_title(f'Prediction: {class_label}')
        ax.set_ylabel('Confidence')

        # Save the graph to an in-memory buffer
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)

        # Render results in a template
        return render_template(
            'result.html',
            prediction=class_label,
            confidence=f'{prediction:.2%}',
            graph_url='/graph'
        )

@app.route('/graph')
def graph():
    # Serve the graph dynamically
    buf = io.BytesIO()
    fig = Figure()
    ax = fig.subplots()
    ax.bar(['Confidence'], [0.5], color=['blue'])  # This can be updated dynamically
    ax.set_ylim(0, 1)
    ax.set_title('Dynamic Graph')
    fig.savefig(buf, format="png")
    buf.seek(0)
    return send_file(buf, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
