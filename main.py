from flask import Flask, request, jsonify
from PIL import Image

from scripts.inference.predict import Predictor
from utils.class_mapping import map_classes

app = Flask(__name__)

# Load the model
model = None  # Load your trained model here

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})

        file = request.files['file']
        image = Image.open(file).convert("RGB")

        # Make prediction using the model
        predictor = Predictor()
        preds = predictor.predict(image)

        response = {}

        for i, c in enumerate(map_classes()):
            print(f"{c}: {preds[i]}")
            response[c] = preds[i]

        return jsonify({'predictions': response})  # Assuming prediction is a single value

@app.route('/')
def index():
    return "Hello, World!"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)