import os
import io
import time
import logging
from flask import Flask, request, render_template, jsonify
import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.models import ResNet50_Weights
from PIL import Image

# Set up logging
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Correctly handle the model path
model_path = os.path.join(os.path.dirname(__file__), 'models', 'best_bone_cancer_model.pth')
app.logger.info(f"Loading model from {model_path}")

# Load the model
model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 3)  # Adjust for your 3 classes

try:
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()  # Set the model to evaluation mode
    app.logger.info("Model loaded successfully")
except Exception as e:
    app.logger.error(f"Failed to load the model: {str(e)}")

# Define the transformation to match training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict_image(image_file):
    start_time = time.time()
    
    # Read the image file
    image = Image.open(image_file).convert('RGB')
    load_time = time.time() - start_time
    app.logger.info(f"Image load time: {load_time:.2f} seconds")

    # Preprocess the image and add batch dimension
    image = transform(image).unsqueeze(0).to(device)
    
    # Make predictions
    with torch.no_grad():
        output = model(image)
        predicted = output.argmax(dim=1).item()
        probabilities = nn.Softmax(dim=1)(output).cpu().numpy()[0]  # Get probabilities

    # Mapping the prediction to class labels
    class_names = ['Chondrosarcoma', 'Ewing Sarcoma', 'Osteosarcoma']
    return class_names[predicted], probabilities

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        if file:
            # Read the file into memory
            in_memory_file = io.BytesIO()
            file.save(in_memory_file)
            in_memory_file.seek(0)
            
            # Make prediction
            try:
                prediction, probabilities = predict_image(in_memory_file)

                # Convert probabilities to standard Python floats
                probabilities = [float(prob) for prob in probabilities]

                return jsonify({
                    'prediction': prediction,
                    'scores': {
                        'Chondrosarcoma': probabilities[0],
                        'Ewing Sarcoma': probabilities[1],
                        'Osteosarcoma': probabilities[2]
                    }
                }), 200  # Ensure a successful response
            except Exception as e:
                app.logger.error(f"Prediction failed: {str(e)}")
                return jsonify({'error': 'Prediction failed', 'details': str(e)}), 500

    # Return the main HTML page on GET request
    return render_template('app.html')

@app.errorhandler(Exception)
def handle_exception(e):
    # Log the error details
    app.logger.error(f"An internal error occurred: {str(e)}")
    
    # Return a proper JSON response with the error
    return jsonify({
        'error': 'An internal server error occurred.',
        'details': str(e)  # This provides the actual exception message
    }), 500


if __name__ == '__main__':
    # Ensure debug mode is off in production
    app.run(host='0.0.0.0', port=5000, debug=False)
