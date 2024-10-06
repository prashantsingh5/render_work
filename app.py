import os
from flask import Flask, request, render_template, jsonify
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import io

app = Flask(__name__)

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the pre-trained ResNet50 model and modify the output layer
model = models.resnet50(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 3)  # Adjust for your 3 classes
model.load_state_dict(torch.load(r'C:\Users\pytorch\Desktop\bone_cancer_detection\best_bone_cancer_model.pth', map_location=device))
model.to(device).eval()  # Set the model to evaluation mode

# Define the transformation to match training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict_image(image_file):
    # Read the image file
    image = Image.open(image_file).convert('RGB')
    
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

    # Return the main HTML page on GET request
    return render_template('app.html')


if __name__ == '__main__':
    app.run(debug=True)
