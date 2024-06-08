from flask import Flask, request, jsonify
from PIL import Image
import torch
from torchvision import transforms
from flask_cors import CORS, cross_origin
import torch.nn.functional as F



app = Flask(__name__)
CORS(app, origins=["http://localhost:5173", "http://localhost:5174"])

def load_trained_model(model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load(model_path, map_location=device)
    model = model.to(device)
    model.eval()
    return model

#this model should be the right model idk check file loc
model_path = "brain_tumor_model_include_four_classification_confidence.pth"
model = load_trained_model(model_path)

#image fixing
transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),

        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

#image prepro

def preprocess_image(image_bytes):
    image=Image.open(image_bytes)
    image=transform(image).unsqueeze(0)
    return image


def predict_class(image_bytes):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    
    image = preprocess_image(image_bytes)
    image = image.to(device)

    #TESTING ;
    with torch.no_grad():
        outputs = model(image)
        probabilities = F.softmax(outputs, dim=1)

        predicted = torch.argmax(probabilities, dim=1)

        predicted_class = predicted.item()
        confidence = probabilities[0][predicted_class].item()

    classes = ['Normal', 'Glioma', 'Meningioma', 'Pituitary']
    return classes[predicted.item()], confidence



#FLASK
@app.route('/predict', methods=['POST'])
@cross_origin() 
#CORS

def predict():
    if 'file' not in request.files:

        return jsonify({'error': 'no file given??'}), 400
    file = request.files['file']
    if file is None or file.filename == "":
        return jsonify({'error': 'no file given??'}), 400
    try:
        predicted_class, confidence = predict_class(file)
        return jsonify({'prediction': predicted_class, 'confidence': confidence})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
#run
if __name__ == '__main__':
    app.run(debug=True, port=5000)