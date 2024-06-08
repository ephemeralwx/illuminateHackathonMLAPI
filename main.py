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

