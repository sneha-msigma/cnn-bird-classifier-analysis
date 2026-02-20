# src/analysis.py

import os
import torch
from PIL import Image
from torchvision import transforms

from src.model import BasicCNN
from src.config import Config
from src.activations import register_conv_hooks
from src.visualization import save_layer_as_grid


def run_filter_visualization(image_path):

    image_name = os.path.splitext(os.path.basename(image_path))[0]
    base_output_dir = os.path.join("filter_maps", image_name)

    os.makedirs(base_output_dir, exist_ok=True)

    model = BasicCNN().to(Config.DEVICE)
    model.load_state_dict(
        torch.load("model.pth", map_location=Config.DEVICE)
    )
    model.eval()

    activations = register_conv_hooks(model)

    transform = transforms.Compose([
        transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
        transforms.ToTensor(),
    ])

    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(Config.DEVICE)

    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.sigmoid(output)
        confidence = prob.item()

    prediction = 1 if confidence > 0.5 else 0

    print("=" * 50)
    print(f"Image: {image_path}")
    print(f"Prediction: {'Bird' if prediction else 'No Bird'}")
    print(f"Confidence: {confidence:.4f}")
    print("=" * 50)

    for layer_name, activation in activations.items():
        save_layer_as_grid(activation, layer_name, base_output_dir)