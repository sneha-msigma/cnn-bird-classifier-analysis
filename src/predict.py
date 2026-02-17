import torch
from torchvision import transforms
from PIL import Image
import sys

# from model import BasicCNN
# from config import Config

from .model import BasicCNN
from .config import Config



def predict_image(image_path):

    # Load model
    model = BasicCNN().to(Config.DEVICE)
    model.load_state_dict(torch.load("model.pth", map_location=Config.DEVICE))
    model.eval()

    # Same transforms used in training
    transform = transforms.Compose([
        transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
        transforms.ToTensor(),
    ])

    # Load image
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(Config.DEVICE)

    # Predict
    with torch.no_grad():
        output = model(image)
        prob = torch.sigmoid(output)
        prediction = 1 if prob.item() > 0.5 else 0

    if prediction == 1:
        print("Prediction: Bird")
    else:
        print("Prediction: No Bird")

    print(f"Confidence: {prob.item():.4f}")


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage: python src/predict.py <image_path>")
    else:
        predict_image(sys.argv[1])
