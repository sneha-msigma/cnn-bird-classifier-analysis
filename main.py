import torch
import mlflow
import os
from src.data import prepare_data_loaders
from src.model import BirdClassifierCNN
from src.train import train_model
from src.evaluate import evaluate_model
from src.config import EXPERIMENT_NAME, RUN_NAME, NUM_CLASSES

def main():
    """
    Main entry point for building and evaluating the Binary Bird Classifier.
    """
    # Set execution device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Step 1: Prepare data streams
    print("Loading data from local system...")
    train_loader, test_loader, class_names = prepare_data_loaders()
    
    if train_loader is None or not class_names:
        print("Dataset not found or empty. Please check your DATA_DIR.")
        return

    print(f"Detected classes: {class_names}")
    # Number of classes in dataset might differ from NUM_CLASSES if data is messy
    actual_num_classes = len(class_names)

    # Step 2: Initialize model
    model = BirdClassifierCNN(num_classes=actual_num_classes).to(device)

    # Step 3: Initialize MLflow and Train
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    with mlflow.start_run(run_name=RUN_NAME):
        print("\nStarting Training Phase...")
        model = train_model(model, train_loader, test_loader, device)
        
        # Step 4: Evaluate
        print("\nStarting Evaluation Phase...")
        evaluate_model(model, test_loader, class_names, device)
        
    print("\nProcess complete. Check MLflow for detailed metrics.")

if __name__ == "__main__":
    main()
