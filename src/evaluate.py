import torch
import torch.nn.functional as F
import mlflow
import os
from .config import OUTPUT_DIR

def evaluate_model(model, test_loader, classes, device):
    """
    Evaluates the model on the test set and prints accuracy per class.
    Logs final results to MLflow.
    """
    model.eval()
    if not classes:
        print("No classes found for evaluation.")
        return 0.0

    correct = 0
    total = 0
    class_correct = list(0. for i in range(len(classes)))
    class_total = list(0. for i in range(len(classes)))
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            c = (predicted == labels).squeeze()
            # If batch size is 1, c might be a scalar
            if c.dim() == 0:
                c = c.unsqueeze(0)
                
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    overall_accuracy = 100 * correct / total if total > 0 else 0
    print(f'\nFinal Accuracy on Test Set: {overall_accuracy:.2f}%')
    mlflow.log_metric("final_test_accuracy", overall_accuracy)

    for i in range(len(classes)):
        if class_total[i] > 0:
            class_acc = 100 * class_correct[i] / class_total[i]
            print(f'Accuracy of {classes[i]}: {class_acc:.2f}%')
            mlflow.log_metric(f"accuracy_{classes[i]}", class_acc)

    return overall_accuracy
