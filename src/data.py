import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from .config import IMAGE_SIZE, BATCH_SIZE, DATA_DIR

def prepare_data_loaders():
    """
    Loads images from the local system and splits them into training and testing sets.
    """
    
    # Standard image transforms
    data_transforms = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    try:
        # Loading from local system using ImageFolder
        # Assumes directory structure: root/class_x/xxx.ext
        full_dataset = datasets.ImageFolder(root=DATA_DIR, transform=data_transforms)
    except Exception as e:
        print(f"Error: Could not find dataset at {DATA_DIR}. {e}")
        return None, None, []

    # Split into train data and test data (80% train, 20% test)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    return train_loader, test_loader, full_dataset.classes
