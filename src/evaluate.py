import torch
import matplotlib.pyplot as plt
from .config import Config
from .data import get_dataloaders


def visualize_third_layer():
    # 1. Load data and model
    _, test_loader = get_dataloaders()
   
    print("Loading model for Layer 3 (64 filters) visualization...")
    model = torch.load("model.pth", map_location=Config.DEVICE)
    model.eval()


    # 2. Get a single image
    images, _ = next(iter(test_loader))
    single_img = images[0].unsqueeze(0).to(Config.DEVICE)


    # 3. Extract Feature Maps from the Third Layer
    # In your BasicCNN:
    # features[0:6] includes (Conv1, ReLU, Pool, Conv2, ReLU, Pool)
    # features[6] is the third Conv2d layer with 64 filters
    with torch.no_grad():
        intermediate_output = model.features[0:6](single_img)
        third_layer_maps = model.features[6](intermediate_output)


    # 4. Plot the 64 Feature Maps
    # We use an 8x8 grid to fit all 64 filters
    fig, axes = plt.subplots(8, 8, figsize=(16, 16))
    fig.suptitle("Third Layer: 64 High-Level Feature Maps", fontsize=20)


    for i in range(64):
        ax = axes.flat[i]
        # 'inferno' or 'magma' works well for deep activations
        f_map = third_layer_maps[0, i].cpu().numpy()
        ax.imshow(f_map, cmap='inferno')
        ax.axis('off')
        if i % 8 == 0: # Add some labels to keep track
            ax.set_title(f"M{i}", fontsize=8)


    plt.tight_layout()
    plt.subplots_adjust(top=0.94)
    plt.show()


if __name__ == "__main__":
    visualize_third_layer()
