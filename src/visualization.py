# src/visualization.py

import os
import math
import matplotlib.pyplot as plt


def save_layer_as_grid(activation_tensor, layer_name, base_output_dir):

    # activation shape: (1, C, H, W)
    feature_map = activation_tensor.squeeze(0)
    num_filters = feature_map.shape[0]

    # Calculate grid size (square-like layout)
    grid_size = math.ceil(math.sqrt(num_filters))

    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))

    axes = axes.flatten()

    for i in range(len(axes)):
        axes[i].axis("off")

    for i in range(num_filters):
        axes[i].imshow(feature_map[i], cmap="viridis")
        axes[i].axis("off")

    plt.suptitle(f"{layer_name} ({num_filters} filters)")

    os.makedirs(base_output_dir, exist_ok=True)

    save_path = os.path.join(base_output_dir, f"{layer_name}_grid.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

    print(f"{layer_name}: Saved grid with {num_filters} filters")