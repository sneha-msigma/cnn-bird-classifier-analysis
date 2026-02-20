import sys
from src.analysis import run_filter_visualization

if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python main_analysis.py <image_path>")
        sys.exit()

    image_path = sys.argv[1]
    run_filter_visualization(image_path)