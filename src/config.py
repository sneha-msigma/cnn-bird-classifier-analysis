import os

# DATASET CONFIGURATION
IMAGE_SIZE = 128
BATCH_SIZE = 32
DATA_DIR = r"C:\Users\User\Downloads\DataSet-20260211T064335Z-3-001"
OUTPUT_DIR = os.path.join(os.getcwd(), "outputs")

# TRAINING HYPERPARAMETERS
LEARNING_RATE = 0.001
EPOCHS = 10
NUM_CLASSES = 2  # Binary classification: Bird or Not Bird

# MLFLOW CONFIGURATION
EXPERIMENT_NAME = "Bird_Binary_Classification"
RUN_NAME = "Initial_CNN_Run"
