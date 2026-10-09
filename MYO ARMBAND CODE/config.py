# config.py
import os

# Paths - Change these based on your environment
# For Kaggle:
# TRAIN_DIR = '/kaggle/input/datasets/farahhassanellaban/ninapro-9-subjects'
# TARGET_DIR = '/kaggle/input/datasets/farahhassanellaban/test-subject'

# For Local/GitHub:
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
TRAIN_DIR = os.path.join(DATA_DIR, 'ninapro-9-subjects')
TARGET_DIR = os.path.join(DATA_DIR, 'test-subject')
WEIGHTS_DIR = os.path.join(BASE_DIR, 'Myo_final_weights')

# Model Architecture
EMBEDDING_DIM = 32
NUM_SENSORS = 8
WINDOW_SIZE = 200
SAMPLING_RATE = 200.0

# Classes
BASE_CLASSES = [0, 1, 2]
NOVEL_CLASSES = [3, 4, 5]
TOTAL_CLASSES = 6

# Training Hyperparameters
LR = 0.001
WEIGHT_DECAY = 1e-3
EPOCHS = 30
PATIENCE = 7
BATCH_SIZE = 64

# Inference & Safety
MC_DROPOUT_PASSES = 5
UNCERTAINTY_THRESHOLD = 0.2
DISTANCE_THRESHOLD = 0.5

# Domain Shift
SHIFT_FACTOR = 0.5
NOISE_STD = 0.2

# Subjects
PRETRAIN_SUBJECTS = [1, 2, 3, 4, 5, 6, 7, 8, 9]
TARGET_SUBJECT = [10]