"""
Configuration file for Random Forest experiments.

Contains dataset paths, experiment parameters, and reproducibility settings.
"""

import os

# Random seed for reproducibility
RANDOM_STATE = 42

# Dataset paths
DATA_DIR = 'data'
HEART_DISEASE_PATH = os.path.join(DATA_DIR, 'heart_disease')
INTEL_IMAGE_PATH = os.path.join(DATA_DIR, 'intel_images')

# Output directories
OUTPUT_DIR = 'outputs'
PLOTS_DIR = os.path.join(OUTPUT_DIR, 'plots')
RESULTS_DIR = os.path.join(OUTPUT_DIR, 'results')

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(HEART_DISEASE_PATH, exist_ok=True)
os.makedirs(INTEL_IMAGE_PATH, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Experiment parameters
N_ESTIMATORS_RANGE = [1, 10, 50, 100, 300]
TEST_SIZE = 0.2
CV_FOLDS = 5

# Decision Tree parameters
TREE_MAX_DEPTH = None
TREE_MIN_SAMPLES_SPLIT = 2
TREE_MIN_SAMPLES_LEAF = 1

# Random Forest parameters
RF_MAX_DEPTH = None
RF_MIN_SAMPLES_SPLIT = 2
RF_MIN_SAMPLES_LEAF = 1
RF_MAX_FEATURES = 'sqrt'
RF_BOOTSTRAP = True
RF_OOB_SCORE = True

# Image processing parameters
IMAGE_SIZE = (64, 64)  # Resize images to this size
IMAGE_CHANNELS = 3  # RGB
MAX_IMAGES_PER_CLASS = 1000  # Limit for faster experiments

# Dataset URLs
HEART_DISEASE_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data'

# Column names for Heart Disease dataset
HEART_DISEASE_COLUMNS = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
]

# Plotting parameters
FIGURE_SIZE = (10, 6)
DPI = 100
PLOT_STYLE = 'seaborn-v0_8-darkgrid'
