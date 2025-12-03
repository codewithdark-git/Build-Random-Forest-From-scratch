"""
Utility functions for Random Forest experiments.

Includes data loading, preprocessing, metrics calculation, and result management.
"""

import numpy as np
import pandas as pd
import os
import pickle
from typing import Tuple, Dict, Any
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from PIL import Image
import glob
from tqdm import tqdm
import config


def load_heart_disease_data() -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load and preprocess Heart Disease UCI dataset.
    
    Returns:
    --------
    X : DataFrame
        Feature matrix
    y : Series
        Target labels (binary: 0 = no disease, 1 = disease)
    """
    # Column names
    columns = config.HEART_DISEASE_COLUMNS
    
    # Try to load from local file first
    local_path = os.path.join(config.HEART_DISEASE_PATH, 'heart_disease.csv')
    
    if os.path.exists(local_path):
        df = pd.read_csv(local_path)
    else:
        # Download from UCI repository
        try:
            df = pd.read_csv(config.HEART_DISEASE_URL, names=columns, na_values='?')
            # Save locally
            df.to_csv(local_path, index=False)
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            print("Creating sample dataset for demonstration...")
            # Create sample data if download fails
            df = create_sample_heart_disease_data()
            df.to_csv(local_path, index=False)
    
    # Handle missing values
    df = df.dropna()
    
    # Convert target to binary (0 = no disease, 1-4 = disease)
    df['target'] = (df['target'] > 0).astype(int)
    
    # Separate features and target
    X = df.drop('target', axis=1)
    y = df['target']
    
    return X, y


def create_sample_heart_disease_data() -> pd.DataFrame:
    """
    Create sample heart disease data for demonstration.
    
    Returns:
    --------
    df : DataFrame
        Sample dataset
    """
    np.random.seed(config.RANDOM_STATE)
    n_samples = 300
    
    data = {
        'age': np.random.randint(30, 80, n_samples),
        'sex': np.random.randint(0, 2, n_samples),
        'cp': np.random.randint(0, 4, n_samples),
        'trestbps': np.random.randint(90, 200, n_samples),
        'chol': np.random.randint(120, 400, n_samples),
        'fbs': np.random.randint(0, 2, n_samples),
        'restecg': np.random.randint(0, 3, n_samples),
        'thalach': np.random.randint(70, 200, n_samples),
        'exang': np.random.randint(0, 2, n_samples),
        'oldpeak': np.random.uniform(0, 6, n_samples),
        'slope': np.random.randint(0, 3, n_samples),
        'ca': np.random.randint(0, 4, n_samples),
        'thal': np.random.randint(0, 4, n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Create target based on some features (simplified logic)
    df['target'] = ((df['age'] > 55) & (df['chol'] > 250) | 
                    (df['trestbps'] > 140) | 
                    (df['oldpeak'] > 2)).astype(int)
    
    return df


def preprocess_tabular_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = config.TEST_SIZE,
    random_state: int = config.RANDOM_STATE
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Preprocess tabular data: split and scale.
    
    Parameters:
    -----------
    X : DataFrame
        Feature matrix
    y : Series
        Target labels
    test_size : float
        Proportion of test set
    random_state : int
        Random seed
    
    Returns:
    --------
    X_train, X_test, y_train, y_test : arrays
        Split and scaled data
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train.values, y_test.values


def load_intel_image_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load Intel Image Classification dataset from disk.
    
    Expects structure:
    data/intel_images/seg_train/<class>/<image>
    data/intel_images/seg_test/<class>/<image>
    
    If not found, creates synthetic data.
    
    Returns:
    --------
    X_train, X_test, y_train, y_test
    """
    intel_path = config.INTEL_IMAGE_PATH
    train_dir = os.path.join(intel_path, 'seg_train')
    test_dir = os.path.join(intel_path, 'seg_test')
    
    # Check if directories exist
    if os.path.exists(train_dir) and os.path.exists(test_dir):
        print(f"Loading images from {intel_path}...")
        
        def load_from_dir(directory):
            classes = sorted(os.listdir(directory))
            X = []
            y = []
            
            print(f"Processing {directory}...")
            for idx, class_name in enumerate(classes):
                class_path = os.path.join(directory, class_name)
                if not os.path.isdir(class_path):
                    continue
                
                image_files = glob.glob(os.path.join(class_path, '*.jpg'))
                print(f"  Class '{class_name}': {len(image_files)} images")
                
                # Limit images per class if configured
                if config.MAX_IMAGES_PER_CLASS:
                    image_files = image_files[:config.MAX_IMAGES_PER_CLASS]
                
                for img_path in tqdm(image_files, desc=f"Loading {class_name}", leave=False):
                    try:
                        with Image.open(img_path) as img:
                            # Resize
                            img = img.resize(config.IMAGE_SIZE)
                            # Convert to array and normalize
                            img_array = np.array(img).flatten() / 255.0
                            X.append(img_array)
                            y.append(idx)
                    except Exception as e:
                        print(f"Error loading {img_path}: {e}")
            
            return np.array(X, dtype=np.float32), np.array(y)

        X_train, y_train = load_from_dir(train_dir)
        X_test, y_test = load_from_dir(test_dir)
        
        print(f"Loaded {len(X_train)} training images and {len(X_test)} test images.")
        return X_train, X_test, y_train, y_test
        
    else:
        print("Intel Image dataset not found at specified path.")
        print("Creating synthetic image data for demonstration...")
        
        np.random.seed(config.RANDOM_STATE)
        
        # Simulate 6 classes
        n_classes = 6
        samples_per_class = 200
        n_samples = n_classes * samples_per_class
        
        # Image dimensions (flattened)
        img_height, img_width = config.IMAGE_SIZE
        n_features = img_height * img_width * config.IMAGE_CHANNELS
        
        X = np.random.rand(n_samples, n_features).astype(np.float32)
        y = np.repeat(np.arange(n_classes), samples_per_class)
        
        # Add structure
        for i in range(n_classes):
            class_mask = y == i
            X[class_mask] += i * 0.1
            X[class_mask] += np.random.randn(samples_per_class, n_features) * 0.05
        
        # Normalize
        X = (X - X.mean()) / X.std()
        
        # Split synthetic data
        return train_test_split(X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=y)


def preprocess_image_data(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Preprocess image data.
    
    Since data is already loaded and normalized (if real), 
    this mainly serves as a pass-through or for additional scaling if needed.
    
    Returns:
    --------
    X_train, X_test, y_train, y_test
    """
    # If using synthetic data, it might not be scaled to [0, 1]
    # But load_intel_image_data handles normalization for real images
    
    return X_train, X_test, y_train, y_test


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate classification metrics.
    
    Parameters:
    -----------
    y_true : array
        True labels
    y_pred : array
        Predicted labels
    
    Returns:
    --------
    metrics : dict
        Dictionary of metric names and values
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }
    
    return metrics


def save_results(results: Dict[str, Any], filename: str) -> None:
    """
    Save experiment results to file.
    
    Parameters:
    -----------
    results : dict
        Results dictionary
    filename : str
        Output filename
    """
    filepath = os.path.join(config.RESULTS_DIR, filename)
    
    with open(filepath, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"Results saved to {filepath}")


def save_results_json(results: Dict[str, Any], filename: str) -> None:
    """
    Save experiment results to JSON file for better readability.
    
    Parameters:
    -----------
    results : dict
        Results dictionary
    filename : str
        Output filename (should end with .json)
    """
    import json
    
    filepath = os.path.join(config.RESULTS_DIR, filename)
    
    # Convert numpy arrays and other non-serializable types to lists
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj
    
    serializable_results = convert_to_serializable(results)
    
    with open(filepath, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"JSON results saved to {filepath}")


def load_results(filename: str) -> Dict[str, Any]:
    """
    Load experiment results from file.
    
    Parameters:
    -----------
    filename : str
        Input filename
    
    Returns:
    --------
    results : dict
        Results dictionary
    """
    filepath = os.path.join(config.RESULTS_DIR, filename)
    
    with open(filepath, 'rb') as f:
        results = pickle.load(f)
    
    return results


def print_metrics(metrics: Dict[str, float], title: str = "Metrics") -> None:
    """
    Print metrics in a formatted way.
    
    Parameters:
    -----------
    metrics : dict
        Metrics dictionary
    title : str
        Title for the metrics
    """
    print(f"\n{title}")
    print("=" * 50)
    for metric_name, value in metrics.items():
        print(f"{metric_name.capitalize():15s}: {value:.4f}")
    print("=" * 50)


def create_results_dataframe(results: Dict[str, Any]) -> pd.DataFrame:
    """
    Create a DataFrame from experiment results.
    
    Parameters:
    -----------
    results : dict
        Results dictionary
    
    Returns:
    --------
    df : DataFrame
        Results as DataFrame
    """
    df = pd.DataFrame(results)
    return df


def get_feature_names_heart_disease() -> list:
    """
    Get feature names for Heart Disease dataset.
    
    Returns:
    --------
    feature_names : list
        List of feature names
    """
    return [col for col in config.HEART_DISEASE_COLUMNS if col != 'target']
