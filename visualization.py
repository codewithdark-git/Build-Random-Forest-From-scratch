"""
Visualization utilities for Random Forest experiments.

Includes functions for plotting accuracy curves, confusion matrices,
feature importance, and training time comparisons.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os
import config


# Set plotting style
try:
    plt.style.use(config.PLOT_STYLE)
except:
    plt.style.use('default')

sns.set_palette("husl")


def plot_accuracy_vs_n_estimators(
    n_estimators_list: list,
    train_accuracies: list,
    test_accuracies: list,
    title: str = "Accuracy vs Number of Trees",
    filename: str = None
) -> None:
    """
    Plot accuracy as a function of number of trees.
    
    Parameters:
    -----------
    n_estimators_list : list
        List of n_estimators values
    train_accuracies : list
        Training accuracies for each n_estimators
    test_accuracies : list
        Test accuracies for each n_estimators
    title : str
        Plot title
    filename : str, optional
        If provided, save plot to this filename
    """
    plt.figure(figsize=config.FIGURE_SIZE, dpi=config.DPI)
    
    plt.plot(n_estimators_list, train_accuracies, 'o-', 
             label='Training Accuracy', linewidth=2, markersize=8)
    plt.plot(n_estimators_list, test_accuracies, 's-', 
             label='Test Accuracy', linewidth=2, markersize=8)
    
    plt.xlabel('Number of Trees (n_estimators)', fontsize=12, fontweight='bold')
    plt.ylabel('Accuracy', fontsize=12, fontweight='bold')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Add value annotations
    for i, (n_est, acc) in enumerate(zip(n_estimators_list, test_accuracies)):
        plt.annotate(f'{acc:.3f}', 
                    xy=(n_est, acc), 
                    xytext=(5, 5), 
                    textcoords='offset points',
                    fontsize=9,
                    alpha=0.7)
    
    plt.tight_layout()
    
    if filename:
        filepath = os.path.join(config.PLOTS_DIR, filename)
        plt.savefig(filepath, dpi=config.DPI, bbox_inches='tight')
        print(f"Plot saved to {filepath}")
    
    plt.show()


def plot_training_time_comparison(
    n_estimators_list: list,
    training_times: list,
    title: str = "Training Time vs Number of Trees",
    filename: str = None
) -> None:
    """
    Plot training time as a function of number of trees.
    
    Parameters:
    -----------
    n_estimators_list : list
        List of n_estimators values
    training_times : list
        Training times for each n_estimators
    title : str
        Plot title
    filename : str, optional
        If provided, save plot to this filename
    """
    plt.figure(figsize=config.FIGURE_SIZE, dpi=config.DPI)
    
    plt.plot(n_estimators_list, training_times, 'o-', 
             linewidth=2, markersize=8, color='coral')
    
    plt.xlabel('Number of Trees (n_estimators)', fontsize=12, fontweight='bold')
    plt.ylabel('Training Time (seconds)', fontsize=12, fontweight='bold')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Add value annotations
    for i, (n_est, time) in enumerate(zip(n_estimators_list, training_times)):
        plt.annotate(f'{time:.2f}s', 
                    xy=(n_est, time), 
                    xytext=(5, 5), 
                    textcoords='offset points',
                    fontsize=9,
                    alpha=0.7)
    
    plt.tight_layout()
    
    if filename:
        filepath = os.path.join(config.PLOTS_DIR, filename)
        plt.savefig(filepath, dpi=config.DPI, bbox_inches='tight')
        print(f"Plot saved to {filepath}")
    
    plt.show()


def plot_tree_vs_forest_comparison(
    metrics_dict: dict,
    title: str = "Decision Tree vs Random Forest Comparison",
    filename: str = None
) -> None:
    """
    Plot comparison between Decision Tree and Random Forest.
    
    Parameters:
    -----------
    metrics_dict : dict
        Dictionary with 'tree' and 'forest' keys, each containing metrics
    title : str
        Plot title
    filename : str, optional
        If provided, save plot to this filename
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=config.DPI)
    
    # Accuracy comparison
    models = ['Decision Tree', 'Random Forest']
    train_accs = [metrics_dict['tree']['train_accuracy'], 
                  metrics_dict['forest']['train_accuracy']]
    test_accs = [metrics_dict['tree']['test_accuracy'], 
                 metrics_dict['forest']['test_accuracy']]
    
    x = np.arange(len(models))
    width = 0.35
    
    axes[0].bar(x - width/2, train_accs, width, label='Training', alpha=0.8)
    axes[0].bar(x + width/2, test_accs, width, label='Test', alpha=0.8)
    axes[0].set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    axes[0].set_title('Accuracy Comparison', fontsize=12, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, (train, test) in enumerate(zip(train_accs, test_accs)):
        axes[0].text(i - width/2, train + 0.01, f'{train:.3f}', 
                    ha='center', va='bottom', fontsize=9)
        axes[0].text(i + width/2, test + 0.01, f'{test:.3f}', 
                    ha='center', va='bottom', fontsize=9)
    
    # Training time comparison
    times = [metrics_dict['tree']['training_time'], 
             metrics_dict['forest']['training_time']]
    
    axes[1].bar(models, times, alpha=0.8, color=['skyblue', 'lightcoral'])
    axes[1].set_ylabel('Training Time (seconds)', fontsize=12, fontweight='bold')
    axes[1].set_title('Training Time Comparison', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, time in enumerate(times):
        axes[1].text(i, time + max(times)*0.02, f'{time:.3f}s', 
                    ha='center', va='bottom', fontsize=9)
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if filename:
        filepath = os.path.join(config.PLOTS_DIR, filename)
        plt.savefig(filepath, dpi=config.DPI, bbox_inches='tight')
        print(f"Plot saved to {filepath}")
    
    plt.show()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list = None,
    title: str = "Confusion Matrix",
    filename: str = None
) -> None:
    """
    Plot confusion matrix.
    
    Parameters:
    -----------
    y_true : array
        True labels
    y_pred : array
        Predicted labels
    class_names : list, optional
        Names of classes
    title : str
        Plot title
    filename : str, optional
        If provided, save plot to this filename
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6), dpi=config.DPI)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if filename:
        filepath = os.path.join(config.PLOTS_DIR, filename)
        plt.savefig(filepath, dpi=config.DPI, bbox_inches='tight')
        print(f"Plot saved to {filepath}")
    
    plt.show()


def plot_feature_importance(
    importances: np.ndarray,
    feature_names: list = None,
    top_n: int = 10,
    title: str = "Feature Importance",
    filename: str = None
) -> None:
    """
    Plot feature importance.
    
    Parameters:
    -----------
    importances : array
        Feature importance values
    feature_names : list, optional
        Names of features
    top_n : int
        Number of top features to show
    title : str
        Plot title
    filename : str, optional
        If provided, save plot to this filename
    """
    if feature_names is None:
        feature_names = [f'Feature {i}' for i in range(len(importances))]
    
    # Sort by importance
    indices = np.argsort(importances)[::-1][:top_n]
    top_importances = importances[indices]
    top_features = [feature_names[i] for i in indices]
    
    plt.figure(figsize=(10, 6), dpi=config.DPI)
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(top_importances)))
    plt.barh(range(len(top_importances)), top_importances, color=colors, alpha=0.8)
    
    plt.yticks(range(len(top_importances)), top_features)
    plt.xlabel('Importance', fontsize=12, fontweight='bold')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, imp in enumerate(top_importances):
        plt.text(imp + max(top_importances)*0.01, i, f'{imp:.4f}', 
                va='center', fontsize=9)
    
    plt.tight_layout()
    
    if filename:
        filepath = os.path.join(config.PLOTS_DIR, filename)
        plt.savefig(filepath, dpi=config.DPI, bbox_inches='tight')
        print(f"Plot saved to {filepath}")
    
    plt.show()


def plot_all_metrics(
    metrics_dict: dict,
    title: str = "Model Performance Metrics",
    filename: str = None
) -> None:
    """
    Plot all metrics (accuracy, precision, recall, F1) for comparison.
    
    Parameters:
    -----------
    metrics_dict : dict
        Dictionary with model names as keys and metrics as values
    title : str
        Plot title
    filename : str, optional
        If provided, save plot to this filename
    """
    metric_names = ['accuracy', 'precision', 'recall', 'f1']
    models = list(metrics_dict.keys())
    
    fig, ax = plt.subplots(figsize=(12, 6), dpi=config.DPI)
    
    x = np.arange(len(metric_names))
    width = 0.35
    
    for i, model in enumerate(models):
        values = [metrics_dict[model].get(metric, 0) for metric in metric_names]
        offset = (i - len(models)/2 + 0.5) * width
        ax.bar(x + offset, values, width, label=model, alpha=0.8)
    
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([m.capitalize() for m in metric_names])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1.1])
    
    plt.tight_layout()
    
    if filename:
        filepath = os.path.join(config.PLOTS_DIR, filename)
        plt.savefig(filepath, dpi=config.DPI, bbox_inches='tight')
        print(f"Plot saved to {filepath}")
    
    plt.show()


def create_summary_plot(
    experiment_results: dict,
    dataset_name: str,
    filename: str = None
) -> None:
    """
    Create a comprehensive summary plot with multiple subplots.
    
    Parameters:
    -----------
    experiment_results : dict
        Dictionary containing all experiment results
    dataset_name : str
        Name of the dataset
    filename : str, optional
        If provided, save plot to this filename
    """
    fig = plt.figure(figsize=(16, 10), dpi=config.DPI)
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # Plot 1: Accuracy vs n_estimators
    ax1 = fig.add_subplot(gs[0, 0])
    n_est = experiment_results['n_estimators_list']
    
    # Handle new key names (prefer sklearn for summary)
    if 'sklearn_test_accuracies' in experiment_results:
        test_acc = experiment_results['sklearn_test_accuracies']
    else:
        test_acc = experiment_results.get('test_accuracies', [])
        
    ax1.plot(n_est, test_acc, 'o-', linewidth=2, markersize=8)
    ax1.set_xlabel('Number of Trees', fontweight='bold')
    ax1.set_ylabel('Test Accuracy', fontweight='bold')
    ax1.set_title('Accuracy vs Number of Trees', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Training time vs n_estimators
    ax2 = fig.add_subplot(gs[0, 1])
    
    if 'sklearn_training_times' in experiment_results:
        times = experiment_results['sklearn_training_times']
    else:
        times = experiment_results.get('training_times', [])
        
    ax2.plot(n_est, times, 's-', linewidth=2, markersize=8, color='coral')
    ax2.set_xlabel('Number of Trees', fontweight='bold')
    ax2.set_ylabel('Training Time (s)', fontweight='bold')
    ax2.set_title('Training Time vs Number of Trees', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Tree vs Forest comparison
    ax3 = fig.add_subplot(gs[1, 0])
    models = ['Decision Tree', 'Random Forest']
    
    # Handle new key names
    if 'sklearn_tree_metrics' in experiment_results:
        tree_metrics = experiment_results['sklearn_tree_metrics']
        forest_metrics = experiment_results['sklearn_forest_metrics']
    else:
        tree_metrics = experiment_results.get('tree_metrics', {'test_accuracy': 0})
        forest_metrics = experiment_results.get('forest_metrics', {'test_accuracy': 0})
        
    tree_acc = tree_metrics.get('test_accuracy', 0)
    forest_acc = forest_metrics.get('test_accuracy', 0)
    
    ax3.bar(models, [tree_acc, forest_acc], alpha=0.8, color=['skyblue', 'lightcoral'])
    ax3.set_ylabel('Test Accuracy', fontweight='bold')
    ax3.set_title('Model Comparison', fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    for i, acc in enumerate([tree_acc, forest_acc]):
        ax3.text(i, acc + 0.01, f'{acc:.3f}', ha='center', va='bottom')
    
    # Plot 4: Feature importance (if available)
    ax4 = fig.add_subplot(gs[1, 1])
    if 'feature_importances' in experiment_results:
        importances = experiment_results['feature_importances']
        top_n = min(10, len(importances))
        indices = np.argsort(importances)[::-1][:top_n]
        ax4.barh(range(top_n), importances[indices], alpha=0.8)
        ax4.set_yticks(range(top_n))
        ax4.set_yticklabels([f'F{i}' for i in indices])
        ax4.set_xlabel('Importance', fontweight='bold')
        ax4.set_title('Top 10 Feature Importances', fontweight='bold')
        ax4.invert_yaxis()
        ax4.grid(True, alpha=0.3, axis='x')
    
    plt.suptitle(f'Random Forest Experiments - {dataset_name}', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    if filename:
        filepath = os.path.join(config.PLOTS_DIR, filename)
        plt.savefig(filepath, dpi=config.DPI, bbox_inches='tight')
        print(f"Summary plot saved to {filepath}")
    
    plt.show()
