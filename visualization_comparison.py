"""
Additional visualization functions for custom vs sklearn comparison.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import config


def plot_custom_vs_sklearn_detailed(
    custom_metrics: dict,
    sklearn_metrics: dict,
    dataset_name: str,
    filename: str = None
) -> None:
    """
    Create detailed comparison plot between custom and scikit-learn implementations.
    
    Parameters:
    -----------
    custom_metrics : dict
        Metrics from custom implementation
    sklearn_metrics : dict
        Metrics from scikit-learn implementation
    dataset_name : str
        Name of dataset
    filename : str, optional
        If provided, save plot to this filename
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=config.DPI)
    
    # Plot 1: Accuracy Comparison
    ax1 = axes[0, 0]
    models = ['Custom\nRandom Forest', 'Scikit-learn\nRandom Forest']
    accuracies = [custom_metrics['test_accuracy'], sklearn_metrics['test_accuracy']]
    colors = ['#FF6B6B', '#4ECDC4']
    
    bars = ax1.bar(models, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Test Accuracy', fontsize=12, fontweight='bold')
    ax1.set_title('Accuracy Comparison', fontsize=13, fontweight='bold')
    ax1.set_ylim([0, 1.0])
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{acc:.4f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Add difference annotation
    diff = abs(accuracies[0] - accuracies[1])
    ax1.text(0.5, 0.95, f'Difference: {diff:.4f}', 
            transform=ax1.transAxes, ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=10)
    
    # Plot 2: All Metrics Comparison
    ax2 = axes[0, 1]
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1']
    custom_values = [
        custom_metrics.get('accuracy', custom_metrics.get('test_accuracy', 0)),
        custom_metrics.get('precision', 0),
        custom_metrics.get('recall', 0),
        custom_metrics.get('f1', 0)
    ]
    sklearn_values = [
        sklearn_metrics.get('accuracy', sklearn_metrics.get('test_accuracy', 0)),
        sklearn_metrics.get('precision', 0),
        sklearn_metrics.get('recall', 0),
        sklearn_metrics.get('f1', 0)
    ]
    
    x = np.arange(len(metric_names))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, custom_values, width, label='Custom', 
                    color='#FF6B6B', alpha=0.8, edgecolor='black')
    bars2 = ax2.bar(x + width/2, sklearn_values, width, label='Scikit-learn', 
                    color='#4ECDC4', alpha=0.8, edgecolor='black')
    
    ax2.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax2.set_title('Performance Metrics Comparison', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(metric_names, fontsize=10)
    ax2.legend(fontsize=10)
    ax2.set_ylim([0, 1.1])
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                        f'{height:.3f}',
                        ha='center', va='bottom', fontsize=8)
    
    # Plot 3: Training Time Comparison
    ax3 = axes[1, 0]
    times = [custom_metrics['training_time'], sklearn_metrics['training_time']]
    bars = ax3.bar(models, times, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax3.set_ylabel('Training Time (seconds)', fontsize=12, fontweight='bold')
    ax3.set_title('Training Time Comparison', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, time in zip(bars, times):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + max(times)*0.02,
                f'{time:.3f}s',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Add speedup annotation
    speedup = custom_metrics['training_time'] / sklearn_metrics['training_time']
    ax3.text(0.5, 0.95, f'Sklearn is {speedup:.1f}x faster', 
            transform=ax3.transAxes, ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5),
            fontsize=10)
    
    # Plot 4: Summary Statistics
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    summary_text = f"""
    COMPARISON SUMMARY - {dataset_name}
    {'='*45}
    
    ACCURACY:
    • Custom RF:      {custom_metrics['test_accuracy']:.4f}
    • Scikit-learn:   {sklearn_metrics['test_accuracy']:.4f}
    • Difference:     {abs(custom_metrics['test_accuracy'] - sklearn_metrics['test_accuracy']):.4f}
    
    TRAINING TIME:
    • Custom RF:      {custom_metrics['training_time']:.3f}s
    • Scikit-learn:   {sklearn_metrics['training_time']:.3f}s
    • Speedup:        {speedup:.1f}x
    
    VALIDATION:
    • Accuracy diff < 5%:  {'✓ PASS' if abs(custom_metrics['test_accuracy'] - sklearn_metrics['test_accuracy']) < 0.05 else '✗ FAIL'}
    • Implementation:      {'✓ Validated' if abs(custom_metrics['test_accuracy'] - sklearn_metrics['test_accuracy']) < 0.08 else '⚠ Review Needed'}
    
    CONCLUSION:
    Custom implementation {'successfully validates' if abs(custom_metrics['test_accuracy'] - sklearn_metrics['test_accuracy']) < 0.08 else 'needs review'}
    against scikit-learn with comparable accuracy.
    Trade-off: Slower training for educational value.
    """
    
    ax4.text(0.1, 0.95, summary_text, transform=ax4.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.suptitle(f'Custom vs Scikit-learn Implementation - {dataset_name}', 
                 fontsize=15, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    if filename:
        filepath = os.path.join(config.PLOTS_DIR, filename)
        plt.savefig(filepath, dpi=config.DPI, bbox_inches='tight')
        print(f"Detailed comparison plot saved to {filepath}")
    
    plt.show()


def plot_implementation_comparison_grid(
    exp1_results: dict,
    exp2_results: dict,
    exp3_results: dict,
    dataset_name: str,
    filename: str = None
) -> None:
    """
    Create comprehensive grid comparing all implementations.
    
    Parameters:
    -----------
    exp1_results : dict
        Results from experiment 1 (n_estimators)
    exp2_results : dict
        Results from experiment 2 (tree vs forest)
    exp3_results : dict
        Results from experiment 3 (custom vs sklearn)
    dataset_name : str
        Name of dataset
    filename : str, optional
        If provided, save plot to this filename
    """
    fig = plt.figure(figsize=(18, 12), dpi=config.DPI)
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)
    
    # Plot 1: Decision Tree Comparison
    ax1 = fig.add_subplot(gs[0, 0])
    models = ['Sklearn\nTree', 'Custom\nTree']
    # Handle both old and new key formats
    sklearn_tree_acc = exp2_results.get('sklearn_tree_metrics', exp2_results.get('tree_metrics'))['test_accuracy']
    custom_tree_acc = exp3_results['custom_tree_metrics']['test_accuracy']
    accs = [sklearn_tree_acc, custom_tree_acc]
    ax1.bar(models, accs, color=['#3498db', '#e74c3c'], alpha=0.8, edgecolor='black')
    ax1.set_ylabel('Test Accuracy', fontweight='bold')
    ax1.set_title('Decision Tree Comparison', fontweight='bold', fontsize=11)
    ax1.set_ylim([0, 1.0])
    ax1.grid(True, alpha=0.3, axis='y')
    for i, (m, a) in enumerate(zip(models, accs)):
        ax1.text(i, a + 0.02, f'{a:.3f}', ha='center', fontsize=9, fontweight='bold')
    
    # Plot 2: Random Forest Comparison (50 trees)
    ax2 = fig.add_subplot(gs[0, 1])
    models = ['Sklearn\nRF (50)', 'Custom\nRF (50)']
    accs = [
        exp3_results['sklearn_rf_metrics']['test_accuracy'],
        exp3_results['custom_rf_metrics']['test_accuracy']
    ]
    ax2.bar(models, accs, color=['#2ecc71', '#f39c12'], alpha=0.8, edgecolor='black')
    ax2.set_ylabel('Test Accuracy', fontweight='bold')
    ax2.set_title('Random Forest Comparison (50 trees)', fontweight='bold', fontsize=11)
    ax2.set_ylim([0, 1.0])
    ax2.grid(True, alpha=0.3, axis='y')
    for i, (m, a) in enumerate(zip(models, accs)):
        ax2.text(i, a + 0.02, f'{a:.3f}', ha='center', fontsize=9, fontweight='bold')
    
    # Plot 3: Training Time Comparison
    ax3 = fig.add_subplot(gs[0, 2])
    implementations = ['Sklearn\nTree', 'Custom\nTree', 'Sklearn\nRF', 'Custom\nRF']
    # Handle both old and new key formats
    sklearn_tree_time = exp2_results.get('sklearn_tree_metrics', exp2_results.get('tree_metrics'))['training_time']
    custom_tree_time = exp3_results['custom_tree_metrics']['training_time']
    sklearn_rf_time = exp3_results['sklearn_rf_metrics']['training_time']
    custom_rf_time = exp3_results['custom_rf_metrics']['training_time']
    
    times = [sklearn_tree_time, custom_tree_time, sklearn_rf_time, custom_rf_time]
    colors_time = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    ax3.bar(implementations, times, color=colors_time, alpha=0.8, edgecolor='black')
    ax3.set_ylabel('Time (seconds)', fontweight='bold')
    ax3.set_title('Training Time Comparison', fontweight='bold', fontsize=11)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.tick_params(axis='x', labelsize=8)
    for i, t in enumerate(times):
        ax3.text(i, t + max(times)*0.02, f'{t:.2f}s', ha='center', fontsize=8)
    
    # Plot 4: Accuracy vs n_estimators (Sklearn)
    ax4 = fig.add_subplot(gs[1, 0:2])
    n_est = exp1_results['n_estimators_list']
    # Handle both old and new key formats
    test_acc = exp1_results.get('sklearn_test_accuracies', exp1_results.get('test_accuracies'))
    ax4.plot(n_est, test_acc, 'o-', linewidth=2.5, markersize=10, 
            color='#2ecc71', label='Scikit-learn RF')
    ax4.set_xlabel('Number of Trees', fontweight='bold')
    ax4.set_ylabel('Test Accuracy', fontweight='bold')
    ax4.set_title('Accuracy vs Number of Trees (Scikit-learn)', fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    for x, y in zip(n_est, test_acc):
        ax4.annotate(f'{y:.3f}', (x, y), textcoords='offset points', 
                    xytext=(0, 8), ha='center', fontsize=8)
    
    # Plot 5: All Metrics Comparison
    ax5 = fig.add_subplot(gs[1, 2])
    metric_names = ['Acc', 'Prec', 'Rec', 'F1']
    custom_vals = [
        exp3_results['custom_rf_metrics'].get('accuracy', exp3_results['custom_rf_metrics']['test_accuracy']),
        exp3_results['custom_rf_metrics']['precision'],
        exp3_results['custom_rf_metrics']['recall'],
        exp3_results['custom_rf_metrics']['f1']
    ]
    sklearn_vals = [
        exp3_results['sklearn_rf_metrics']['accuracy'],
        exp3_results['sklearn_rf_metrics']['precision'],
        exp3_results['sklearn_rf_metrics']['recall'],
        exp3_results['sklearn_rf_metrics']['f1']
    ]
    
    x = np.arange(len(metric_names))
    width = 0.35
    ax5.bar(x - width/2, sklearn_vals, width, label='Sklearn', color='#2ecc71', alpha=0.8)
    ax5.bar(x + width/2, custom_vals, width, label='Custom', color='#f39c12', alpha=0.8)
    ax5.set_ylabel('Score', fontweight='bold')
    ax5.set_title('Metrics Comparison', fontweight='bold', fontsize=11)
    ax5.set_xticks(x)
    ax5.set_xticklabels(metric_names, fontsize=9)
    ax5.legend(fontsize=9)
    ax5.set_ylim([0, 1.1])
    ax5.grid(True, alpha=0.3, axis='y')
    
    # Plot 6: Model Performance Summary
    ax6 = fig.add_subplot(gs[2, :])
    ax6.axis('off')
    
    summary = f"""
    COMPREHENSIVE COMPARISON SUMMARY - {dataset_name}
    {'='*120}
    
    DECISION TREE:                          RANDOM FOREST (50 trees):                   RANDOM FOREST (100 trees):
    • Sklearn:  {sklearn_tree_acc:.4f} ({sklearn_tree_time:.3f}s)            • Sklearn:  {exp3_results['sklearn_rf_metrics']['test_accuracy']:.4f} ({exp3_results['sklearn_rf_metrics']['training_time']:.3f}s)                • Sklearn:  {exp2_results.get('sklearn_forest_metrics', exp2_results.get('forest_metrics'))['test_accuracy']:.4f} ({exp2_results.get('sklearn_forest_metrics', exp2_results.get('forest_metrics'))['training_time']:.3f}s)
    • Custom:   {custom_tree_acc:.4f} ({custom_tree_time:.3f}s)            • Custom:   {exp3_results['custom_rf_metrics']['test_accuracy']:.4f} ({exp3_results['custom_rf_metrics']['training_time']:.2f}s)
    
    
    """
    
    ax6.text(0.05, 0.95, summary, transform=ax6.transAxes,
            fontsize=9, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.9, pad=1))
    
    plt.suptitle(f'Complete Implementation Comparison - {dataset_name}', 
                 fontsize=16, fontweight='bold', y=0.998)
    
    if filename:
        filepath = os.path.join(config.PLOTS_DIR, filename)
        plt.savefig(filepath, dpi=config.DPI, bbox_inches='tight')
        print(f"Comprehensive comparison grid saved to {filepath}")
    
    plt.show()
