"""
Experiments on Intel Image Classification Dataset

This script performs the experiments specified in Question 2, Part B:
1. Vary n_estimators and plot test accuracy
2. Compare Decision Tree vs Random Forest
3. Compare custom (scratch) vs scikit-learn implementations

Dataset: Intel Image Classification
Task: Multi-class classification (6 classes)
"""

import numpy as np
import time
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Import custom implementations
from decision_tree_scratch import DecisionTree
from random_forest_scratch import RandomForest

import config
import utils
import visualization
import visualization_comparison


def experiment_1_vary_n_estimators(X_train, X_test, y_train, y_test):
    """
    Experiment 1: Vary number of trees and measure performance.
    
    Tests n_estimators = [1, 10, 50, 100, 300]
    Runs BOTH custom and scikit-learn implementations.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 1: Varying Number of Trees (n_estimators)")
    print("="*70)
    
    n_estimators_list = config.N_ESTIMATORS_RANGE
    
    # Scikit-learn results
    sklearn_train_accuracies = []
    sklearn_test_accuracies = []
    sklearn_training_times = []
    
    # Custom implementation results
    custom_train_accuracies = []
    custom_test_accuracies = []
    custom_training_times = []
    
    for n_est in n_estimators_list:
        print(f"\n--- Testing with n_estimators={n_est} ---")
        
        # Train Scikit-learn Random Forest
        print(f"  Training Scikit-learn Random Forest...")
        start_time = time.time()
        sklearn_rf = RandomForestClassifier(
            n_estimators=n_est,
            max_features='sqrt',
            random_state=config.RANDOM_STATE,
            n_jobs=-1
        )
        sklearn_rf.fit(X_train, y_train)
        sklearn_time = time.time() - start_time
        
        sklearn_train_pred = sklearn_rf.predict(X_train)
        sklearn_test_pred = sklearn_rf.predict(X_test)
        
        sklearn_train_acc = accuracy_score(y_train, sklearn_train_pred)
        sklearn_test_acc = accuracy_score(y_test, sklearn_test_pred)
        
        sklearn_train_accuracies.append(sklearn_train_acc)
        sklearn_test_accuracies.append(sklearn_test_acc)
        sklearn_training_times.append(sklearn_time)
        
        print(f"    Sklearn - Train: {sklearn_train_acc:.4f}, Test: {sklearn_test_acc:.4f}, Time: {sklearn_time:.3f}s")
        
        # Train Custom Random Forest
        # Use parallel processing (n_jobs=-1) for faster training
        # Limit max_depth to speed up training on high-dimensional data
        if n_est <= 100:
            print(f"  Training Custom Random Forest...")
            start_time = time.time()
            custom_rf = RandomForest(
                n_estimators=n_est,
                max_depth=15,  # Limit depth for speed
                max_features='sqrt',
                random_state=config.RANDOM_STATE,
                n_jobs=-1  # Enable parallel processing
            )
            custom_rf.fit(X_train, y_train)
            custom_time = time.time() - start_time
            
            custom_train_pred = custom_rf.predict(X_train)
            custom_test_pred = custom_rf.predict(X_test)
            
            custom_train_acc = accuracy_score(y_train, custom_train_pred)
            custom_test_acc = accuracy_score(y_test, custom_test_pred)
            
            custom_train_accuracies.append(custom_train_acc)
            custom_test_accuracies.append(custom_test_acc)
            custom_training_times.append(custom_time)
            
            print(f"    Custom  - Train: {custom_train_acc:.4f}, Test: {custom_test_acc:.4f}, Time: {custom_time:.3f}s")
        else:
            print(f"    Custom  - Skipped (too slow for {n_est} trees)")
            custom_train_accuracies.append(None)
            custom_test_accuracies.append(None)
            custom_training_times.append(None)
    
    # Plot results - Sklearn
    print("\nGenerating plots for Scikit-learn...")
    visualization.plot_accuracy_vs_n_estimators(
        n_estimators_list,
        sklearn_train_accuracies,
        sklearn_test_accuracies,
        title="Image Classification: Accuracy vs Number of Trees (Scikit-learn)",
        filename="image_sklearn_accuracy_vs_n_estimators.png"
    )
    
    visualization.plot_training_time_comparison(
        n_estimators_list,
        sklearn_training_times,
        title="Image Classification: Training Time vs Number of Trees (Scikit-learn)",
        filename="image_sklearn_training_time.png"
    )
    
    # Plot results - Custom
    print("\nGenerating plots for Custom implementation...")
    custom_n_est = [n for n, acc in zip(n_estimators_list, custom_test_accuracies) if acc is not None]
    custom_train_acc_filtered = [acc for acc in custom_train_accuracies if acc is not None]
    custom_test_acc_filtered = [acc for acc in custom_test_accuracies if acc is not None]
    custom_time_filtered = [t for t in custom_training_times if t is not None]
    
    if custom_n_est:
        visualization.plot_accuracy_vs_n_estimators(
            custom_n_est,
            custom_train_acc_filtered,
            custom_test_acc_filtered,
            title="Image Classification: Accuracy vs Number of Trees (Custom)",
            filename="image_custom_accuracy_vs_n_estimators.png"
        )
        
        visualization.plot_training_time_comparison(
            custom_n_est,
            custom_time_filtered,
            title="Image Classification: Training Time vs Number of Trees (Custom)",
            filename="image_custom_training_time.png"
        )
    
    # Comparison plot
    print("\nGenerating comparison plot...")
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy comparison
    ax1.plot(n_estimators_list, sklearn_test_accuracies, 'o-', 
             label='Scikit-learn', linewidth=2, markersize=8, color='#2ecc71')
    ax1.plot(custom_n_est, custom_test_acc_filtered, 's-', 
             label='Custom', linewidth=2, markersize=8, color='#e74c3c')
    ax1.set_xlabel('Number of Trees', fontweight='bold')
    ax1.set_ylabel('Test Accuracy', fontweight='bold')
    ax1.set_title('Test Accuracy Comparison', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Training time comparison
    ax2.plot(n_estimators_list, sklearn_training_times, 'o-', 
             label='Scikit-learn', linewidth=2, markersize=8, color='#2ecc71')
    ax2.plot(custom_n_est, custom_time_filtered, 's-', 
             label='Custom', linewidth=2, markersize=8, color='#e74c3c')
    ax2.set_xlabel('Number of Trees', fontweight='bold')
    ax2.set_ylabel('Training Time (s)', fontweight='bold')
    ax2.set_title('Training Time Comparison', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('Custom vs Scikit-learn: n_estimators Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    filepath = os.path.join(config.PLOTS_DIR, 'image_exp1_comparison.png')
    plt.savefig(filepath, dpi=config.DPI, bbox_inches='tight')
    print(f"Comparison plot saved to {filepath}")
    plt.show()
    
    return {
        'n_estimators_list': n_estimators_list,
        'sklearn_train_accuracies': sklearn_train_accuracies,
        'sklearn_test_accuracies': sklearn_test_accuracies,
        'sklearn_training_times': sklearn_training_times,
        'custom_train_accuracies': custom_train_accuracies,
        'custom_test_accuracies': custom_test_accuracies,
        'custom_training_times': custom_training_times,
        'custom_n_estimators_tested': custom_n_est
    }


def experiment_2_tree_vs_forest(X_train, X_test, y_train, y_test):
    """
    Experiment 2: Compare Decision Tree vs Random Forest.
    
    Runs BOTH custom and scikit-learn implementations.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 2: Decision Tree vs Random Forest Comparison")
    print("="*70)
    
    # ===== SCIKIT-LEARN MODELS =====
    print("\n[SCIKIT-LEARN IMPLEMENTATIONS]")
    
    # Train Scikit-learn Decision Tree
    print("\nTraining Scikit-learn Decision Tree...")
    start_time = time.time()
    sklearn_tree = DecisionTreeClassifier(random_state=config.RANDOM_STATE)
    sklearn_tree.fit(X_train, y_train)
    sklearn_tree_time = time.time() - start_time
    
    sklearn_tree_train_pred = sklearn_tree.predict(X_train)
    sklearn_tree_test_pred = sklearn_tree.predict(X_test)
    
    sklearn_tree_metrics = utils.calculate_metrics(y_test, sklearn_tree_test_pred)
    sklearn_tree_metrics['train_accuracy'] = accuracy_score(y_train, sklearn_tree_train_pred)
    sklearn_tree_metrics['test_accuracy'] = accuracy_score(y_test, sklearn_tree_test_pred)
    sklearn_tree_metrics['training_time'] = sklearn_tree_time
    
    utils.print_metrics(sklearn_tree_metrics, "Scikit-learn Decision Tree Performance")
    
    # Train Scikit-learn Random Forest
    print("\nTraining Scikit-learn Random Forest...")
    start_time = time.time()
    sklearn_rf = RandomForestClassifier(
        n_estimators=100,
        random_state=config.RANDOM_STATE,
        n_jobs=-1
    )
    sklearn_rf.fit(X_train, y_train)
    sklearn_rf_time = time.time() - start_time
    
    sklearn_rf_train_pred = sklearn_rf.predict(X_train)
    sklearn_rf_test_pred = sklearn_rf.predict(X_test)
    
    sklearn_rf_metrics = utils.calculate_metrics(y_test, sklearn_rf_test_pred)
    sklearn_rf_metrics['train_accuracy'] = accuracy_score(y_train, sklearn_rf_train_pred)
    sklearn_rf_metrics['test_accuracy'] = accuracy_score(y_test, sklearn_rf_test_pred)
    sklearn_rf_metrics['training_time'] = sklearn_rf_time
    
    utils.print_metrics(sklearn_rf_metrics, "Scikit-learn Random Forest Performance")
    
    # ===== CUSTOM MODELS =====
    print("\n[CUSTOM IMPLEMENTATIONS]")
    
    # Train Custom Decision Tree
    print("\nTraining Custom Decision Tree...")
    start_time = time.time()
    custom_tree = DecisionTree(
        max_depth=15,  # Limit depth for image data
        min_samples_split=5,
        criterion='gini',
        random_state=config.RANDOM_STATE
    )
    custom_tree.fit(X_train, y_train)
    custom_tree_time = time.time() - start_time
    
    custom_tree_train_pred = custom_tree.predict(X_train)
    custom_tree_test_pred = custom_tree.predict(X_test)
    
    custom_tree_metrics = utils.calculate_metrics(y_test, custom_tree_test_pred)
    custom_tree_metrics['train_accuracy'] = accuracy_score(y_train, custom_tree_train_pred)
    custom_tree_metrics['test_accuracy'] = accuracy_score(y_test, custom_tree_test_pred)
    custom_tree_metrics['training_time'] = custom_tree_time
    
    utils.print_metrics(custom_tree_metrics, "Custom Decision Tree Performance")
    
    # Train Custom Random Forest
    print("\nTraining Custom Random Forest...")
    # Use fewer trees and parallel processing for speed
    start_time = time.time()
    custom_rf = RandomForest(
        n_estimators=50,  # Use 50 trees for reasonable speed
        max_depth=15,     # Limit depth
        max_features='sqrt',
        random_state=config.RANDOM_STATE,
        n_jobs=-1         # Enable parallel processing
    )
    custom_rf.fit(X_train, y_train)
    custom_rf_time = time.time() - start_time
    
    custom_rf_train_pred = custom_rf.predict(X_train)
    custom_rf_test_pred = custom_rf.predict(X_test)
    
    custom_rf_metrics = utils.calculate_metrics(y_test, custom_rf_test_pred)
    custom_rf_metrics['train_accuracy'] = accuracy_score(y_train, custom_rf_train_pred)
    custom_rf_metrics['test_accuracy'] = accuracy_score(y_test, custom_rf_test_pred)
    custom_rf_metrics['training_time'] = custom_rf_time
    
    utils.print_metrics(custom_rf_metrics, "Custom Random Forest Performance")
    
    # ===== COMPARISONS =====
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    
    print("\n[Scikit-learn Models]")
    print(f"  Decision Tree: Test Acc = {sklearn_tree_metrics['test_accuracy']:.4f}, Time = {sklearn_tree_time:.3f}s")
    print(f"  Random Forest: Test Acc = {sklearn_rf_metrics['test_accuracy']:.4f}, Time = {sklearn_rf_time:.3f}s")
    print(f"  Improvement:   {sklearn_rf_metrics['test_accuracy'] - sklearn_tree_metrics['test_accuracy']:.4f}")
    
    print("\n[Custom Models]")
    print(f"  Decision Tree: Test Acc = {custom_tree_metrics['test_accuracy']:.4f}, Time = {custom_tree_time:.3f}s")
    print(f"  Random Forest: Test Acc = {custom_rf_metrics['test_accuracy']:.4f}, Time = {custom_rf_time:.3f}s")
    print(f"  Improvement:   {custom_rf_metrics['test_accuracy'] - custom_tree_metrics['test_accuracy']:.4f}")
    
    print("\n[Implementation Comparison]")
    print(f"  Tree Accuracy Diff:  {abs(sklearn_tree_metrics['test_accuracy'] - custom_tree_metrics['test_accuracy']):.4f}")
    print(f"  Forest Accuracy Diff: {abs(sklearn_rf_metrics['test_accuracy'] - custom_rf_metrics['test_accuracy']):.4f}")
    
    # Plot comparison
    print("\nGenerating comparison plots...")
    
    # Sklearn Tree vs Forest
    sklearn_comparison = {
        'tree': sklearn_tree_metrics,
        'forest': sklearn_rf_metrics
    }
    visualization.plot_tree_vs_forest_comparison(
        sklearn_comparison,
        title="Image Classification: Scikit-learn Tree vs Forest",
        filename="image_sklearn_tree_vs_forest.png"
    )
    
    # Custom Tree vs Forest
    custom_comparison = {
        'tree': custom_tree_metrics,
        'forest': custom_rf_metrics
    }
    visualization.plot_tree_vs_forest_comparison(
        custom_comparison,
        title="Image Classification: Custom Tree vs Forest",
        filename="image_custom_tree_vs_forest.png"
    )
    
    # Class names for image dataset
    class_names = ['Class 0', 'Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5']
    
    # Plot confusion matrices
    visualization.plot_confusion_matrix(
        y_test, sklearn_rf_test_pred,
        class_names=class_names,
        title="Scikit-learn Random Forest - Confusion Matrix",
        filename="image_sklearn_rf_confusion_matrix.png"
    )
    
    visualization.plot_confusion_matrix(
        y_test, custom_rf_test_pred,
        class_names=class_names,
        title="Custom Random Forest - Confusion Matrix",
        filename="image_custom_rf_confusion_matrix.png"
    )
    
    return {
        'sklearn_tree_metrics': sklearn_tree_metrics,
        'sklearn_forest_metrics': sklearn_rf_metrics,
        'custom_tree_metrics': custom_tree_metrics,
        'custom_forest_metrics': custom_rf_metrics,
        # Keep old keys for backward compatibility
        'tree_metrics': sklearn_tree_metrics,
        'forest_metrics': sklearn_rf_metrics
    }


def experiment_3_custom_vs_sklearn(X_train, X_test, y_train, y_test):
    """
    Experiment 3: Compare custom (from scratch) vs scikit-learn implementations.
    
    Demonstrates that our custom implementation works correctly on image data.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 3: Custom Implementation vs Scikit-learn")
    print("="*70)
    
    # Train Custom Decision Tree
    print("\nTraining Custom Decision Tree (from scratch)...")
    start_time = time.time()
    custom_tree = DecisionTree(
        max_depth=15,
        min_samples_split=5,
        criterion='gini',
        random_state=config.RANDOM_STATE
    )
    custom_tree.fit(X_train, y_train)
    custom_tree_time = time.time() - start_time
    
    custom_tree_test_pred = custom_tree.predict(X_test)
    custom_tree_metrics = utils.calculate_metrics(y_test, custom_tree_test_pred)
    custom_tree_metrics['test_accuracy'] = accuracy_score(y_test, custom_tree_test_pred)
    custom_tree_metrics['training_time'] = custom_tree_time
    
    utils.print_metrics(custom_tree_metrics, "Custom Decision Tree Performance")
    
    # Train Custom Random Forest (fewer trees for speed)
    print("\nTraining Custom Random Forest (from scratch)...")
    print("Note: Using 30 trees for faster training on high-dimensional data")
    start_time = time.time()
    custom_rf = RandomForest(
        n_estimators=30,
        max_depth=15,
        max_features='sqrt',
        random_state=config.RANDOM_STATE,
        n_jobs=1
    )
    custom_rf.fit(X_train, y_train)
    custom_rf_time = time.time() - start_time
    
    custom_rf_test_pred = custom_rf.predict(X_test)
    custom_rf_metrics = utils.calculate_metrics(y_test, custom_rf_test_pred)
    custom_rf_metrics['test_accuracy'] = accuracy_score(y_test, custom_rf_test_pred)
    custom_rf_metrics['training_time'] = custom_rf_time
    
    utils.print_metrics(custom_rf_metrics, "Custom Random Forest Performance")
    
    # Train Scikit-learn Random Forest for comparison
    print("\nTraining Scikit-learn Random Forest...")
    start_time = time.time()
    sklearn_rf = RandomForestClassifier(
        n_estimators=30,
        max_depth=15,
        random_state=config.RANDOM_STATE,
        n_jobs=-1
    )
    sklearn_rf.fit(X_train, y_train)
    sklearn_rf_time = time.time() - start_time
    
    sklearn_rf_test_pred = sklearn_rf.predict(X_test)
    sklearn_rf_metrics = utils.calculate_metrics(y_test, sklearn_rf_test_pred)
    sklearn_rf_metrics['test_accuracy'] = accuracy_score(y_test, sklearn_rf_test_pred)
    sklearn_rf_metrics['training_time'] = sklearn_rf_time
    
    utils.print_metrics(sklearn_rf_metrics, "Scikit-learn Random Forest Performance")
    
    # Comparison
    print("\n" + "="*70)
    print("COMPARISON: Custom vs Scikit-learn")
    print("="*70)
    print(f"\nCustom RF Test Accuracy:    {custom_rf_metrics['test_accuracy']:.4f}")
    print(f"Scikit-learn RF Test Accuracy: {sklearn_rf_metrics['test_accuracy']:.4f}")
    print(f"Difference: {abs(custom_rf_metrics['test_accuracy'] - sklearn_rf_metrics['test_accuracy']):.4f}")
    print(f"\nCustom RF Training Time:    {custom_rf_time:.3f}s")
    print(f"Scikit-learn RF Training Time: {sklearn_rf_time:.3f}s")
    
    return {
        'custom_tree_metrics': custom_tree_metrics,
        'custom_rf_metrics': custom_rf_metrics,
        'sklearn_rf_metrics': sklearn_rf_metrics
    }


def main():
    """
    Main function to run all experiments on Image Classification dataset.
    """
    print("\n" + "="*70)
    print("RANDOM FOREST EXPERIMENTS - IMAGE CLASSIFICATION DATASET")
    print("="*70)
    
    # Load and preprocess data
    print("\nLoading Intel Image Classification dataset...")
    X_train, X_test, y_train, y_test = utils.load_intel_image_data()
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    print(f"Number of features: {X_train.shape[1]}")
    print(f"Number of classes: {len(np.unique(y_train))}")
    print(f"Class distribution (Train): {dict(zip(*np.unique(y_train, return_counts=True)))}")
    
    print("\nPreprocessing data...")
    X_train, X_test, y_train, y_test = utils.preprocess_image_data(X_train, X_test, y_train, y_test)
    print(f"Training set size: {len(y_train)}")
    print(f"Test set size: {len(y_test)}")
    
    # Run experiments
    exp1_results = experiment_1_vary_n_estimators(X_train, X_test, y_train, y_test)
    exp2_results = experiment_2_tree_vs_forest(X_train, X_test, y_train, y_test)
    exp3_results = experiment_3_custom_vs_sklearn(X_train, X_test, y_train, y_test)
    
    # Combine results
    all_results = {
        **exp1_results,
        **exp2_results,
        **exp3_results
    }
    
    # Save results
    utils.save_results(all_results, 'image_classification_results.pkl')
    utils.save_results_json(all_results, 'image_classification_results.json')
    
    # Create comprehensive comparison plots
    print("\nGenerating comprehensive comparison plots...")
    
    # Detailed custom vs sklearn comparison
    visualization_comparison.plot_custom_vs_sklearn_detailed(
        exp3_results['custom_rf_metrics'],
        exp3_results['sklearn_rf_metrics'],
        "Image Classification",
        filename="image_custom_vs_sklearn_detailed.png"
    )
    
    # Complete implementation comparison grid
    visualization_comparison.plot_implementation_comparison_grid(
        exp1_results,
        exp2_results,
        exp3_results,
        "Image Classification",
        filename="image_complete_comparison.png"
    )
    
    # Create summary plot
    print("\nGenerating summary plot...")
    visualization.create_summary_plot(
        all_results,
        "Image Classification",
        filename="image_classification_summary.png"
    )
    
    print("\n" + "="*70)
    print("EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"\nResults saved to: {config.RESULTS_DIR}")
    print(f"Plots saved to: {config.PLOTS_DIR}")
    
    # Print key findings
    print("\n" + "="*70)
    print("KEY FINDINGS")
    print("="*70)
    
    print("\n1. Effect of Number of Trees (Scikit-learn):")
    print(f"   - With 1 tree: {exp1_results['sklearn_test_accuracies'][0]:.4f}")
    print(f"   - With 10 trees: {exp1_results['sklearn_test_accuracies'][1]:.4f}")
    print(f"   - With 100 trees: {exp1_results['sklearn_test_accuracies'][3]:.4f}")
    print(f"   - With 300 trees: {exp1_results['sklearn_test_accuracies'][4]:.4f}")
    print(f"   - Improvement from 1 to 300 trees: {exp1_results['sklearn_test_accuracies'][4] - exp1_results['sklearn_test_accuracies'][0]:.4f}")
    
    print("\n2. Effect of Number of Trees (Custom):")
    custom_accs = [a for a in exp1_results['custom_test_accuracies'] if a is not None]
    if len(custom_accs) >= 4:
        print(f"   - With 1 tree: {custom_accs[0]:.4f}")
        print(f"   - With 10 trees: {custom_accs[1]:.4f}")
        print(f"   - With 100 trees: {custom_accs[3]:.4f}")
        print(f"   - Improvement from 1 to 100 trees: {custom_accs[3] - custom_accs[0]:.4f}")
    
    print("\n3. Decision Tree vs Random Forest (Scikit-learn):")
    sklearn_tree_acc = exp2_results['sklearn_tree_metrics']['test_accuracy']
    sklearn_rf_acc = exp2_results['sklearn_forest_metrics']['test_accuracy']
    print(f"   - Decision Tree accuracy: {sklearn_tree_acc:.4f}")
    print(f"   - Random Forest accuracy: {sklearn_rf_acc:.4f}")
    print(f"   - Improvement: {sklearn_rf_acc - sklearn_tree_acc:.4f}")
    
    print("\n4. Decision Tree vs Random Forest (Custom):")
    custom_tree_acc = exp2_results['custom_tree_metrics']['test_accuracy']
    custom_rf_acc = exp2_results['custom_forest_metrics']['test_accuracy']
    print(f"   - Decision Tree accuracy: {custom_tree_acc:.4f}")
    print(f"   - Random Forest accuracy: {custom_rf_acc:.4f}")
    print(f"   - Improvement: {custom_rf_acc - custom_tree_acc:.4f}")
    
    print("\n5. Custom vs Scikit-learn Comparison:")
    print(f"   - Tree accuracy difference: {abs(sklearn_tree_acc - custom_tree_acc):.4f}")
    print(f"   - Forest accuracy difference: {abs(sklearn_rf_acc - custom_rf_acc):.4f}")
    print(f"   - Custom implementation {'validated ✓' if abs(sklearn_rf_acc - custom_rf_acc) < 0.08 else 'needs review'}")
    
    print("\n6. Training Time Comparison:")
    sklearn_tree_time = exp2_results['sklearn_tree_metrics']['training_time']
    custom_tree_time = exp2_results['custom_tree_metrics']['training_time']
    sklearn_rf_time = exp2_results['sklearn_forest_metrics']['training_time']
    custom_rf_time = exp2_results['custom_forest_metrics']['training_time']
    print(f"   - Sklearn Tree: {sklearn_tree_time:.3f}s vs Custom Tree: {custom_tree_time:.3f}s")
    print(f"   - Sklearn Forest: {sklearn_rf_time:.3f}s vs Custom Forest: {custom_rf_time:.3f}s")
    print(f"   - Sklearn is {custom_rf_time/sklearn_rf_time:.1f}x faster for Random Forest")

if __name__ == "__main__":
    main()
