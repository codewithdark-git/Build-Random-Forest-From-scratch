"""
Experiments on Heart Disease UCI Dataset

This script performs the experiments specified in Question 2, Part B:
1. Vary n_estimators and plot test accuracy
2. Compare Decision Tree vs Random Forest
3. Compare custom (scratch) vs scikit-learn implementations

Dataset: Heart Disease UCI
Task: Binary classification (disease vs no disease)
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
    Runs BOTH custom and scikit-learn implementations for comparison.
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
        
        # Train Custom Random Forest (use fewer trees for very large n_estimators)
        if n_est <= 100:
            print(f"  Training Custom Random Forest...")
            start_time = time.time()
            custom_rf = RandomForest(
                n_estimators=n_est,
                max_features='sqrt',
                random_state=config.RANDOM_STATE,
                n_jobs=1
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
            # Skip custom for 300 trees (too slow)
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
        title="Heart Disease: Accuracy vs Number of Trees (Scikit-learn)",
        filename="heart_disease_sklearn_accuracy_vs_n_estimators.png"
    )
    
    visualization.plot_training_time_comparison(
        n_estimators_list,
        sklearn_training_times,
        title="Heart Disease: Training Time vs Number of Trees (Scikit-learn)",
        filename="heart_disease_sklearn_training_time.png"
    )
    
    # Plot results - Custom (only for n_estimators <= 100)
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
            title="Heart Disease: Accuracy vs Number of Trees (Custom)",
            filename="heart_disease_custom_accuracy_vs_n_estimators.png"
        )
        
        visualization.plot_training_time_comparison(
            custom_n_est,
            custom_time_filtered,
            title="Heart Disease: Training Time vs Number of Trees (Custom)",
            filename="heart_disease_custom_training_time.png"
        )
    
    # Comparison plot (for n_estimators where both exist)
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
    filepath = os.path.join(config.PLOTS_DIR, 'heart_disease_exp1_comparison.png')
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
    
    Uses default parameters for fair comparison.
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
        max_depth=None,
        min_samples_split=2,
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
    start_time = time.time()
    custom_rf = RandomForest(
        n_estimators=100,
        max_depth=None,
        max_features='sqrt',
        random_state=config.RANDOM_STATE,
        n_jobs=1
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
        title="Heart Disease: Scikit-learn Tree vs Forest",
        filename="heart_disease_sklearn_tree_vs_forest.png"
    )
    
    # Custom Tree vs Forest
    custom_comparison = {
        'tree': custom_tree_metrics,
        'forest': custom_rf_metrics
    }
    visualization.plot_tree_vs_forest_comparison(
        custom_comparison,
        title="Heart Disease: Custom Tree vs Forest",
        filename="heart_disease_custom_tree_vs_forest.png"
    )
    
    # Plot confusion matrices for sklearn
    visualization.plot_confusion_matrix(
        y_test, sklearn_tree_test_pred,
        class_names=['No Disease', 'Disease'],
        title="Scikit-learn Decision Tree - Confusion Matrix",
        filename="heart_disease_sklearn_tree_confusion_matrix.png"
    )
    
    visualization.plot_confusion_matrix(
        y_test, sklearn_rf_test_pred,
        class_names=['No Disease', 'Disease'],
        title="Scikit-learn Random Forest - Confusion Matrix",
        filename="heart_disease_sklearn_rf_confusion_matrix.png"
    )
    
    # Plot confusion matrices for custom
    visualization.plot_confusion_matrix(
        y_test, custom_tree_test_pred,
        class_names=['No Disease', 'Disease'],
        title="Custom Decision Tree - Confusion Matrix",
        filename="heart_disease_custom_tree_confusion_matrix.png"
    )
    
    visualization.plot_confusion_matrix(
        y_test, custom_rf_test_pred,
        class_names=['No Disease', 'Disease'],
        title="Custom Random Forest - Confusion Matrix",
        filename="heart_disease_custom_rf_confusion_matrix.png"
    )
    
    # Feature importance (sklearn)
    feature_names = utils.get_feature_names_heart_disease()
    visualization.plot_feature_importance(
        sklearn_rf.feature_importances_,
        feature_names=feature_names,
        top_n=10,
        title="Scikit-learn Random Forest - Feature Importance",
        filename="heart_disease_sklearn_feature_importance.png"
    )
    
    # Feature importance (custom)
    custom_importances = custom_rf.feature_importances()
    visualization.plot_feature_importance(
        custom_importances,
        feature_names=feature_names,
        top_n=10,
        title="Custom Random Forest - Feature Importance",
        filename="heart_disease_custom_feature_importance.png"
    )
    
    return {
        'sklearn_tree_metrics': sklearn_tree_metrics,
        'sklearn_forest_metrics': sklearn_rf_metrics,
        'custom_tree_metrics': custom_tree_metrics,
        'custom_forest_metrics': custom_rf_metrics,
        'sklearn_feature_importances': sklearn_rf.feature_importances_,
        'custom_feature_importances': custom_importances,
        # Keep old keys for backward compatibility
        'tree_metrics': sklearn_tree_metrics,
        'forest_metrics': sklearn_rf_metrics,
        'feature_importances': sklearn_rf.feature_importances_
    }


def experiment_3_custom_vs_sklearn(X_train, X_test, y_train, y_test):
    """
    Experiment 3: Compare custom (from scratch) vs scikit-learn implementations.
    
    Demonstrates that our custom implementation works correctly.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 3: Custom Implementation vs Scikit-learn")
    print("="*70)
    
    # Train Custom Decision Tree
    print("\nTraining Custom Decision Tree (from scratch)...")
    start_time = time.time()
    custom_tree = DecisionTree(
        max_depth=10,
        min_samples_split=2,
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
    print("\nTraining Custom Random Forest (from scratch)...")
    start_time = time.time()
    custom_rf = RandomForest(
        n_estimators=50,  # Use fewer trees for faster training
        max_depth=10,
        max_features='sqrt',
        random_state=config.RANDOM_STATE,
        n_jobs=1  # Sequential for custom implementation
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
    
    # Train Scikit-learn Random Forest for comparison
    print("\nTraining Scikit-learn Random Forest...")
    start_time = time.time()
    sklearn_rf = RandomForestClassifier(
        n_estimators=50,
        max_depth=10,
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
    Main function to run all experiments on Heart Disease dataset.
    """
    print("\n" + "="*70)
    print("RANDOM FOREST EXPERIMENTS - HEART DISEASE UCI DATASET")
    print("="*70)
    
    # Load and preprocess data
    print("\nLoading Heart Disease UCI dataset...")
    X, y = utils.load_heart_disease_data()
    print(f"Dataset shape: {X.shape}")
    print(f"Number of samples: {len(y)}")
    print(f"Number of features: {X.shape[1]}")
    print(f"Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    
    print("\nPreprocessing data...")
    X_train, X_test, y_train, y_test = utils.preprocess_tabular_data(X, y)
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
    utils.save_results(all_results, 'heart_disease_results.pkl')
    utils.save_results_json(all_results, 'heart_disease_results.json')
    
    # Create comprehensive comparison plots
    print("\nGenerating comprehensive comparison plots...")
    
    # Detailed custom vs sklearn comparison
    visualization_comparison.plot_custom_vs_sklearn_detailed(
        exp3_results['custom_rf_metrics'],
        exp3_results['sklearn_rf_metrics'],
        "Heart Disease UCI",
        filename="heart_disease_custom_vs_sklearn_detailed.png"
    )
    
    # Complete implementation comparison grid
    visualization_comparison.plot_implementation_comparison_grid(
        exp1_results,
        exp2_results,
        exp3_results,
        "Heart Disease UCI",
        filename="heart_disease_complete_comparison.png"
    )
    
    # Create summary plot
    print("\nGenerating summary plot...")
    visualization.create_summary_plot(
        all_results,
        "Heart Disease UCI",
        filename="heart_disease_summary.png"
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
    
    improvement = exp1_results['sklearn_test_accuracies'][-1] - exp1_results['sklearn_test_accuracies'][0]
    print(f"   - Improvement from 1 to 300 trees: {improvement:.4f}")
    
    print("\n2. Decision Tree vs Random Forest (Scikit-learn):")
    tree_acc = exp2_results['sklearn_tree_metrics']['test_accuracy']
    rf_acc = exp2_results['sklearn_forest_metrics']['test_accuracy']
    print(f"   - Decision Tree accuracy: {tree_acc:.4f}")
    print(f"   - Random Forest accuracy: {rf_acc:.4f}")
    print(f"   - Improvement: {rf_acc - tree_acc:.4f}")
    
    print("\n3. Training Time:")
    tree_time = exp2_results['sklearn_tree_metrics']['training_time']
    rf_time = exp2_results['sklearn_forest_metrics']['training_time']
    print(f"   - Decision Tree: {tree_time:.3f}s")
    print(f"   - Random Forest (100 trees): {rf_time:.3f}s")
    print(f"   - Time ratio: {rf_time/tree_time:.2f}x")
    
    print("\n4. Most Important Features:")
    feature_names = utils.get_feature_names_heart_disease()
    importances = exp2_results['feature_importances']
    top_indices = np.argsort(importances)[::-1][:5]
    for i, idx in enumerate(top_indices, 1):
        print(f"   {i}. {feature_names[idx]}: {importances[idx]:.4f}")
    
    print("\n5. Custom Implementation Validation:")
    custom_acc = exp3_results['custom_rf_metrics']['test_accuracy']
    sklearn_acc = exp3_results['sklearn_rf_metrics']['test_accuracy']
    print(f"   - Custom Random Forest accuracy: {custom_acc:.4f}")
    print(f"   - Scikit-learn RF accuracy: {sklearn_acc:.4f}")
    print(f"   - Difference: {abs(custom_acc - sklearn_acc):.4f}")
    print(f"   - Custom implementation {'validated ✓' if abs(custom_acc - sklearn_acc) < 0.08 else 'needs review'}")

if __name__ == "__main__":
    main()
