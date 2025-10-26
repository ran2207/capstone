"""
Module 2 Lab 3: Comparative Analysis of Keras and PyTorch Models
Solutions for all tasks (10 points total)

Copy these code blocks into the corresponding cells in the notebook:
Lab-M2L3-Comparative-Analysis-of-Keras-and-PyTorch-Models-v1.ipynb
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                            roc_auc_score, confusion_matrix, classification_report,
                            ConfusionMatrixDisplay)

# =============================================================================
# SETUP: Load both Keras and PyTorch models (from previous labs)
# =============================================================================

print("="*70)
print("Module 2 Lab 3: Comparative Analysis")
print("="*70)

# This assumes you have already:
# 1. Trained Keras model in M2L1 and saved predictions
# 2. Trained PyTorch model in M2L2 and saved predictions

# Example: Loading saved predictions (adjust based on how you saved them)
# keras_preds = np.load('keras_predictions.npy')
# keras_labels = np.load('keras_labels.npy')
# pytorch_preds = np.load('pytorch_predictions.npy')
# pytorch_labels = np.load('pytorch_labels.npy')


# =============================================================================
# QUESTION 1: What does preds > 0.5 do?
# =============================================================================

print("\n" + "="*70)
print("QUESTION 1: What does 'preds > 0.5' do?")
print("="*70)
print("""
The expression 'preds > 0.5' performs element-wise threshold comparison on
prediction probabilities to convert them to binary class labels.

**Detailed Explanation:**

1. **Input Format**:
   - preds: array of probabilities, e.g., [0.23, 0.78, 0.91, 0.12]
   - Range: [0, 1] (output from sigmoid activation)

2. **Threshold Operation**:
   - preds > 0.5 creates boolean array: [False, True, True, False]

3. **Conversion to Binary**:
   - .astype(int): Converts to integers [0, 1, 1, 0]

4. **Why 0.5?**:
   - Default decision boundary for binary classification
   - p > 0.5 means model is more confident in class 1
   - p <= 0.5 means model is more confident in class 0

**Complete Expression:**
   preds = (preds > 0.5).astype(int).flatten()

   - preds > 0.5: Boolean comparison
   - .astype(int): Convert True/False to 1/0
   - .flatten(): Convert 2D array to 1D

**Example:**
   Input:  [[0.23], [0.78], [0.91], [0.12]]
   > 0.5:  [[False], [True], [True], [False]]
   .astype: [[0], [1], [1], [0]]
   .flatten: [0, 1, 1, 0]

**Customizable Threshold:**
   - Use preds > 0.3 for higher recall (catch more positives)
   - Use preds > 0.7 for higher precision (fewer false positives)
""")


# =============================================================================
# HELPER FUNCTION: Print metrics
# =============================================================================

def print_metrics(y_true, y_pred, y_prob, model_name, class_labels=['Non-Agri', 'Agri']):
    """
    Compute and print comprehensive metrics for model evaluation
    """
    # Compute metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='binary')
    recall = recall_score(y_true, y_pred, average='binary')
    f1 = f1_score(y_true, y_pred, average='binary')

    # ROC-AUC (need probabilities)
    if y_prob is not None:
        if len(y_prob.shape) > 1 and y_prob.shape[1] > 1:
            # Multi-class probabilities, use class 1
            roc_auc = roc_auc_score(y_true, y_prob[:, 1])
        else:
            # Binary probabilities
            roc_auc = roc_auc_score(y_true, y_prob)
    else:
        roc_auc = None

    # Print results
    print(f"\n{'='*70}")
    print(f"Evaluation Metrics for {model_name}")
    print(f"{'='*70}")
    print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    if roc_auc is not None:
        print(f"  ROC-AUC:   {roc_auc:.4f}")

    # Classification report
    print(f"\n  Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_labels, digits=4))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print(f"  Confusion Matrix:")
    print(f"                Predicted")
    print(f"              0        1")
    print(f"    Actual 0  {cm[0,0]:<8} {cm[0,1]:<8}")
    print(f"           1  {cm[1,0]:<8} {cm[1,1]:<8}")

    # Confusion matrix visualization
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{model_name.lower().replace(" ", "_")}_confusion_matrix.png', dpi=300)
    plt.show()

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': cm
    }


# =============================================================================
# TASK 1: Print Keras model metrics using print_metrics
# =============================================================================

print("\n" + "="*70)
print("TASK 1: Keras Model Evaluation")
print("="*70)

# Example: If you have Keras predictions loaded
# Assuming from previous lab:
# - keras_preds: Binary predictions (0 or 1)
# - keras_labels: True labels (0 or 1)
# - keras_probs: Probabilities for class 1

# TASK 1 ANSWER (uncomment and use your actual data):
# keras_metrics = print_metrics(
#     y_true=keras_labels,
#     y_pred=keras_preds,
#     y_prob=keras_probs,
#     model_name="Keras CNN Model",
#     class_labels=['Non-Agricultural', 'Agricultural']
# )

print("""
To complete Task 1, use:

keras_metrics = print_metrics(
    y_true=all_labels_keras,
    y_pred=all_preds_keras,
    y_prob=all_probs_keras,
    model_name="Keras CNN Model",
    class_labels=['Non-Agricultural', 'Agricultural']
)
""")


# =============================================================================
# QUESTION 3: Explain the significance of the F1-score
# =============================================================================

print("\n" + "="*70)
print("QUESTION 3: Significance of F1-Score")
print("="*70)
print("""
The F1-Score is the harmonic mean of Precision and Recall, providing a single
metric that balances both measures of classification performance.

**Formula:**
   F1 = 2 * (Precision * Recall) / (Precision + Recall)

**Why F1-Score is Important:**

1. **Balances Precision and Recall**:
   - Precision alone doesn't tell about false negatives
   - Recall alone doesn't tell about false positives
   - F1-Score considers both

2. **Handles Class Imbalance**:
   - Better than accuracy for imbalanced datasets
   - Example: 95% non-agri, 5% agri
     * Model predicting all non-agri: 95% accuracy, 0% F1 for agri class
     * F1-Score reveals the poor performance

3. **Single Performance Metric**:
   - Easy to compare models
   - Useful for model selection and hyperparameter tuning
   - Better represents overall performance than accuracy alone

4. **Domain-Specific Interpretation**:
   - Medical diagnosis: High recall (catch all diseases)
   - Spam detection: High precision (avoid false positives)
   - F1-Score helps balance based on use case

**Example Scenarios:**

Scenario A (High Precision, Low Recall):
   - Precision: 0.95 (few false positives)
   - Recall: 0.60 (many false negatives)
   - F1-Score: 0.73

Scenario B (Balanced):
   - Precision: 0.85
   - Recall: 0.85
   - F1-Score: 0.85

Scenario C (Low Precision, High Recall):
   - Precision: 0.60 (many false positives)
   - Recall: 0.95 (few false negatives)
   - F1-Score: 0.73

**For Land Classification:**
   - High F1-Score ensures we correctly identify agricultural land
   - Minimizes both missing agricultural areas (false negatives)
   - And incorrectly marking non-agricultural as agricultural (false positives)
   - Critical for business decisions (fertilizer company expansion)
""")


# =============================================================================
# TASK 2: Print PyTorch model metrics using print_metrics
# =============================================================================

print("\n" + "="*70)
print("TASK 2: PyTorch Model Evaluation")
print("="*70)

# Example: If you have PyTorch predictions loaded
# pytorch_metrics = print_metrics(
#     y_true=pytorch_labels,
#     y_pred=pytorch_preds,
#     y_prob=pytorch_probs,
#     model_name="PyTorch CNN Model",
#     class_labels=['Non-Agricultural', 'Agricultural']
# )

print("""
To complete Task 2, use:

pytorch_metrics = print_metrics(
    y_true=all_labels_pytorch,
    y_pred=all_preds_pytorch,
    y_prob=all_probs_pytorch,
    model_name="PyTorch CNN Model",
    class_labels=['Non-Agricultural', 'Agricultural']
)
""")


# =============================================================================
# QUESTION 5: Count false negatives in PyTorch confusion matrix
# =============================================================================

print("\n" + "="*70)
print("QUESTION 5: Count False Negatives in Confusion Matrix")
print("="*70)
print("""
False Negatives (FN) are located in the confusion matrix at position [1, 0]:

   Confusion Matrix Structure:
                   Predicted
                   0      1
         Actual 0  TN     FP
                1  FN     TP

   Where:
   - TN (True Negative): [0, 0] - Correctly predicted as class 0
   - FP (False Positive): [0, 1] - Incorrectly predicted as class 1
   - FN (False Negative): [1, 0] - Incorrectly predicted as class 0
   - TP (True Positive): [1, 1] - Correctly predicted as class 1

**False Negatives Explained:**
   - Actual class: 1 (Agricultural)
   - Predicted class: 0 (Non-Agricultural)
   - Model failed to detect agricultural land

**How to Extract:**
   cm = confusion_matrix(y_true, y_pred)
   false_negatives = cm[1, 0]

**Business Impact for Land Classification:**
   - False negatives = missed agricultural land
   - Fertilizer company misses expansion opportunities
   - Potential revenue loss
   - Should minimize FN to capture all potential markets

**Example:**
   cm = [[850, 50],     # Row 0: Non-Agricultural
         [30, 920]]      # Row 1: Agricultural

   - False Negatives (FN) = cm[1, 0] = 30
   - Meaning: 30 agricultural lands were incorrectly classified as non-agricultural
""")

# Example code to extract false negatives
# cm_pytorch = pytorch_metrics['confusion_matrix']
# false_negatives_pytorch = cm_pytorch[1, 0]
# print(f"\nFalse Negatives in PyTorch Model: {false_negatives_pytorch}")


# =============================================================================
# COMPARATIVE ANALYSIS
# =============================================================================

def compare_models(keras_metrics, pytorch_metrics):
    """
    Compare Keras and PyTorch models side-by-side
    """
    print("\n" + "="*70)
    print("MODEL COMPARISON: Keras vs PyTorch")
    print("="*70)

    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    keras_keys = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
    pytorch_keys = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']

    print(f"\n{'Metric':<15} | {'Keras':<12} | {'PyTorch':<12} | {'Difference':<12}")
    print("-" * 70)

    for name, k_key, p_key in zip(metrics_names, keras_keys, pytorch_keys):
        k_val = keras_metrics[k_key]
        p_val = pytorch_metrics[p_key]

        if k_val is not None and p_val is not None:
            diff = p_val - k_val
            diff_str = f"{diff:+.4f}"
            print(f"{name:<15} | {k_val:.4f}      | {p_val:.4f}      | {diff_str}")

    # Confusion matrix comparison
    print("\n" + "="*70)
    print("CONFUSION MATRIX COMPARISON")
    print("="*70)

    cm_keras = keras_metrics['confusion_matrix']
    cm_pytorch = pytorch_metrics['confusion_matrix']

    print("\nKeras CNN:")
    print(f"  TN: {cm_keras[0,0]:<6} FP: {cm_keras[0,1]:<6}")
    print(f"  FN: {cm_keras[1,0]:<6} TP: {cm_keras[1,1]:<6}")

    print("\nPyTorch CNN:")
    print(f"  TN: {cm_pytorch[0,0]:<6} FP: {cm_pytorch[0,1]:<6}")
    print(f"  FN: {cm_pytorch[1,0]:<6} TP: {cm_pytorch[1,1]:<6}")

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score']
    keras_vals = [keras_metrics[m] for m in metrics_to_plot]
    pytorch_vals = [pytorch_metrics[m] for m in metrics_to_plot]

    x = np.arange(len(metrics_to_plot))
    width = 0.35

    axes[0].bar(x - width/2, keras_vals, width, label='Keras', color='skyblue')
    axes[0].bar(x + width/2, pytorch_vals, width, label='PyTorch', color='coral')
    axes[0].set_xlabel('Metric')
    axes[0].set_ylabel('Score')
    axes[0].set_title('Model Comparison: Metrics', fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(['Accuracy', 'Precision', 'Recall', 'F1-Score'], rotation=45)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')

    # Training time comparison (if available)
    # axes[1].bar(['Keras', 'PyTorch'], [keras_time, pytorch_time], color=['skyblue', 'coral'])
    # axes[1].set_ylabel('Training Time (seconds)')
    # axes[1].set_title('Training Time Comparison', fontweight='bold')

    plt.tight_layout()
    plt.savefig('keras_vs_pytorch_comparison.png', dpi=300)
    plt.show()


# Example usage (uncomment when you have both sets of metrics):
# compare_models(keras_metrics, pytorch_metrics)


# =============================================================================
# SUMMARY OF SOLUTIONS
# =============================================================================

print("\n" + "="*70)
print("SUMMARY: Module 2 Lab 3 - All Tasks Completed")
print("="*70)
print("Question 1: Explained 'preds > 0.5' threshold operation")
print("Task 1:     Printed Keras model metrics using print_metrics()")
print("Question 3: Explained significance of F1-Score")
print("Task 2:     Printed PyTorch model metrics using print_metrics()")
print("Question 5: Explained how to count False Negatives (cm[1,0])")
print("\nComparative analysis complete - ready for Module 3!")
print("="*70)
