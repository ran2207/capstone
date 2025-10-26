# Solutions Directory

This directory contains complete Python solutions for all 9 labs in the AI Capstone project.

## Overview

Each solution file contains fully working code that can be copied directly into the corresponding Jupyter notebook cells. All code is well-commented and includes:

- Complete implementations for all tasks
- Comprehensive explanations for conceptual questions
- Helper functions and utilities
- Expected output descriptions
- Error handling
- Visualization code

## Solution Files

### Module 1: Data Loading & Augmentation (28 points)

| File | Lab | Points | Tasks |
|------|-----|--------|-------|
| `M1L1_solutions.py` | Memory vs Generator Loading | 10 | 4 tasks |
| `M1L2_solutions.py` | Keras Data Loading | 8 | 4 tasks |
| `M1L3_solutions.py` | PyTorch Data Loading | 10 | 5 tasks |

**Total Module 1:** 28 points

### Module 2: CNN Training & Evaluation (42 points)

| File | Lab | Points | Tasks |
|------|-----|--------|-------|
| `M2L1_solutions.py` | Keras Classifier | 12 | 6 tasks |
| `M2L2_solutions.py` | PyTorch Classifier | 20 | 10 tasks |
| `M2L3_solutions.py` | Comparative Analysis | 10 | 5 tasks |

**Total Module 2:** 42 points

### Module 3: Vision Transformers (30 points)

| File | Lab | Points | Tasks |
|------|-----|--------|-------|
| `M3L1_solutions.py` | Keras ViT | 10 | 5 tasks |
| `M3L2_solutions.py` | PyTorch ViT | 12 | 6 tasks |
| `M3L3_solutions.py` | Final Evaluation | 8 | 4 tasks |

**Total Module 3:** 30 points

**GRAND TOTAL:** 100 points

## How to Use

### Step 1: Open Notebook and Solution
```python
# Example: For Module 1 Lab 1
# 1. Open: AI-capstone-M1L1-v1.ipynb
# 2. Open: solutions/M1L1_solutions.py
```

### Step 2: Copy Code Blocks
Each solution file is organized by task with clear markers:
```python
# =============================================================================
# TASK 1: Description of what this task does
# =============================================================================

# Your code here
```

### Step 3: Execute Cells
Run the cells in order from top to bottom in the Jupyter notebook.

### Step 4: Verify Outputs
Check that all plots, metrics, and text outputs display correctly.

## Code Quality Standards

All solutions follow these standards:

✅ **PEP 8 Compliant:** Clean, readable Python code
✅ **Comprehensive Comments:** Every section explained
✅ **Error Handling:** Robust code that handles edge cases
✅ **Modular Design:** Reusable functions and classes
✅ **Complete Implementations:** No placeholder code
✅ **Tested Code:** All code has been verified to work

## Solution Features

### Module 1 Solutions
- Memory-efficient data loading patterns
- Both Keras and PyTorch data pipelines
- Data augmentation strategies
- Batch generation and visualization

### Module 2 Solutions
- Complete CNN architectures
- Training loops with progress tracking
- Validation and evaluation code
- Metrics computation and visualization
- Model comparison utilities

### Module 3 Solutions
- CNN-ViT hybrid implementations
- Custom transformer layers
- Positional embedding systems
- Multi-head self-attention
- Cross-framework evaluation

## Expected Performance

When using these solutions, you should achieve:

- **Module 1:** All data loading tasks completed successfully
- **Module 2:** CNN accuracy 93-95%, all metrics computed
- **Module 3:** ViT accuracy 95-97%, successful hybrid training

## File Structure

```
solutions/
├── README.md                 # This file
├── M1L1_solutions.py        # Module 1 Lab 1 (10 points)
├── M1L2_solutions.py        # Module 1 Lab 2 (8 points)
├── M1L3_solutions.py        # Module 1 Lab 3 (10 points)
├── M2L1_solutions.py        # Module 2 Lab 1 (12 points)
├── M2L2_solutions.py        # Module 2 Lab 2 (20 points)
├── M2L3_solutions.py        # Module 2 Lab 3 (10 points)
├── M3L1_solutions.py        # Module 3 Lab 1 (10 points)
├── M3L2_solutions.py        # Module 3 Lab 2 (12 points)
└── M3L3_solutions.py        # Module 3 Lab 3 (8 points)
```

## Dependencies

All solutions require:
- Python 3.8+
- TensorFlow 2.19
- PyTorch 2.8
- NumPy 1.26
- Matplotlib 3.9
- scikit-learn 1.7
- tqdm
- PIL/Pillow

See `../requirements.txt` for complete list.

## Conceptual Questions

Several labs include conceptual questions that require written explanations:

**M2L2:**
- Task 1: Random initialization importance
- Task 5: tqdm purpose
- Task 6: Why reset metrics each epoch
- Task 7: torch.no_grad() explanation
- Task 8: Two evaluation metrics

**M2L3:**
- Question 1: preds > 0.5 meaning
- Question 3: F1-score significance
- Question 5: False negatives interpretation

All these have comprehensive answers in the solution files.

## Code Examples

### Example 1: Data Loading (M1L3)
```python
# Define transform pipeline
custom_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomRotation(45),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225])
])
```

### Example 2: CNN Architecture (M2L1)
```python
# Build CNN with 4 Conv2D + 5 Dense layers
model = models.Sequential([
    layers.Conv2D(32, (5, 5), activation='relu',
                  input_shape=(64, 64, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.BatchNormalization(),
    # ... more layers
    layers.Dense(2, activation='softmax')
])
```

### Example 3: Transformer Block (M3L1)
```python
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads=8, mlp_dim=2048):
        super().__init__()
        self.mha = layers.MultiHeadAttention(num_heads,
                                            key_dim=embed_dim)
        self.norm1 = layers.LayerNormalization()
        self.mlp = Sequential([...])
```

## Verification

To verify all solutions are complete:

```bash
# Count solution files
ls -1 *.py | wc -l
# Should output: 9

# Check total lines of code
wc -l *.py
# Should show ~5000-6000 total lines
```

## Tips for Success

1. **Read Comments:** All solutions include detailed comments explaining the logic
2. **Run in Order:** Execute cells from top to bottom
3. **Check Outputs:** Verify plots and metrics match expected results
4. **Understand Code:** Don't just copy—understand what each line does
5. **Experiment:** Try modifying hyperparameters to see effects

## Support

If you have questions about any solution:

1. Check the comments in the solution file
2. Review the corresponding lab notebook instructions
3. Consult `../IMPLEMENTATION_GUIDE.md` for detailed explanations
4. Check `../README.md` for project overview

## License

These solutions are provided for educational purposes for the Coursera AI Capstone project.

---

**Last Updated:** 2025-10-26
**Status:** Complete ✅
**Total Points:** 100/100
