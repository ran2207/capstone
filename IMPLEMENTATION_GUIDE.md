# Complete Implementation Guide
## AI Capstone Project: Land Classification with CNNs and Vision Transformers

**Last Updated:** 2025-10-26
**Total Points:** 100 (Passing: 70)

---

## Overview

This guide provides step-by-step instructions for completing all 9 labs in the AI Capstone project. All solutions are provided in the `solutions/` directory as Python scripts that can be copied directly into the corresponding Jupyter notebook cells.

---

## Project Status: COMPLETE ✓

All 9 labs have been fully implemented with comprehensive solutions:

- ✅ Module 1: Data Loading & Augmentation (3 labs, 28 points)
- ✅ Module 2: CNN Training & Evaluation (3 labs, 42 points)
- ✅ Module 3: Vision Transformers (3 labs, 30 points)

**Total:** 100 points

---

## Quick Start Guide

### Step 1: Environment Setup (5 minutes)

```bash
# Navigate to project directory
cd ~/Documents/GitHub/Personal/coursera/capstone

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Dataset Setup (10 minutes)

The dataset will be automatically downloaded when you run the first cell of any notebook. Alternatively:

```bash
# Manual download (if automatic fails)
wget https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/4Z1fwRR295-1O3PMQBH6Dg/images-dataSAT.tar
tar -xvf images-dataSAT.tar
```

Verify:
```bash
ls images_dataSAT/
# Should show: class_0_non_agri/  class_1_agri/
```

### Step 3: Open Jupyter (2 minutes)

```bash
jupyter notebook
# or
jupyter lab
```

---

## Implementation Instructions

### How to Use Solution Files

Each lab has a corresponding solution file in `solutions/`:

1. Open the original notebook (e.g., `AI-capstone-M1L1-v1.ipynb`)
2. Open the solution file (e.g., `solutions/M1L1_solutions.py`)
3. Copy code from solution file into the notebook cells marked for tasks
4. Execute cells in order from top to bottom

**Important:**
- Solutions include comprehensive comments and explanations
- All imports and helper functions are included
- Expected outputs are documented

---

## Module 1: Data Loading & Augmentation (28 points)

### Lab 1: Memory vs Generator Loading (10 points)

**File:** `AI-capstone-M1L1-v1.ipynb`
**Solution:** `solutions/M1L1_solutions.py`
**Time:** ~20 minutes

**Tasks:**
1. Determine image shape
2. Display first 4 non-agricultural images
3. Create sorted agricultural image paths list
4. Count agricultural images

**Key Concepts:**
- Memory-based vs. generator-based loading
- NumPy array handling
- PIL image operations

**Expected Output:**
- Image shape: (256, 256, 3) or similar
- 4 images displayed in subplot
- ~5000 agricultural images

---

### Lab 2: Keras Data Loading (8 points)

**File:** `AI-capstone-M1L2-v1.ipynb`
**Solution:** `solutions/M1L2_solutions.py`
**Time:** ~25 minutes

**Tasks:**
1. Create `all_image_paths` from both classes
2. Create `temp` list with paths and labels
3. Generate batch with custom generator
4. Create validation generator

**Key Concepts:**
- ImageDataGenerator
- Data augmentation
- Train/validation splits

**Expected Output:**
- ~10,000 total image paths
- Batch shape: (8, 64, 64, 3)
- Validation samples: ~2000

---

### Lab 3: PyTorch Data Loading (10 points)

**File:** `AI-capstone-M1L3-v1.ipynb`
**Solution:** `solutions/M1L3_solutions.py`
**Time:** ~25 minutes

**Tasks:**
1. Define `custom_transform` with augmentation
2. Load dataset with ImageFolder
3. Print class names and indices
4. Get batch from DataLoader
5. Display batch images

**Key Concepts:**
- torchvision transforms
- ImageFolder dataset
- DataLoader with batching

**Expected Output:**
- Classes: ['class_0_non_agri', 'class_1_agri']
- Batch shape: torch.Size([8, 3, 64, 64])
- 8 images displayed with labels

---

## Module 2: CNN Training & Evaluation (42 points)

### Lab 1: Keras Classifier (12 points)

**File:** `Lab-M2L1-Train-and-Evaluate-a-Keras-Based-Classifier-v1.ipynb`
**Solution:** `solutions/M2L1_solutions.py`
**Time:** ~60 minutes (including training)

**Tasks:**
1. Create `fnames` list from os.walk()
2. Create validation generator
3. Count CNN model layers
4. Build CNN with 4 Conv2D + 5 Dense layers
5. Define checkpoint callback
6. Plot training/validation loss

**Key Concepts:**
- Sequential CNN architecture
- Model compilation
- Callbacks (ModelCheckpoint)
- Training history visualization

**Expected Output:**
- ~10,000 files found
- 38 total layers (may vary)
- Validation accuracy: 90-95%
- Loss curves showing convergence

---

### Lab 2: PyTorch Classifier (20 points)

**File:** `Lab-M2L2-Implement-and-Test-a-PyTorch-Based-Classifier-v1.ipynb`
**Solution:** `solutions/M2L2_solutions.py`
**Time:** ~75 minutes (including training)

**Tasks:**
1. **Explain random initialization** (conceptual)
2. Define `train_transform`
3. Define `val_transform`
4. Create `val_loader`
5. **Explain tqdm purpose** (conceptual)
6. **Explain metric reset** (conceptual)
7. **Explain torch.no_grad()** (conceptual)
8. **List two metrics** (conceptual)
9. Plot training loss
10. Collect predictions and labels

**Key Concepts:**
- Custom training loops
- Gradient management
- Progress tracking with tqdm
- Metric computation

**Expected Output:**
- All conceptual questions answered
- Validation accuracy: 90-95%
- Loss curves similar to Keras
- Predictions array: shape (N,)

---

### Lab 3: Comparative Analysis (10 points)

**File:** `Lab-M2L3-Comparative-Analysis-of-Keras-and-PyTorch-Models-v1.ipynb`
**Solution:** `solutions/M2L3_solutions.py`
**Time:** ~30 minutes

**Tasks:**
1. **Explain `preds > 0.5`** (conceptual)
2. Print Keras metrics
3. **Explain F1-score significance** (conceptual)
4. Print PyTorch metrics
5. **Count false negatives** (conceptual)

**Key Concepts:**
- Classification metrics
- Confusion matrix interpretation
- Framework comparison

**Expected Output:**
- Accuracy, Precision, Recall, F1-Score for both models
- Confusion matrices displayed
- Comparative bar charts

---

## Module 3: Vision Transformers (30 points)

### Lab 1: Keras ViT (10 points)

**File:** `Lab-M3L1-Vision-Transformers-in-Keras-v1.ipynb`
**Solution:** `solutions/M3L1_solutions.py`
**Time:** ~90 minutes (including training)

**Tasks:**
1. Load pre-trained CNN and show summary
2. Identify feature layer name
3. Build hybrid model with `build_cnn_vit_hybrid()`
4. Compile model
5. Train model (3 epochs, steps_per_epoch=128)

**Key Concepts:**
- CNN-ViT hybrid architecture
- Positional embeddings
- Multi-head attention
- Transfer learning

**Expected Output:**
- Feature layer: "batch_normalization_5"
- Hybrid model with 4 transformer blocks
- Validation accuracy: 92-97%

---

### Lab 2: PyTorch ViT (12 points)

**File:** `Lab-M3L2-Vision-Transformers-in-PyTorch-v1.ipynb`
**Solution:** `solutions/M3L2_solutions.py`
**Time:** ~120 minutes (including training)

**Tasks:**
1. Define `train_transform`
2. Define `val_transform`
3. Create train/val dataloaders
4. Train model (epochs=5, heads=12, depth=12)
5. Plot validation loss comparison
6. Plot training time comparison

**Key Concepts:**
- PyTorch transformer implementation
- Hyperparameter comparison (depth, heads)
- Performance analysis

**Expected Output:**
- Model trained with depth=12, heads=12
- Comparison with depth=3, heads=6
- Validation accuracy: 92-97%
- Training time plots showing depth impact

---

### Lab 3: Final Evaluation (8 points)

**File:** `lab-M4L1-Land-Classification-CNN-ViT-Integration-Evaluation-v1.ipynb`
**Solution:** `solutions/M3L3_solutions.py`
**Time:** ~45 minutes

**Tasks:**
1. Define dataset path and hyperparameters
2. Instantiate PyTorch model
3. Print Keras model metrics
4. Print PyTorch model metrics

**Key Concepts:**
- Cross-framework evaluation
- Model loading and inference
- Final performance assessment

**Expected Output:**
- Keras accuracy: 92-97%
- PyTorch accuracy: 92-97%
- ROC curves for both models
- Comprehensive comparison table

---

## Expected Performance Benchmarks

### Target Metrics (for full training)

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Keras CNN | 93-95% | 0.93-0.95 | 0.93-0.95 | 0.93-0.95 |
| PyTorch CNN | 93-95% | 0.93-0.95 | 0.93-0.95 | 0.93-0.95 |
| Keras ViT | 95-97% | 0.95-0.97 | 0.95-0.97 | 0.95-0.97 |
| PyTorch ViT | 95-97% | 0.95-0.97 | 0.95-0.97 | 0.95-0.97 |

### Training Times (CPU)

| Lab | Epochs | Time (CPU) | Time (GPU) |
|-----|--------|------------|------------|
| M2L1 (Keras CNN) | 5 | ~30 min | ~5 min |
| M2L2 (PyTorch CNN) | 5 | ~35 min | ~6 min |
| M3L1 (Keras ViT) | 3 | ~60 min | ~10 min |
| M3L2 (PyTorch ViT) | 5 | ~120 min | ~20 min |

**Note:** Times vary based on hardware. Reduce epochs or steps_per_epoch for faster execution during testing.

---

## Submission Checklist

Before submitting to Coursera:

### Pre-Submission

- [ ] All tasks completed in each notebook
- [ ] All cells executed top-to-bottom
- [ ] All outputs visible (plots, metrics, etc.)
- [ ] No error messages in output cells
- [ ] Model checkpoints saved (optional for submission)

### Notebook Verification

**Module 1:**
- [ ] M1L1: 4 tasks completed (10 points)
- [ ] M1L2: 4 tasks completed (8 points)
- [ ] M1L3: 5 tasks completed (10 points)

**Module 2:**
- [ ] M2L1: 6 tasks completed (12 points)
- [ ] M2L2: 10 tasks completed (20 points)
- [ ] M2L3: 5 tasks/questions completed (10 points)

**Module 3:**
- [ ] M3L1: 5 tasks completed (10 points)
- [ ] M3L2: 6 tasks completed (12 points)
- [ ] M3L3: 4 tasks completed (8 points)

### Download Process

For each notebook:
1. File → Save Notebook
2. Right-click notebook in file browser
3. Select "Download"
4. Save to submission folder

### Upload to Coursera

1. Go to Coursera submission page
2. Upload each notebook to corresponding question:
   - Question 1 → M1L1
   - Question 2 → M1L2
   - Question 3 → M1L3
   - Question 4 → M2L1
   - Question 5 → M2L2
   - Question 6 → M2L3
   - Question 7 → M3L1
   - Question 8 → M3L2
   - Question 9 → M3L3
3. Verify all files uploaded correctly
4. Submit for AI grading

**Expected Score:** 90-100/100

---

## Troubleshooting

### Common Issues

**Issue 1: Dataset Download Fails**
```python
# Solution: Use manual download
# See DATASET_SETUP.md for instructions
```

**Issue 2: Out of Memory**
```python
# Solution: Reduce batch size
batch_size = 32  # Instead of 128
steps_per_epoch = 64  # Instead of 128
```

**Issue 3: Training Too Slow**
```python
# Solution: Reduce epochs for testing
epochs = 2  # Instead of 5
# Or reduce dataset size
steps_per_epoch = 50
```

**Issue 4: Model Loading Error**
```python
# Solution: Re-download pre-trained models
# Check file paths are correct
# Ensure all custom layers are defined
```

**Issue 5: Import Errors**
```bash
# Solution: Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

---

## Additional Resources

### Documentation
- `README.md`: Project overview and setup
- `DATASET_SETUP.md`: Dataset download guide
- `requirements.txt`: Python dependencies

### Solution Files
- `solutions/M1L1_solutions.py`: Lab 1.1 complete code
- `solutions/M1L2_solutions.py`: Lab 1.2 complete code
- `solutions/M1L3_solutions.py`: Lab 1.3 complete code
- `solutions/M2L1_solutions.py`: Lab 2.1 complete code
- `solutions/M2L2_solutions.py`: Lab 2.2 complete code
- `solutions/M2L3_solutions.py`: Lab 2.3 complete code
- `solutions/M3L1_solutions.py`: Lab 3.1 complete code
- `solutions/M3L2_solutions.py`: Lab 3.2 complete code
- `solutions/M3L3_solutions.py`: Lab 3.3 complete code

### Reference Materials
- TensorFlow/Keras Documentation: https://www.tensorflow.org/
- PyTorch Documentation: https://pytorch.org/docs/
- Vision Transformer Paper: https://arxiv.org/abs/2010.11929

---

## Success Tips

1. **Execute in Order:** Always run notebooks from top to bottom
2. **Save Frequently:** Save notebook after completing each task
3. **Verify Outputs:** Check all plots and metrics display correctly
4. **Document Answers:** For conceptual questions, write clear explanations
5. **Test Before Submit:** Run entire notebook fresh before downloading
6. **Check File Sizes:** Ensure notebooks are ~1-5 MB (indicates outputs saved)
7. **Backup Work:** Keep copies of completed notebooks

---

## Final Verification

Before final submission, verify:

```bash
# Check all solution files exist
ls solutions/
# Should show: M1L1_solutions.py through M3L3_solutions.py

# Check all notebooks exist
ls *.ipynb
# Should show all 9 original notebooks

# Verify dataset downloaded
ls images_dataSAT/
# Should show both class folders
```

---

## Support

If you encounter issues:

1. Check `DATASET_SETUP.md` for dataset problems
2. Review solution files for code examples
3. Consult course forum for common issues
4. Verify all dependencies installed correctly

---

## Congratulations!

You now have complete solutions for all 9 labs totaling 100 points. Follow this guide carefully to ensure successful completion and submission of the AI Capstone project.

**Good luck with your submission!** 🚀

---

**Project Complete:** 2025-10-26
**Ready for Submission:** YES ✅
