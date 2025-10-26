# AI Capstone Project: Land Classification with Deep Learning

## Project Overview

This repository contains the complete implementation of a Coursera AI Capstone project focused on **satellite imagery land classification**. The project uses both **Keras (TensorFlow)** and **PyTorch** frameworks to build progressively sophisticated models:

1. **Traditional CNNs** for local feature extraction
2. **Vision Transformers (ViTs)** for global context modeling
3. **Hybrid CNN-ViT architectures** combining both approaches

**Use Case:** Classify satellite images into agricultural vs. non-agricultural land to help fertilizer companies identify expansion opportunities.

---

## Project Structure

```
coursera/capstone/
├── README.md                                          # This file
├── DATASET_SETUP.md                                   # Dataset download guide
├── requirements.txt                                   # Python dependencies
├── .gitignore                                         # Git ignore rules
│
├── AI-capstone-M1L1-v1.ipynb                         # Module 1 Lab 1 (10 pts)
├── AI-capstone-M1L2-v1.ipynb                         # Module 1 Lab 2 (8 pts)
├── AI-capstone-M1L3-v1.ipynb                         # Module 1 Lab 3 (10 pts)
│
├── Lab-M2L1-Train-and-Evaluate-a-Keras-Based-Classifier-v1.ipynb      # (12 pts)
├── Lab-M2L2-Implement-and-Test-a-PyTorch-Based-Classifier-v1.ipynb    # (20 pts)
├── Lab-M2L3-Comparative-Analysis-of-Keras-and-PyTorch-Models-v1.ipynb # (10 pts)
│
├── Lab-M3L1-Vision-Transformers-in-Keras-v1.ipynb                     # (10 pts)
├── Lab-M3L2-Vision-Transformers-in-PyTorch-v1.ipynb                   # (12 pts)
└── lab-M4L1-Land-Classification-CNN-ViT-Integration-Evaluation-v1.ipynb # (8 pts)
```

**Total Points:** 100 (Pass threshold: 70%)

---

## Learning Modules

### Module 1: Data Loading and Exploration (28 points)
- **Lab 1:** Memory-based vs Generator-based data loading
- **Lab 2:** Keras ImageDataGenerator and augmentation
- **Lab 3:** PyTorch DataLoader and transforms

**Key Skills:** Efficient data pipelines, augmentation strategies, memory management

### Module 2: CNN Classifiers (42 points)
- **Lab 4:** Keras CNN architecture, training, evaluation
- **Lab 5:** PyTorch CNN with custom training loops
- **Lab 6:** Comparative analysis of both frameworks

**Key Skills:** CNN design, training pipelines, performance metrics, framework comparison

### Module 3: Vision Transformers (30 points)
- **Lab 7:** Keras CNN-ViT hybrid model
- **Lab 8:** PyTorch CNN-ViT hybrid with hyperparameter tuning
- **Lab 9:** Final cross-framework evaluation

**Key Skills:** Transformer architectures, attention mechanisms, hybrid models, production deployment

---

## Quick Start

### 1. Environment Setup

```bash
# Clone the repository
cd ~/Documents/GitHub/Personal/coursera/capstone

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Dataset Setup

The dataset will be automatically downloaded when you run the notebooks. However, you can manually download it:

```bash
# See DATASET_SETUP.md for detailed instructions
# Or run the first cell of any notebook
```

**Dataset Details:**
- **Source:** IBM Cloud Storage
- **Size:** ~5000 images per class (10,000 total)
- **Classes:** Agricultural land (class_1_agri), Non-agricultural land (class_0_non_agri)
- **Format:** JPG images, 64x64 pixels (after preprocessing)

### 3. Running the Notebooks

Execute notebooks **in order** for best results:

```bash
# Start Jupyter
jupyter notebook

# Or use JupyterLab
jupyter lab
```

**Execution Order:**
1. Module 1: M1L1 → M1L2 → M1L3
2. Module 2: M2L1 → M2L2 → M2L3
3. Module 3: M3L1 → M3L2 → M3L3 (M4L1)

---

## Technical Details

### Model Architectures

#### CNN (Baseline)
```
Input (64x64x3)
├── Conv2D(32) → ReLU → MaxPool → BatchNorm
├── Conv2D(64) → ReLU → MaxPool → BatchNorm
├── Conv2D(128) → ReLU → MaxPool → BatchNorm
├── Conv2D(256) → ReLU → MaxPool → BatchNorm
├── Conv2D(512) → ReLU → MaxPool → BatchNorm
├── Conv2D(1024) → ReLU → MaxPool → BatchNorm
├── GlobalAvgPool
└── Dense(2048) → Dense(num_classes)
```

**Performance:** ~95% accuracy

#### CNN-ViT Hybrid
```
Input (64x64x3)
├── CNN Backbone (feature extraction)
│   └── Output: (B, 1024, H', W')
├── Patch Embedding (1x1 conv)
│   └── Output: (B, num_patches, 768)
├── Positional Encoding
├── Transformer Blocks (depth: 3-12)
│   ├── Multi-Head Self-Attention (heads: 6-12)
│   ├── LayerNorm
│   ├── MLP (expand 4x)
│   └── Residual connections
└── Classification Head
    └── Output: (B, num_classes)
```

**Performance:** ~97% accuracy (with proper hyperparameters)

### Training Configuration

| Parameter | CNN | ViT |
|-----------|-----|-----|
| Optimizer | Adam | Adam |
| Learning Rate | 0.001 | 0.0001 |
| Batch Size | 128 | 32-128 |
| Epochs | 15-20 | 15-20 |
| Loss | Binary Crossentropy | Categorical Crossentropy |
| Data Aug | Rotation, Flip, Shear | Same |

### Evaluation Metrics

- **Accuracy:** Overall correct predictions
- **Precision:** True positives / (True positives + False positives)
- **Recall:** True positives / (True positives + False negatives)
- **F1-Score:** Harmonic mean of precision and recall
- **ROC-AUC:** Area under ROC curve
- **Confusion Matrix:** Detailed classification breakdown

---

## File Descriptions

### Notebooks

| Notebook | Description | Points | Key Tasks |
|----------|-------------|--------|-----------|
| M1L1 | Memory vs Generator Loading | 10 | Image shape, display, path lists, counting |
| M1L2 | Keras Data Loading | 8 | Combine paths, temp list, batch generation |
| M1L3 | PyTorch Data Loading | 10 | Custom transforms, ImageFolder, DataLoader |
| M2L1 | Keras Classifier | 12 | Build CNN, train, evaluate, plot |
| M2L2 | PyTorch Classifier | 20 | Training loop, metrics, explanations |
| M2L3 | Comparative Analysis | 10 | Compare frameworks, metrics, confusion matrices |
| M3L1 | Keras ViT | 10 | Load CNN, build hybrid, train |
| M3L2 | PyTorch ViT | 12 | Implement ViT, train, compare depths |
| M3L3 | Final Evaluation | 8 | Cross-framework evaluation, final metrics |

### Configuration Files

- **requirements.txt:** All Python package dependencies
- **.gitignore:** Excludes datasets, models, and temporary files
- **DATASET_SETUP.md:** Detailed dataset download instructions

---

## Model Checkpoints

Pre-trained models are automatically downloaded during notebook execution:

| Model | File | Size | Accuracy |
|-------|------|------|----------|
| Keras CNN | ai-capstone-keras-best-model-model.keras | ~50 MB | ~95% |
| PyTorch CNN | ai-capstone-pytorch-best-model.pth | ~45 MB | ~95% |
| Keras ViT | keras-cnn-vit-ai-capstone.keras | ~80 MB | ~97% |
| PyTorch ViT | pytorch-cnn-vit-ai-capstone-model-state-dict.pth | ~75 MB | ~97% |

---

## Submission Instructions

### For Coursera AI Grading

1. **Complete all tasks** in each notebook
2. **Execute all cells** from top to bottom
3. **Verify outputs** are displayed correctly
4. **Save notebooks** (File → Save)
5. **Download notebooks:**
   - Right-click notebook in file browser
   - Select "Download"
6. **Upload to Coursera** submission page

**Important:**
- Submit all 9 notebooks
- Ensure all cell outputs are visible
- Check that all tasks are marked complete

---

## Troubleshooting

### Common Issues

**1. Dataset Download Fails**
```python
# Solution: Use manual download in DATASET_SETUP.md
# Or check internet connection and retry
```

**2. CUDA Out of Memory**
```python
# Solution: Reduce batch_size
batch_size = 32  # Instead of 128
```

**3. Model Loading Errors**
```python
# Solution: Re-download model checkpoint
# Check file path is correct
```

**4. Import Errors**
```bash
# Solution: Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

---

## Performance Benchmarks

### Expected Runtimes (CPU)

| Lab | Training Time | Inference Time |
|-----|---------------|----------------|
| M2L1 (Keras CNN) | ~30 min (5 epochs) | ~2 min |
| M2L2 (PyTorch CNN) | ~35 min (5 epochs) | ~3 min |
| M3L1 (Keras ViT) | ~60 min (5 epochs) | ~5 min |
| M3L2 (PyTorch ViT) | ~75 min (5 epochs) | ~6 min |

**Note:** GPU training is ~10x faster

---

## Additional Resources

### Recommended Reading
- [Vision Transformer Paper (Dosovitskiy et al., 2020)](https://arxiv.org/abs/2010.11929)
- [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929)
- [TensorFlow/Keras Documentation](https://www.tensorflow.org/)
- [PyTorch Documentation](https://pytorch.org/docs/)

### Related Projects
- Image classification with transfer learning
- Semantic segmentation for satellite imagery
- Object detection in aerial images

---

## License

This project is created for educational purposes as part of the Coursera AI Capstone course.

---

## Author

**Student Submission for Coursera AI Capstone**
Land Classification using CNNs and Vision Transformers

---

## Acknowledgments

- IBM Skills Network for dataset and lab infrastructure
- TensorFlow and PyTorch teams for excellent frameworks
- Coursera for the comprehensive AI specialization

---

**Last Updated:** 2025-10-26
