"""
Module 2 Lab 2: Implement and Test a PyTorch-Based Classifier
Solutions for all tasks (20 points total)

Copy these code blocks into the corresponding cells in the notebook:
Lab-M2L2-Implement-and-Test-a-PyTorch-Based-Classifier-v1.ipynb
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# =============================================================================
# TASK 1: Explain the usefulness of random initialization
# =============================================================================

print("="*70)
print("TASK 1: Usefulness of Random Initialization")
print("="*70)
print("""
Random initialization is crucial in neural networks for several reasons:

1. **Breaking Symmetry**: If all weights are initialized to the same value
   (e.g., all zeros), all neurons in a layer would compute the same output and
   receive the same gradients during backpropagation. This means they would all
   update identically, making multiple neurons redundant.

2. **Enabling Learning**: Random initialization creates diverse initial states
   for neurons, allowing them to learn different features from the data.

3. **Gradient Flow**: Proper random initialization (like He initialization for
   ReLU activations) helps maintain good gradient flow during backpropagation,
   preventing vanishing or exploding gradients.

4. **Convergence**: Well-initialized networks converge faster and reach better
   local minima compared to poorly initialized networks.

**Common Initialization Strategies:**
- Xavier/Glorot: For sigmoid/tanh activations
- He initialization: For ReLU activations (used in our CNN)
- Uniform or Normal distributions with controlled variance

In our CNN, we use He uniform initialization which is designed to work well
with ReLU activations by maintaining variance across layers.
""")


# =============================================================================
# TASK 2: Define train_transform pipeline with augmentation
# =============================================================================

img_size = 64

train_transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.RandomRotation(40),                    # Random rotation ±40 degrees
    transforms.RandomHorizontalFlip(p=0.5),          # 50% chance horizontal flip
    transforms.RandomVerticalFlip(p=0.2),            # 20% chance vertical flip
    transforms.RandomAffine(0, shear=0.2),          # Shear transformation
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Color augmentation
    transforms.ToTensor(),                            # Convert to tensor
    transforms.Normalize(                             # Normalize with ImageNet stats
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

print("\nTASK 2: Training Transform Pipeline Created")
print("  - Resize to 64x64")
print("  - RandomRotation(40)")
print("  - RandomHorizontalFlip(0.5)")
print("  - RandomVerticalFlip(0.2)")
print("  - RandomAffine with shear(0.2)")
print("  - ColorJitter")
print("  - ToTensor")
print("  - Normalize (ImageNet statistics)")


# =============================================================================
# TASK 3: Define val_transform pipeline (no augmentation)
# =============================================================================

val_transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

print("\nTASK 3: Validation Transform Pipeline Created")
print("  - Resize to 64x64")
print("  - ToTensor")
print("  - Normalize (same as training)")


# =============================================================================
# LOAD DATASET AND CREATE TRAIN/VAL SPLIT
# =============================================================================

dataset_path = './images_dataSAT'

# Load full dataset
full_dataset = datasets.ImageFolder(root=dataset_path, transform=train_transform)

# Split into train (80%) and validation (20%)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size

train_dataset, val_dataset = random_split(
    full_dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

# Apply validation transform to val dataset
val_dataset.dataset.transform = val_transform

print(f"\nDataset Split:")
print(f"  Total samples: {len(full_dataset)}")
print(f"  Training samples: {len(train_dataset)}")
print(f"  Validation samples: {len(val_dataset)}")


# =============================================================================
# TASK 4: Create val_loader for the validation dataset
# =============================================================================

batch_size = 128

# Create training loader
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0
)

# Create validation loader (TASK 4 ANSWER)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,  # No shuffling for validation
    num_workers=0
)

print(f"\nTASK 4: DataLoaders Created")
print(f"  Train batches: {len(train_loader)}")
print(f"  Validation batches: {len(val_loader)}")
print(f"  Batch size: {batch_size}")


# =============================================================================
# TASK 5: Purpose of tqdm
# =============================================================================

print("\n" + "="*70)
print("TASK 5: Purpose of tqdm")
print("="*70)
print("""
tqdm (name derived from Arabic "taqaddum" meaning "progress") is a Python
library that provides fast, extensible progress bars for loops and iterations.

**Purpose in Machine Learning Training:**

1. **Visual Feedback**: Shows real-time progress of training/validation loops,
   making it easier to monitor long-running processes.

2. **Time Estimation**: Provides ETA (Estimated Time of Arrival) and elapsed
   time, helping you plan workflow and identify bottlenecks.

3. **Iteration Rate**: Shows iterations per second, useful for performance
   monitoring and hardware utilization analysis.

4. **User Experience**: Makes CLIs more user-friendly by providing interactive
   feedback instead of blank terminals.

**Example Usage:**
   for batch in tqdm(train_loader, desc="Training"):
       # Training code here
       pass

**Output Example:**
   Training: 100%|████████████| 45/45 [01:23<00:00,  1.85s/it]

This shows: 100% complete, 45/45 batches, 1:23 elapsed, 1.85 seconds per batch.
""")


# =============================================================================
# DEFINE CNN MODEL
# =============================================================================

class ConvNet(nn.Module):
    def __init__(self, num_classes=2):
        super(ConvNet, self).__init__()

        # Convolutional feature extractor
        self.features = nn.Sequential(
            # Block 1: 3 -> 32
            nn.Conv2d(3, 32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(32),

            # Block 2: 32 -> 64
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(64),

            # Block 3: 64 -> 128
            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(128),

            # Block 4: 128 -> 256
            nn.Conv2d(128, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(256),

            # Block 5: 256 -> 512
            nn.Conv2d(256, 512, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(512),

            # Block 6: 512 -> 1024
            nn.Conv2d(512, 1024, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(1024),
        )

        # Global pooling and classifier
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024, 2048),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(2048),
            nn.Dropout(0.4),
            nn.Linear(2048, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x


# Initialize model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ConvNet(num_classes=2).to(device)

print(f"\nModel initialized on device: {device}")
print(f"\nModel Architecture:")
print(model)


# =============================================================================
# TASK 6: Explain why train_loss, train_correct, train_total are reset every epoch
# =============================================================================

print("\n" + "="*70)
print("TASK 6: Why Reset Metrics Every Epoch")
print("="*70)
print("""
The variables train_loss, train_correct, and train_total are reset at the
beginning of each epoch for several important reasons:

1. **Per-Epoch Statistics**: We want to measure the model's performance
   specifically for each epoch, not accumulate across all epochs.

2. **Tracking Progress**: Resetting allows us to see how the model improves
   from epoch to epoch. If we didn't reset:
   - Loss would keep accumulating → meaningless large numbers
   - Accuracy would be averaged across all training history
   - We couldn't detect if training has plateaued or degraded

3. **Meaningful Comparisons**: Each epoch should be evaluated independently
   so we can compare:
   - Is epoch 5 better than epoch 4?
   - Has the model stopped improving?
   - Should we stop training (early stopping)?

4. **Debugging and Monitoring**: Epoch-wise metrics help identify:
   - Overfitting (train accuracy high, val accuracy low)
   - Underfitting (both accuracies plateau at low values)
   - Training instability (metrics fluctuating wildly)

**Example:**
   Epoch 1: Loss=0.5, Accuracy=80%
   Epoch 2: Loss=0.3, Accuracy=88%  ← Shows clear improvement

Without resetting, these individual epoch insights would be lost.
""")


# =============================================================================
# TRAINING FUNCTION WITH TQDM
# =============================================================================

def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()

    running_loss = 0.0
    running_corrects = 0
    total_samples = 0

    for images, labels in tqdm(loader, desc="Training"):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == labels).item()
        total_samples += images.size(0)

    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects / total_samples

    return epoch_loss, epoch_acc


# =============================================================================
# TASK 7: Why use torch.no_grad() in the validation loop?
# =============================================================================

print("\n" + "="*70)
print("TASK 7: Why Use torch.no_grad() in Validation Loop")
print("="*70)
print("""
torch.no_grad() is a context manager that disables gradient computation.
It should be used during validation/testing for several critical reasons:

1. **Memory Efficiency**:
   - Gradients require significant memory to store intermediate activations
   - Without gradients, memory usage can be reduced by 50% or more
   - Allows larger batch sizes during inference

2. **Computational Speed**:
   - Skipping gradient computation makes forward pass faster
   - No need to track operations for backpropagation
   - Validation runs 2-3x faster

3. **Preventing Accidental Updates**:
   - Ensures model weights cannot be accidentally modified during validation
   - Adds a safety layer against programming errors

4. **Correct Model Behavior**:
   - Some layers (Dropout, BatchNorm) behave differently in eval mode
   - Combined with model.eval(), ensures proper inference behavior

**Example:**
   with torch.no_grad():
       outputs = model(images)
       # No gradients computed or stored

vs.

   outputs = model(images)
   # Gradients computed and stored (unnecessary overhead)

**Memory Comparison:**
   - With gradients: ~4GB VRAM
   - Without gradients: ~2GB VRAM
   → Allows 2x larger validation batches
""")


# =============================================================================
# VALIDATION FUNCTION
# =============================================================================

def validate_epoch(model, loader, criterion, device):
    """Validate for one epoch"""
    model.eval()

    running_loss = 0.0
    running_corrects = 0
    total_samples = 0

    with torch.no_grad():  # TASK 7: No gradient computation
        for images, labels in tqdm(loader, desc="Validation"):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass only
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Statistics
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels).item()
            total_samples += images.size(0)

    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects / total_samples

    return epoch_loss, epoch_acc


# =============================================================================
# TASK 8: List two metrics used to evaluate training performance
# =============================================================================

print("\n" + "="*70)
print("TASK 8: Two Metrics Used to Evaluate Training Performance")
print("="*70)
print("""
Two essential metrics for evaluating training performance:

1. **LOSS (Cross-Entropy Loss)**:
   - Measures how well the model's predictions match the true labels
   - Lower loss = better predictions
   - Training loss should decrease over epochs
   - Validation loss helps detect overfitting
   - Formula for cross-entropy: L = -Σ y_true * log(y_pred)

2. **ACCURACY**:
   - Percentage of correct predictions
   - Range: 0% to 100%
   - Easy to interpret and communicate
   - Training accuracy shows if model is learning
   - Validation accuracy shows if model generalizes
   - Formula: Accuracy = (Correct Predictions) / (Total Predictions)

**Additional Important Metrics:**
- Precision: True Positives / (True Positives + False Positives)
- Recall: True Positives / (True Positives + False Negatives)
- F1-Score: Harmonic mean of precision and recall
- ROC-AUC: Area under ROC curve

**Why These Two Are Primary:**
- Loss guides the optimization (gradient descent minimizes loss)
- Accuracy provides interpretable performance measure
- Together they give complete picture of model performance
""")


# =============================================================================
# TRAINING LOOP
# =============================================================================

# Training configuration
num_epochs = 5
learning_rate = 0.001

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Storage for metrics
train_losses = []
train_accs = []
val_losses = []
val_accs = []

print(f"\nStarting training for {num_epochs} epochs...")
print(f"Device: {device}")
print(f"Learning rate: {learning_rate}")
print(f"Batch size: {batch_size}\n")

best_val_acc = 0.0
for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")
    print("-" * 50)

    start_time = time.time()

    # Train
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    train_losses.append(train_loss)
    train_accs.append(train_acc)

    # Validate
    val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
    val_losses.append(val_loss)
    val_accs.append(val_acc)

    epoch_time = time.time() - start_time

    # Print metrics
    print(f"\nResults:")
    print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
    print(f"  Time: {epoch_time:.2f}s")

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'pytorch_cnn_best_model.pth')
        print(f"  *** New best model saved! Val Acc: {val_acc:.4f} ***")

print("\nTraining completed!")
print(f"Best validation accuracy: {best_val_acc:.4f}")


# =============================================================================
# TASK 9: Plot model training loss
# =============================================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot Loss (TASK 9 ANSWER)
axes[0].plot(range(1, num_epochs+1), train_losses, label='Training Loss', marker='o', color='coral')
axes[0].plot(range(1, num_epochs+1), val_losses, label='Validation Loss', marker='s', color='dodgerblue')
axes[0].set_title('Model Loss Over Epochs', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot Accuracy
axes[1].plot(range(1, num_epochs+1), train_accs, label='Training Accuracy', marker='o')
axes[1].plot(range(1, num_epochs+1), val_accs, label='Validation Accuracy', marker='s')
axes[1].set_title('Model Accuracy Over Epochs', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('pytorch_cnn_training_history.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nTraining plots saved as 'pytorch_cnn_training_history.png'")


# =============================================================================
# TASK 10: Retrieve predictions all_preds and ground truth all_labels from val_loader
# =============================================================================

# Collect all predictions and labels
all_preds = []
all_labels = []

model.eval()
with torch.no_grad():
    for images, labels in tqdm(val_loader, desc="Collecting predictions"):
        images = images.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

# Convert to numpy arrays
all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

print(f"\nTASK 10: Predictions Collected")
print(f"  Total samples: {len(all_preds)}")
print(f"  Predictions shape: {all_preds.shape}")
print(f"  Labels shape: {all_labels.shape}")

# Compute detailed metrics
accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average='binary')
recall = recall_score(all_labels, all_preds, average='binary')
f1 = f1_score(all_labels, all_preds, average='binary')

print(f"\nDetailed Metrics:")
print(f"  Accuracy:  {accuracy:.4f}")
print(f"  Precision: {precision:.4f}")
print(f"  Recall:    {recall:.4f}")
print(f"  F1-Score:  {f1:.4f}")


# =============================================================================
# BONUS: Confusion Matrix
# =============================================================================

from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(all_labels, all_preds)
class_names = ['Non-Agricultural', 'Agricultural']

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix - PyTorch CNN', fontsize=14, fontweight='bold')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('pytorch_cnn_confusion_matrix.png', dpi=300)
plt.show()


# =============================================================================
# SUMMARY OF SOLUTIONS
# =============================================================================

print("\n" + "="*70)
print("SUMMARY: Module 2 Lab 2 - All Tasks Completed")
print("="*70)
print(f"Task 1: Explained random initialization importance")
print(f"Task 2: Created train_transform with augmentation")
print(f"Task 3: Created val_transform without augmentation")
print(f"Task 4: Created val_loader with {len(val_loader)} batches")
print(f"Task 5: Explained purpose of tqdm for progress tracking")
print(f"Task 6: Explained why metrics are reset each epoch")
print(f"Task 7: Explained torch.no_grad() for memory efficiency")
print(f"Task 8: Listed Loss and Accuracy as primary metrics")
print(f"Task 9: Plotted training loss curves")
print(f"Task 10: Collected {len(all_preds)} predictions and labels")
print(f"\nFinal Validation Accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
print("="*70)
