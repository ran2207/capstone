"""
Module 1 Lab 3: Data Loading and Augmentation Using PyTorch
Solutions for all tasks (10 points total)

Copy these code blocks into the corresponding cells in the notebook:
AI-capstone-M1L3-v1.ipynb
"""

import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# =============================================================================
# TASK 1: Define transformation pipeline custom_transform
# =============================================================================

custom_transform = transforms.Compose([
    transforms.Resize((64, 64)),                    # Resize to 64x64 pixels
    transforms.RandomHorizontalFlip(p=0.5),         # Random horizontal flip with 50% probability
    transforms.RandomVerticalFlip(p=0.2),           # Random vertical flip with 20% probability
    transforms.RandomRotation(45),                  # Random rotation up to 45 degrees
    transforms.ToTensor(),                          # Convert PIL image to tensor
    transforms.Normalize(                           # Normalize using ImageNet stats
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

print("Custom transform pipeline created with:")
print("  - Resize to 64x64")
print("  - RandomHorizontalFlip (p=0.5)")
print("  - RandomVerticalFlip (p=0.2)")
print("  - RandomRotation (45 degrees)")
print("  - ToTensor")
print("  - Normalize (ImageNet stats)")


# =============================================================================
# TASK 2: Load dataset using datasets.ImageFolder with custom_transform
# =============================================================================

# Define dataset path
dataset_path = './images_dataSAT'

# Load dataset using ImageFolder
imagefolder_dataset = datasets.ImageFolder(
    root=dataset_path,
    transform=custom_transform
)

print(f"\nDataset loaded from: {dataset_path}")
print(f"Total images in dataset: {len(imagefolder_dataset)}")


# =============================================================================
# TASK 3: Print class names and indices from imagefolder_dataset
# =============================================================================

# Get class names and their corresponding indices
class_names = imagefolder_dataset.classes
class_to_idx = imagefolder_dataset.class_to_idx

print("\nClass names:")
for idx, class_name in enumerate(class_names):
    print(f"  {idx}: {class_name}")

print("\nClass to index mapping:")
for class_name, idx in class_to_idx.items():
    print(f"  '{class_name}' -> {idx}")


# =============================================================================
# TASK 4: Retrieve and display image shapes from a batch in imagefolder_loader
# =============================================================================

# Create DataLoader with batch size 8
imagefolder_loader = DataLoader(
    imagefolder_dataset,
    batch_size=8,
    shuffle=True,
    num_workers=0  # Set to 0 for compatibility, increase for performance
)

# Get one batch
batch_images, batch_labels = next(iter(imagefolder_loader))

print(f"\nBatch information:")
print(f"  Batch images shape: {batch_images.shape}")
print(f"  Batch labels shape: {batch_labels.shape}")
print(f"  Batch labels: {batch_labels.tolist()}")

# Print individual image shapes
print(f"\nIndividual image shapes:")
for idx in range(batch_images.shape[0]):
    print(f"  Image {idx + 1}: {batch_images[idx].shape}")


# =============================================================================
# TASK 5: Display images in the custom loader batch
# =============================================================================

def denormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Denormalize tensor for visualization
    """
    tensor = tensor.clone()
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

def imshow(img, title=None):
    """
    Display tensor as image
    """
    img = denormalize(img)
    img = img.numpy().transpose((1, 2, 0))
    img = np.clip(img, 0, 1)
    plt.imshow(img)
    if title:
        plt.title(title)
    plt.axis('off')

# Display the batch
print("\nDisplaying batch of images...")
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

for idx in range(8):
    img = batch_images[idx]
    label = batch_labels[idx].item()
    class_name = class_names[label]

    plt.subplot(2, 4, idx + 1)
    imshow(img, title=f"{class_name} (Label: {label})")

plt.suptitle("Batch from PyTorch DataLoader with Custom Transforms")
plt.tight_layout()
plt.show()


# =============================================================================
# ADDITIONAL: Create separate train/val dataloaders (common pattern)
# =============================================================================

# Define transforms without augmentation for validation
val_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Load full dataset
full_dataset = datasets.ImageFolder(root=dataset_path, transform=custom_transform)

# Split into train and validation (80/20)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size

train_dataset, val_dataset = torch.utils.data.random_split(
    full_dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)  # For reproducibility
)

# Update validation dataset transform
val_dataset.dataset.transform = val_transform

# Create DataLoaders
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0
)

print(f"\nTrain/Val split:")
print(f"  Training samples: {len(train_dataset)}")
print(f"  Validation samples: {len(val_dataset)}")
print(f"  Train batches: {len(train_loader)}")
print(f"  Val batches: {len(val_loader)}")


# =============================================================================
# BONUS: Show effect of augmentation
# =============================================================================

# Get a single image from dataset
img, label = imagefolder_dataset[0]

# Create transform without augmentation for comparison
no_aug_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Load same image without augmentation
dataset_no_aug = datasets.ImageFolder(
    root=dataset_path,
    transform=no_aug_transform
)

# Show original vs augmented
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Original (no augmentation)
img_orig, _ = dataset_no_aug[0]
axes[0].imshow(denormalize(img_orig).numpy().transpose(1, 2, 0))
axes[0].set_title("Original (No Augmentation)")
axes[0].axis('off')

# With augmentation
axes[1].imshow(denormalize(img).numpy().transpose(1, 2, 0))
axes[1].set_title("With Random Augmentation")
axes[1].axis('off')

plt.suptitle("Effect of Data Augmentation")
plt.tight_layout()
plt.show()


# =============================================================================
# SUMMARY OF SOLUTIONS
# =============================================================================

print("\n" + "="*60)
print("SUMMARY: Module 1 Lab 3 - All Tasks Completed")
print("="*60)
print(f"Task 1: Created custom_transform with augmentation")
print(f"Task 2: Loaded dataset with {len(imagefolder_dataset)} images")
print(f"Task 3: Class names: {class_names}")
print(f"        Class indices: {class_to_idx}")
print(f"Task 4: Batch shape: {batch_images.shape}")
print(f"Task 5: Displayed {batch_images.shape[0]} images from batch")
print("="*60)
