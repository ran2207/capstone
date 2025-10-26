"""
Module 1 Lab 2: Data Loading and Augmentation Using Keras
Solutions for all tasks (8 points total)

Copy these code blocks into the corresponding cells in the notebook:
AI-capstone-M1L2-v1.ipynb
"""

import os
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# =============================================================================
# TASK 1: Create all_image_paths list containing paths from both folders
# =============================================================================

# Define directories
dir_non_agri = './images_dataSAT/class_0_non_agri/'
dir_agri = './images_dataSAT/class_1_agri/'

# Get all image paths from both directories
non_agri_paths = [os.path.join(dir_non_agri, f) for f in os.listdir(dir_non_agri)
                  if f.endswith(('.jpg', '.jpeg', '.png'))]
agri_paths = [os.path.join(dir_agri, f) for f in os.listdir(dir_agri)
              if f.endswith(('.jpg', '.jpeg', '.png'))]

# Combine both lists
all_image_paths = non_agri_paths + agri_paths

print(f"Total non-agricultural images: {len(non_agri_paths)}")
print(f"Total agricultural images: {len(agri_paths)}")
print(f"Total images in all_image_paths: {len(all_image_paths)}")


# =============================================================================
# TASK 2: Create temporary list by binding image paths and labels, print 5 random samples
# =============================================================================

# Create labels: 0 for non-agricultural, 1 for agricultural
labels = [0] * len(non_agri_paths) + [1] * len(agri_paths)

# Combine paths and labels using zip
temp = list(zip(all_image_paths, labels))

# Shuffle the temporary list
random.shuffle(temp)

# Print 5 random samples
print("\n5 Random samples from temp list:")
print("="*80)
samples = random.sample(temp, 5)
for idx, (path, label) in enumerate(samples, 1):
    print(f"{idx}. Path: {path}")
    print(f"   Label: {label} ({'Agricultural' if label == 1 else 'Non-Agricultural'})")
    print()


# =============================================================================
# TASK 3: Generate a data batch (batch size = 8) using custom_data_generator
# =============================================================================

def custom_data_generator(image_paths, labels, batch_size=8, img_size=(64, 64)):
    """
    Custom generator that yields batches of images and labels
    """
    num_samples = len(image_paths)
    indices = list(range(num_samples))

    while True:
        # Shuffle indices for each epoch
        random.shuffle(indices)

        for start_idx in range(0, num_samples, batch_size):
            batch_indices = indices[start_idx:start_idx + batch_size]

            # Initialize batch arrays
            batch_images = []
            batch_labels = []

            for idx in batch_indices:
                # Load and preprocess image
                img = Image.open(image_paths[idx])
                img = img.resize(img_size)
                img_array = np.array(img) / 255.0  # Normalize to [0, 1]

                batch_images.append(img_array)
                batch_labels.append(labels[idx])

            yield np.array(batch_images), np.array(batch_labels)


# Generate a sample batch
print("\nGenerating sample batch with batch_size=8...")
gen = custom_data_generator(all_image_paths, labels, batch_size=8)
batch_images, batch_labels = next(gen)

print(f"Batch images shape: {batch_images.shape}")
print(f"Batch labels shape: {batch_labels.shape}")
print(f"Batch labels: {batch_labels}")

# Visualize the batch
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()
for idx in range(8):
    axes[idx].imshow(batch_images[idx])
    axes[idx].set_title(f"Label: {int(batch_labels[idx])} ({'Agri' if batch_labels[idx] == 1 else 'Non-Agri'})")
    axes[idx].axis('off')
plt.tight_layout()
plt.show()


# =============================================================================
# TASK 4: Create validation data using ImageDataGenerator with batch size of 8
# =============================================================================

# Define image data generator with augmentation and validation split
datagen = ImageDataGenerator(
    rescale=1./255,                  # Normalize pixel values
    rotation_range=40,               # Random rotation
    width_shift_range=0.2,          # Horizontal shift
    height_shift_range=0.2,         # Vertical shift
    shear_range=0.2,                # Shear transformation
    zoom_range=0.2,                 # Random zoom
    horizontal_flip=True,            # Random horizontal flip
    fill_mode='nearest',             # Fill strategy for transformations
    validation_split=0.2             # 20% for validation
)

# Create training data generator
train_generator = datagen.flow_from_directory(
    './images_dataSAT',              # Root directory
    target_size=(64, 64),           # Resize images
    batch_size=8,
    class_mode='binary',             # Binary classification (0 or 1)
    subset='training',               # Training subset
    shuffle=True
)

# Create validation data generator
validation_generator = datagen.flow_from_directory(
    './images_dataSAT',              # Same root directory
    target_size=(64, 64),
    batch_size=8,
    class_mode='binary',
    subset='validation',             # Validation subset
    shuffle=True
)

print("\nTraining generator:")
print(f"  Samples: {train_generator.samples}")
print(f"  Batch size: {train_generator.batch_size}")
print(f"  Classes: {train_generator.class_indices}")

print("\nValidation generator:")
print(f"  Samples: {validation_generator.samples}")
print(f"  Batch size: {validation_generator.batch_size}")
print(f"  Classes: {validation_generator.class_indices}")

# Get a sample batch from validation generator
val_batch_images, val_batch_labels = next(validation_generator)
print(f"\nValidation batch images shape: {val_batch_images.shape}")
print(f"Validation batch labels shape: {val_batch_labels.shape}")


# =============================================================================
# BONUS: Visualize validation batch with augmentation
# =============================================================================

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()
for idx in range(min(8, len(val_batch_images))):
    axes[idx].imshow(val_batch_images[idx])
    axes[idx].set_title(f"Label: {int(val_batch_labels[idx])}")
    axes[idx].axis('off')
plt.suptitle("Validation Batch with Augmentation")
plt.tight_layout()
plt.show()


# =============================================================================
# SUMMARY OF SOLUTIONS
# =============================================================================

print("\n" + "="*60)
print("SUMMARY: Module 1 Lab 2 - All Tasks Completed")
print("="*60)
print(f"Task 1: Created all_image_paths with {len(all_image_paths)} images")
print(f"Task 2: Created temp list with zip and printed 5 samples")
print(f"Task 3: Generated batch of {batch_images.shape[0]} images")
print(f"Task 4: Created validation generator with {validation_generator.samples} samples")
print("="*60)
