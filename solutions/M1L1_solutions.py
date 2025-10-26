"""
Module 1 Lab 1: Memory-Based vs Generator-Based Data Loading
Solutions for all tasks (10 points total)

Copy these code blocks into the corresponding cells in the notebook:
AI-capstone-M1L1-v1.ipynb
"""

# =============================================================================
# TASK 1: Determine the shape of a single image (Question 1)
# =============================================================================

import numpy as np
from PIL import Image
import os

# Load a single image to check its shape
image_path = os.path.join('./images_dataSAT/class_0_non_agri', os.listdir('./images_dataSAT/class_0_non_agri')[0])
image_data = np.array(Image.open(image_path))

# Display the shape
print(f"Shape of a single image: {image_data.shape}")
# Expected output: (height, width, channels) e.g., (256, 256, 3)


# =============================================================================
# TASK 2: Display the first four images in class_0_non_agri directory (Question 2)
# =============================================================================

import matplotlib.pyplot as plt

# Directory path
dir_non_agri = './images_dataSAT/class_0_non_agri/'

# Get list of image files
image_files = sorted([f for f in os.listdir(dir_non_agri) if f.endswith(('.jpg', '.jpeg', '.png'))])[:4]

# Display the first 4 images
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for idx, img_file in enumerate(image_files):
    img_path = os.path.join(dir_non_agri, img_file)
    img = Image.open(img_path)
    axes[idx].imshow(img)
    axes[idx].set_title(f"Image {idx + 1}")
    axes[idx].axis('off')
plt.tight_layout()
plt.show()


# =============================================================================
# TASK 3: Create agri_images_paths list with sorted full paths (Question 3)
# =============================================================================

# Directory for agricultural images
dir_agri = './images_dataSAT/class_1_agri/'

# Create list of full file paths for all agricultural images, sorted
agri_images_paths = sorted([os.path.join(dir_agri, f) for f in os.listdir(dir_agri)
                            if f.endswith(('.jpg', '.jpeg', '.png'))])

# Print first few paths to verify
print(f"Total agricultural image paths: {len(agri_images_paths)}")
print(f"First 5 paths:\n{agri_images_paths[:5]}")


# =============================================================================
# TASK 4: Determine the number of agricultural land images (Question 4)
# =============================================================================

# Count the number of agricultural images
num_agri_images = len(agri_images_paths)

print(f"\nNumber of images of agricultural land: {num_agri_images}")
# Expected output: ~5000 images


# =============================================================================
# ADDITIONAL CODE: Complete memory-based loading example
# =============================================================================

# Memory-based loading (load all images into memory)
def load_images_memory(image_paths, max_images=100):
    """
    Load images into memory as numpy arrays
    Use max_images to limit memory usage for demonstration
    """
    images = []
    labels = []

    for img_path in image_paths[:max_images]:
        img = Image.open(img_path)
        img = img.resize((64, 64))  # Resize for consistency
        img_array = np.array(img) / 255.0  # Normalize to [0, 1]
        images.append(img_array)
        labels.append(1)  # Label 1 for agricultural land

    return np.array(images), np.array(labels)

# Example usage
print("\nLoading sample images into memory...")
sample_images, sample_labels = load_images_memory(agri_images_paths, max_images=50)
print(f"Loaded {len(sample_images)} images into memory")
print(f"Shape of image batch: {sample_images.shape}")
print(f"Memory usage (approx): {sample_images.nbytes / (1024**2):.2f} MB")


# =============================================================================
# Generator-based loading example (for comparison)
# =============================================================================

def image_generator(image_paths, batch_size=32):
    """
    Generator that yields batches of images without loading all into memory
    """
    num_samples = len(image_paths)

    while True:  # Infinite loop for continuous batching
        for offset in range(0, num_samples, batch_size):
            batch_paths = image_paths[offset:offset + batch_size]

            batch_images = []
            batch_labels = []

            for img_path in batch_paths:
                img = Image.open(img_path)
                img = img.resize((64, 64))
                img_array = np.array(img) / 255.0
                batch_images.append(img_array)
                batch_labels.append(1)

            yield np.array(batch_images), np.array(batch_labels)

# Example usage
print("\n\nTesting generator-based loading...")
gen = image_generator(agri_images_paths, batch_size=8)
batch_imgs, batch_lbls = next(gen)
print(f"Generated batch shape: {batch_imgs.shape}")
print(f"Batch labels shape: {batch_lbls.shape}")


# =============================================================================
# SUMMARY OF SOLUTIONS
# =============================================================================

print("\n" + "="*60)
print("SUMMARY: Module 1 Lab 1 - All Tasks Completed")
print("="*60)
print(f"Task 1: Image shape = {image_data.shape}")
print(f"Task 2: Displayed first 4 non-agricultural images")
print(f"Task 3: Created agri_images_paths with {len(agri_images_paths)} sorted paths")
print(f"Task 4: Number of agricultural images = {num_agri_images}")
print("="*60)
