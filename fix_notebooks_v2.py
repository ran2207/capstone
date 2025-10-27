#!/usr/bin/env python3
"""
Fix notebooks based on grader feedback:
1. Split large solution cells into smaller cells
2. Remove syntax errors
3. Ensure all cells execute properly
4. Make sure outputs are visible
"""
import json
import re
from pathlib import Path

def clean_solution_code(code):
    """Clean solution code and split into logical sections"""
    # Remove excessive comments/headers
    code = re.sub(r'# =+\n# (TASK|Task) \d+:.*?\n# =+\n', '', code)

    # Split by task/section
    sections = []
    current_section = []

    lines = code.split('\n')
    for line in lines:
        # Check if this is a new logical section (imports, task marker, etc.)
        if (line.startswith('import ') or line.startswith('from ')) and current_section:
            # Flush current section
            if current_section:
                sections.append('\n'.join(current_section))
                current_section = []

        current_section.append(line)

    # Flush remaining
    if current_section:
        sections.append('\n'.join(current_section))

    return sections

def fix_m1l1_notebook():
    """Fix Module 1 Lab 1 - Missing outputs for tasks 2,3,4,5"""
    nb_path = "AI-capstone-M1L1-v1.ipynb"
    with open(nb_path, 'r') as f:
        nb = json.load(f)

    # Find the large solution cell at the end
    solution_cell_idx = None
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            if 'M1L1' in source or len(source) > 2000:
                solution_cell_idx = i
                break

    if solution_cell_idx:
        # Remove the large solution cell
        large_cell = nb['cells'].pop(solution_cell_idx)

        # Add back individual task cells
        tasks = [
            # Task 1 - already working
            None,

            # Task 2: Display first 4 non-agri images
            """# Task 2: Display first 4 non-agricultural images
import matplotlib.pyplot as plt
from PIL import Image

dir_non_agri = './images_dataSAT/class_0_non_agri/'
image_files = sorted([f for f in os.listdir(dir_non_agri)
                     if f.endswith(('.jpg', '.jpeg', '.png'))])[:4]

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for idx, img_file in enumerate(image_files):
    img_path = os.path.join(dir_non_agri, img_file)
    img = Image.open(img_path)
    axes[idx].imshow(img)
    axes[idx].set_title(f"Image {idx + 1}")
    axes[idx].axis('off')
plt.tight_layout()
plt.show()""",

            # Task 3: Create agri_images_paths
            """# Task 3: Create sorted agricultural image paths
dir_agri = './images_dataSAT/class_1_agri/'
agri_images_paths = sorted([os.path.join(dir_agri, f)
                           for f in os.listdir(dir_agri)
                           if f.endswith(('.jpg', '.jpeg', '.png'))])
print(f"Total agricultural image paths: {len(agri_images_paths)}")
print(f"First 5 paths: {agri_images_paths[:5]}")""",

            # Task 4: Count agricultural images
            """# Task 4: Count agricultural images
num_agri_images = len(agri_images_paths)
print(f"Number of agricultural images: {num_agri_images}")""",

            # Task 5: Display first 4 agri images
            """# Task 5: Display first 4 agricultural images
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for idx in range(4):
    img = Image.open(agri_images_paths[idx])
    axes[idx].imshow(img)
    axes[idx].set_title(f"Agri Image {idx + 1}")
    axes[idx].axis('off')
plt.tight_layout()
plt.show()"""
        ]

        # Insert task cells
        for task_code in tasks:
            if task_code:
                nb['cells'].append({
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": task_code.split('\n')
                })

    # Save
    with open(nb_path, 'w') as f:
        json.dump(nb, f, indent=2)

    print(f"✓ Fixed {nb_path}")

def fix_m1l2_notebook():
    """Fix Module 1 Lab 2 - Split large failing cell"""
    nb_path = "AI-capstone-M1L2-v1.ipynb"
    with open(nb_path, 'r') as f:
        nb = json.load(f)

    # Remove large solution cell
    nb['cells'] = [c for c in nb['cells']
                   if not (c['cell_type'] == 'code' and len(''.join(c.get('source', []))) > 2000)]

    # Add proper task cells
    tasks = [
        """# Task 1: Create all_image_paths
import os

dir_non_agri = './images_dataSAT/class_0_non_agri/'
dir_agri = './images_dataSAT/class_1_agri/'

non_agri_paths = [os.path.join(dir_non_agri, f)
                  for f in os.listdir(dir_non_agri)
                  if f.endswith(('.jpg', '.jpeg', '.png'))]

agri_paths = [os.path.join(dir_agri, f)
              for f in os.listdir(dir_agri)
              if f.endswith(('.jpg', '.jpeg', '.png'))]

all_image_paths = non_agri_paths + agri_paths
print(f"Total images: {len(all_image_paths)}")""",

        """# Task 2: Create temp list and print 5 samples
import random

labels = [0] * len(non_agri_paths) + [1] * len(agri_paths)
temp = list(zip(all_image_paths, labels))
random.shuffle(temp)

print("5 Random samples:")
samples = random.sample(temp, 5)
for idx, (path, label) in enumerate(samples, 1):
    print(f"{idx}. {path} | Label: {label}")""",

        """# Task 3: Generate batch using custom generator
import numpy as np
from PIL import Image

def custom_data_generator(image_paths, labels, batch_size=8, img_size=(64, 64)):
    num_samples = len(image_paths)
    indices = list(range(num_samples))

    while True:
        random.shuffle(indices)
        for start_idx in range(0, num_samples, batch_size):
            batch_indices = indices[start_idx:start_idx + batch_size]
            batch_images = []
            batch_labels = []

            for idx in batch_indices:
                img = Image.open(image_paths[idx])
                img = img.resize(img_size)
                img_array = np.array(img) / 255.0
                batch_images.append(img_array)
                batch_labels.append(labels[idx])

            yield np.array(batch_images), np.array(batch_labels)

gen = custom_data_generator(all_image_paths, labels, batch_size=8)
batch_images, batch_labels = next(gen)
print(f"Batch shape: {batch_images.shape}")
print(f"Labels: {batch_labels}")""",

        """# Task 4: Create validation generator
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

validation_generator = datagen.flow_from_directory(
    './images_dataSAT',
    target_size=(64, 64),
    batch_size=8,
    class_mode='binary',
    subset='validation',
    shuffle=True
)

print(f"Validation samples: {validation_generator.samples}")"""
    ]

    for task_code in tasks:
        nb['cells'].append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": task_code.split('\n')
        })

    with open(nb_path, 'w') as f:
        json.dump(nb, f, indent=2)

    print(f"✓ Fixed {nb_path}")

def fix_m1l3_notebook():
    """Fix Module 1 Lab 3 - Fix SyntaxError and split cells"""
    nb_path = "AI-capstone-M1L3-v1.ipynb"
    with open(nb_path, 'r') as f:
        nb = json.load(f)

    # Remove problematic solution cell
    nb['cells'] = [c for c in nb['cells']
                   if not (c['cell_type'] == 'code' and len(''.join(c.get('source', []))) > 2000)]

    tasks = [
        """# Task 1: Define custom_transform
import torch
from torchvision import datasets, transforms

custom_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomRotation(45),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

print("Custom transform pipeline created")""",

        """# Task 2: Load dataset with ImageFolder
dataset_path = './images_dataSAT'
imagefolder_dataset = datasets.ImageFolder(
    root=dataset_path,
    transform=custom_transform
)
print(f"Total images in dataset: {len(imagefolder_dataset)}")""",

        """# Task 3: Print class names and indices
print("Class names:", imagefolder_dataset.classes)
print("Class to index mapping:", imagefolder_dataset.class_to_idx)""",

        """# Task 4: Get batch from DataLoader
from torch.utils.data import DataLoader

imagefolder_loader = DataLoader(imagefolder_dataset, batch_size=8, shuffle=True, num_workers=0)
batch_images, batch_labels = next(iter(imagefolder_loader))
print(f"Batch images shape: {batch_images.shape}")
print(f"Batch labels: {batch_labels}")""",

        """# Task 5: Display images from batch
import matplotlib.pyplot as plt
import numpy as np

def denormalize(tensor):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    tensor = tensor.clone()
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()
for idx in range(8):
    img = denormalize(batch_images[idx])
    img = img.numpy().transpose((1, 2, 0))
    img = np.clip(img, 0, 1)
    axes[idx].imshow(img)
    axes[idx].set_title(f"Label: {batch_labels[idx].item()}")
    axes[idx].axis('off')
plt.tight_layout()
plt.show()"""
    ]

    for task_code in tasks:
        nb['cells'].append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": task_code.split('\n')
        })

    with open(nb_path, 'w') as f:
        json.dump(nb, f, indent=2)

    print(f"✓ Fixed {nb_path}")

def main():
    print("Fixing notebooks based on grader feedback...")
    print("="*60)

    fix_m1l1_notebook()
    fix_m1l2_notebook()
    fix_m1l3_notebook()

    print("="*60)
    print("✓ All notebooks fixed")
    print("\nNext: Execute notebooks to generate outputs")

if __name__ == "__main__":
    main()
