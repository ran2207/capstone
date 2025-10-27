# Resubmission Guide - Fix Failed Submissions

## 🔴 Problem Identified

You submitted **Python solution files (.py)** instead of **completed Jupyter Notebooks (.ipynb)**.

The Coursera AI grader needs:
- ✅ Jupyter Notebook files (.ipynb)
- ✅ With all cells executed
- ✅ With visible outputs (plots, metrics, text)

**NOT** raw Python scripts!

---

## ✅ Complete Fix Instructions

### Overview
1. Open original notebook (.ipynb)
2. Copy code from solution file (.py) into notebook cells
3. Execute all cells
4. Verify outputs appear
5. Save notebook
6. Upload .ipynb file to Coursera

---

## Question 1: Module 1 Lab 1 (10 points)

### Files Needed
- **Notebook:** `AI-capstone-M1L1-v1.ipynb`
- **Solution:** `solutions/M1L1_solutions.py`

### Steps

1. **Open Jupyter Notebook**
   ```bash
   jupyter notebook AI-capstone-M1L1-v1.ipynb
   ```

2. **Complete Tasks** (copy from M1L1_solutions.py):

   **Task 1: Image Shape**
   ```python
   import numpy as np
   from PIL import Image
   import os

   image_path = os.path.join('./images_dataSAT/class_0_non_agri',
                             os.listdir('./images_dataSAT/class_0_non_agri')[0])
   image_data = np.array(Image.open(image_path))
   print(f"Shape of a single image: {image_data.shape}")
   ```

   **Task 2: Display 4 Images**
   ```python
   import matplotlib.pyplot as plt

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
   plt.show()
   ```

   **Task 3: Create agri_images_paths**
   ```python
   dir_agri = './images_dataSAT/class_1_agri/'
   agri_images_paths = sorted([os.path.join(dir_agri, f)
                               for f in os.listdir(dir_agri)
                               if f.endswith(('.jpg', '.jpeg', '.png'))])
   print(f"Total paths: {len(agri_images_paths)}")
   print(f"First 5: {agri_images_paths[:5]}")
   ```

   **Task 4: Count Images**
   ```python
   num_agri_images = len(agri_images_paths)
   print(f"Number of agricultural images: {num_agri_images}")
   ```

   **Task 5: Display 4 Agri Images**
   ```python
   fig, axes = plt.subplots(1, 4, figsize=(16, 4))
   for idx in range(4):
       img = Image.open(agri_images_paths[idx])
       axes[idx].imshow(img)
       axes[idx].set_title(f"Agri Image {idx + 1}")
       axes[idx].axis('off')
   plt.tight_layout()
   plt.show()
   ```

3. **Execute All Cells**: Menu → Run → Run All Cells

4. **Verify Outputs**:
   - ✅ Image shape printed
   - ✅ 4 non-agri images displayed
   - ✅ Path count printed
   - ✅ Count printed
   - ✅ 4 agri images displayed

5. **Save**: File → Save

6. **Upload**: `AI-capstone-M1L1-v1.ipynb` to Question 1

---

## Question 2: Module 1 Lab 2 (8 points)

### Files Needed
- **Notebook:** `AI-capstone-M1L2-v1.ipynb`
- **Solution:** `solutions/M1L2_solutions.py`

### Steps

1. **Open notebook**

2. **Task 1: all_image_paths**
   ```python
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
   print(f"Total images: {len(all_image_paths)}")
   ```

   **Task 2: Create temp list**
   ```python
   import random

   labels = [0] * len(non_agri_paths) + [1] * len(agri_paths)
   temp = list(zip(all_image_paths, labels))
   random.shuffle(temp)

   print("5 Random samples:")
   samples = random.sample(temp, 5)
   for idx, (path, label) in enumerate(samples, 1):
       print(f"{idx}. {path} | Label: {label}")
   ```

   **Task 3: Generate batch**
   ```python
   from tensorflow.keras.preprocessing.image import ImageDataGenerator

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
   print(f"Labels: {batch_labels}")
   ```

   **Task 4: Validation generator**
   ```python
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

   print(f"Validation samples: {validation_generator.samples}")
   ```

3. **Execute all cells and verify outputs**

4. **Save and upload** `AI-capstone-M1L2-v1.ipynb`

---

## Question 3: Module 1 Lab 3 (10 points)

### Files Needed
- **Notebook:** `AI-capstone-M1L3-v1.ipynb`
- **Solution:** `solutions/M1L3_solutions.py`

### Key Tasks

**Task 1: custom_transform**
```python
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
```

**Task 2: Load dataset**
```python
dataset_path = './images_dataSAT'
imagefolder_dataset = datasets.ImageFolder(
    root=dataset_path,
    transform=custom_transform
)
print(f"Total images: {len(imagefolder_dataset)}")
```

**Task 3: Print classes**
```python
print("Class names:", imagefolder_dataset.classes)
print("Class to index:", imagefolder_dataset.class_to_idx)
```

**Task 4: Get batch and shapes**
```python
from torch.utils.data import DataLoader

imagefolder_loader = DataLoader(imagefolder_dataset, batch_size=8, shuffle=True)
batch_images, batch_labels = next(iter(imagefolder_loader))
print(f"Batch images shape: {batch_images.shape}")
print(f"Batch labels: {batch_labels}")
```

**Task 5: Display images**
```python
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
plt.show()
```

Execute, verify, save, and upload!

---

## Quick Checklist for All Questions

For **each question**:

- [ ] Open correct `.ipynb` notebook
- [ ] Copy code from corresponding `.py` solution file
- [ ] Execute **ALL** cells (Run → Run All Cells)
- [ ] Verify **all outputs appear** (prints, plots, metrics)
- [ ] **Save** notebook (File → Save)
- [ ] Upload the **.ipynb file** (NOT .py file!)
- [ ] Verify file uploaded correctly on Coursera

---

## File Mapping Reference

| Question | Notebook to Upload | Solution Reference |
|----------|-------------------|-------------------|
| 1 | AI-capstone-M1L1-v1.ipynb | solutions/M1L1_solutions.py |
| 2 | AI-capstone-M1L2-v1.ipynb | solutions/M1L2_solutions.py |
| 3 | AI-capstone-M1L3-v1.ipynb | solutions/M1L3_solutions.py |
| 4 | Lab-M2L1-...ipynb | solutions/M2L1_solutions.py |
| 5 | Lab-M2L2-...ipynb | solutions/M2L2_solutions.py |
| 6 | Lab-M2L3-...ipynb | solutions/M2L3_solutions.py |
| 7 | Lab-M3L1-...ipynb | solutions/M3L1_solutions.py |
| 8 | Lab-M3L2-...ipynb | solutions/M3L2_solutions.py |
| 9 | lab-M4L1-...ipynb | solutions/M3L3_solutions.py |

---

## Common Mistakes to Avoid

❌ **DON'T:**
- Upload .py files
- Upload notebooks without executing cells
- Upload notebooks without visible outputs
- Skip any tasks

✅ **DO:**
- Upload .ipynb files
- Execute all cells first
- Verify all outputs are visible
- Complete all tasks
- Save before uploading

---

## Getting Help

If you get stuck:
1. Check `solutions/` directory for complete code
2. Read `IMPLEMENTATION_GUIDE.md` for detailed instructions
3. Verify dataset is downloaded to `images_dataSAT/`
4. Make sure all dependencies are installed (`pip install -r requirements.txt`)

---

**Good luck with resubmission!** 🚀

You have all the solutions - just need to put them in the right format!
