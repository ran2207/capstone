# Dataset Setup Guide

## Overview

This guide provides detailed instructions for downloading and setting up the satellite imagery dataset used in the AI Capstone project.

**Dataset Name:** Images DataSAT
**Source:** IBM Cloud Object Storage
**Size:** ~450 MB (compressed), ~600 MB (extracted)
**Images:** ~10,000 total (5,000 per class)
**Classes:** Agricultural land, Non-agricultural land

---

## Method 1: Automatic Download (Recommended)

The notebooks include automatic dataset downloading using the `skillsnetwork` library. Simply run the data download cells in any notebook:

```python
import os
import skillsnetwork

data_dir = "."
dataset_url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/4Z1fwRR295-1O3PMQBH6Dg/images-dataSAT.tar"

await skillsnetwork.prepare(url=dataset_url, path=data_dir, overwrite=True)
```

**Expected Output:**
```
Dataset URL: https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/...
Downloading...
Extracting to current directory...
Done!
```

---

## Method 2: Manual Download (Fallback)

If Method 1 fails due to network issues or environment restrictions:

### Step 1: Download the Dataset

**Using Browser:**
1. Open this URL in your browser:
   ```
   https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/4Z1fwRR295-1O3PMQBH6Dg/images-dataSAT.tar
   ```
2. Save file as `images-dataSAT.tar` to your project directory

**Using wget (Linux/Mac):**
```bash
cd ~/Documents/GitHub/Personal/coursera/capstone
wget https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/4Z1fwRR295-1O3PMQBH6Dg/images-dataSAT.tar
```

**Using curl (Linux/Mac):**
```bash
cd ~/Documents/GitHub/Personal/coursera/capstone
curl -O https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/4Z1fwRR295-1O3PMQBH6Dg/images-dataSAT.tar
```

**Using Python:**
```python
import httpx
import os

url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/4Z1fwRR295-1O3PMQBH6Dg/images-dataSAT.tar"
tar_path = "images-dataSAT.tar"

async def download_dataset():
    async with httpx.AsyncClient(timeout=300.0) as client:
        response = await client.get(url, follow_redirects=True)
        response.raise_for_status()
        with open(tar_path, "wb") as f:
            f.write(response.content)
        print(f"Downloaded: {tar_path}")

await download_dataset()
```

### Step 2: Extract the Archive

**Using Python:**
```python
import tarfile

tar_path = "images-dataSAT.tar"
extract_dir = "."

with tarfile.open(tar_path, 'r:*') as tar_ref:
    tar_ref.extractall(path=extract_dir)
    print(f"Extracted to: {extract_dir}")
```

**Using Command Line (Linux/Mac):**
```bash
tar -xvf images-dataSAT.tar
```

**Using Command Line (Windows with 7-Zip):**
```cmd
7z x images-dataSAT.tar
```

### Step 3: Verify Extraction

After extraction, you should have this structure:

```
coursera/capstone/
└── images_dataSAT/
    ├── class_0_non_agri/
    │   ├── image_0000.jpg
    │   ├── image_0001.jpg
    │   ├── ...
    │   └── image_4999.jpg  (~5000 images)
    └── class_1_agri/
        ├── image_0000.jpg
        ├── image_0001.jpg
        ├── ...
        └── image_4999.jpg  (~5000 images)
```

**Verification Script:**
```python
import os

dataset_path = "images_dataSAT"
class_0_path = os.path.join(dataset_path, "class_0_non_agri")
class_1_path = os.path.join(dataset_path, "class_1_agri")

# Count images
non_agri_count = len([f for f in os.listdir(class_0_path) if f.endswith(('.jpg', '.png'))])
agri_count = len([f for f in os.listdir(class_1_path) if f.endswith(('.jpg', '.png'))])

print(f"Non-agricultural images: {non_agri_count}")
print(f"Agricultural images: {agri_count}")
print(f"Total images: {non_agri_count + agri_count}")

# Expected output:
# Non-agricultural images: ~5000
# Agricultural images: ~5000
# Total images: ~10000
```

---

## Method 3: Google Colab (Cloud-Based)

If running on Google Colab, use this approach:

```python
!wget -q https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/4Z1fwRR295-1O3PMQBH6Dg/images-dataSAT.tar
!tar -xf images-dataSAT.tar
!ls -lh images_dataSAT/
```

---

## Dataset Description

### Class Distribution

| Class | Label | Count | Description |
|-------|-------|-------|-------------|
| Non-Agricultural Land | 0 | ~5000 | Urban areas, water bodies, forests, bare land |
| Agricultural Land | 1 | ~5000 | Farmland, crop fields, plantations |

### Image Specifications

- **Format:** JPEG
- **Original Size:** Variable (resized to 64x64 during preprocessing)
- **Color Space:** RGB (3 channels)
- **Bit Depth:** 8-bit per channel
- **Total Size:** ~600 MB (uncompressed)

### Sample Images

**Non-Agricultural Land Examples:**
- Urban buildings and roads
- Water bodies (lakes, rivers)
- Rocky terrain
- Dense forests

**Agricultural Land Examples:**
- Crop fields (various stages)
- Farmland patterns
- Irrigation systems
- Agricultural plots

---

## Data Preprocessing

The notebooks apply these preprocessing steps:

### Common Transforms
```python
# Resizing to standard dimensions
transforms.Resize((64, 64))

# Normalization (ImageNet stats)
transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

# Convert to tensor
transforms.ToTensor()
```

### Training Augmentation
```python
# Rotation
transforms.RandomRotation(40)

# Horizontal flip
transforms.RandomHorizontalFlip(p=0.5)

# Vertical flip
transforms.RandomVerticalFlip(p=0.2)

# Affine transforms
transforms.RandomAffine(0, shear=0.2)

# Zoom
transforms.RandomResizedCrop(64, scale=(0.8, 1.0))
```

---

## Troubleshooting

### Issue 1: Download Hangs or Times Out

**Solution:**
```python
# Increase timeout
async with httpx.AsyncClient(timeout=600.0) as client:
    response = await client.get(url)
```

**Alternative:** Download using browser and place in project directory manually

### Issue 2: Permission Denied During Extraction

**Solution (Linux/Mac):**
```bash
chmod +w .
tar -xvf images-dataSAT.tar
```

**Solution (Windows):**
- Right-click folder → Properties → Security
- Grant write permissions

### Issue 3: Corrupted Archive

**Solution:**
1. Delete `images-dataSAT.tar`
2. Re-download using Method 2
3. Verify file size (~450 MB)

**Check integrity:**
```python
import os
file_size = os.path.getsize("images-dataSAT.tar")
print(f"File size: {file_size / (1024**2):.2f} MB")
# Should be ~450 MB
```

### Issue 4: Not Enough Disk Space

**Required Space:**
- Compressed: 450 MB
- Extracted: 600 MB
- Models: ~300 MB
- **Total: ~1.5 GB**

**Solution:**
- Free up disk space
- Use external drive
- Use cloud storage (Google Drive, Colab)

---

## Data Storage Recommendations

### Local Development
```
~/Documents/GitHub/Personal/coursera/capstone/
└── images_dataSAT/  # Keep dataset in project root
```

### Google Colab
```
/content/
└── images_dataSAT/  # Will be lost after session
```

**Persistent Storage on Colab:**
```python
from google.colab import drive
drive.mount('/content/drive')

# Extract to Google Drive
!tar -xf images-dataSAT.tar -C /content/drive/MyDrive/
```

### Cloud Storage (S3, GCS, Azure)
For production deployments, consider cloud storage:

```python
# Example: AWS S3
import boto3
s3 = boto3.client('s3')
# Upload logic here
```

---

## Dataset Statistics

After loading, you can compute statistics:

```python
import numpy as np
from PIL import Image
import os

def compute_dataset_stats(dataset_path):
    all_images = []

    for class_folder in ['class_0_non_agri', 'class_1_agri']:
        folder_path = os.path.join(dataset_path, class_folder)
        for img_file in os.listdir(folder_path)[:100]:  # Sample 100
            if img_file.endswith(('.jpg', '.png')):
                img_path = os.path.join(folder_path, img_file)
                img = Image.open(img_path)
                img_array = np.array(img) / 255.0
                all_images.append(img_array)

    all_images = np.array(all_images)
    mean = np.mean(all_images, axis=(0,1,2))
    std = np.std(all_images, axis=(0,1,2))

    print(f"Dataset Mean (RGB): {mean}")
    print(f"Dataset Std (RGB): {std}")
    return mean, std

mean, std = compute_dataset_stats("images_dataSAT")
```

---

## Alternative Dataset Sources

If IBM Cloud Storage is unavailable, you can use alternative satellite imagery datasets:

### 1. EuroSAT
- **URL:** https://github.com/phelber/EuroSAT
- **Size:** 27,000 images, 10 classes
- **Resolution:** 64x64 pixels

### 2. UCMerced Land Use
- **URL:** http://weegee.vision.ucmerced.edu/datasets/landuse.html
- **Size:** 2,100 images, 21 classes
- **Resolution:** 256x256 pixels

### 3. NWPU-RESISC45
- **URL:** http://www.escience.cn/people/JunweiHan/NWPU-RESISC45.html
- **Size:** 31,500 images, 45 classes
- **Resolution:** 256x256 pixels

**Note:** Using alternative datasets will require modifying the number of classes and data paths in notebooks.

---

## Data Validation Checklist

Before proceeding with training:

- [ ] Dataset extracted to `images_dataSAT/`
- [ ] Both class folders present
- [ ] ~5000 images per class
- [ ] Images can be opened with PIL/OpenCV
- [ ] Sufficient disk space for models (~1 GB)
- [ ] Read/write permissions on directory

---

## Support

If you encounter persistent issues:

1. Check notebook-specific download cells (fallback methods included)
2. Verify internet connection and firewall settings
3. Try different download method (browser vs. script)
4. Check available disk space
5. Consult course forum or instructor

---

**Last Updated:** 2025-10-26
