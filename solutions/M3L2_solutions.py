"""
Module 3 Lab 2: Vision Transformers in PyTorch
Solutions for all tasks (12 points total)

Copy these code blocks into the corresponding cells in the notebook:
Lab-M3L2-Vision-Transformers-in-PyTorch-v1.ipynb
"""

import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

# =============================================================================
# PYTORCH VIT ARCHITECTURE COMPONENTS
# =============================================================================

class ConvNet(nn.Module):
    """CNN backbone for feature extraction"""
    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2), nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2), nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2), nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2), nn.BatchNorm2d(256),
            nn.Conv2d(256, 512, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2), nn.BatchNorm2d(512),
            nn.Conv2d(512, 1024, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2), nn.BatchNorm2d(1024)
        )

    def forward_features(self, x):
        return self.features(x)


class PatchEmbed(nn.Module):
    """Convert CNN features to token embeddings"""
    def __init__(self, input_channel=1024, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(input_channel, embed_dim, kernel_size=1)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)  # (B,L,D)
        return x


class MHSA(nn.Module):
    """Multi-Head Self Attention"""
    def __init__(self, dim, heads=8, dropout=0.):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x):
        B, N, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.reshape(B, N, self.heads, -1).transpose(1, 2)
        k = k.reshape(B, N, self.heads, -1).transpose(1, 2)
        v = v.reshape(B, N, self.heads, -1).transpose(1, 2)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = self.attn_drop(attn.softmax(dim=-1))
        x = torch.matmul(attn, v).transpose(1, 2).reshape(B, N, D)
        return self.proj_drop(self.proj(x))


class TransformerBlock(nn.Module):
    """Transformer encoder block"""
    def __init__(self, dim, heads, mlp_ratio=4., dropout=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = MHSA(dim, heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp   = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViT(nn.Module):
    """Vision Transformer"""
    def __init__(self, in_ch=1024, num_classes=2,
                 embed_dim=768, depth=6, heads=8,
                 mlp_ratio=4., dropout=0.1, max_tokens=50):
        super().__init__()
        self.patch = PatchEmbed(in_ch, embed_dim)
        self.cls   = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos   = nn.Parameter(torch.randn(1, max_tokens, embed_dim))
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.patch(x)
        B, L, _ = x.shape
        cls = self.cls.expand(B, -1, -1)
        x = torch.cat((cls, x), 1)
        x = x + self.pos[:, :L + 1]
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.norm(x)[:, 0])


class CNN_ViT_Hybrid(nn.Module):
    """CNN-ViT Hybrid model"""
    def __init__(self, num_classes=2, embed_dim=768, depth=6, heads=8):
        super().__init__()
        self.cnn = ConvNet(num_classes)
        self.vit = ViT(num_classes=num_classes,
                      embed_dim=embed_dim,
                      depth=depth,
                      heads=heads)

    def forward(self, x):
        return self.vit(self.cnn.forward_features(x))


# =============================================================================
# TASK 1: Define train_transform
# =============================================================================

img_size = 64

# TASK 1 ANSWER:
train_transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.RandomRotation(40),
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(0, shear=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

print("TASK 1: Training transform created")
print("  - Resize(64, 64)")
print("  - RandomRotation(40)")
print("  - RandomHorizontalFlip()")
print("  - RandomAffine(shear=0.2)")
print("  - Normalize")


# =============================================================================
# TASK 2: Define val_transform
# =============================================================================

# TASK 2 ANSWER:
val_transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

print("\nTASK 2: Validation transform created")
print("  - Resize(64, 64)")
print("  - Normalize (no augmentation)")


# =============================================================================
# DATASET LOADING
# =============================================================================

dataset_path = './images_dataSAT'

# Load full dataset
full_dataset = datasets.ImageFolder(root=dataset_path, transform=train_transform)

# Split 80/20
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size

train_dataset, val_dataset = random_split(
    full_dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

# Apply validation transform
val_dataset.dataset.transform = val_transform

print(f"\nDataset split:")
print(f"  Training: {len(train_dataset)}")
print(f"  Validation: {len(val_dataset)}")


# =============================================================================
# TASK 3: Create train_loader and val_loader
# =============================================================================

batch_size = 32

# TASK 3 ANSWER:
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0
)

print(f"\nTASK 3: DataLoaders created")
print(f"  Train batches: {len(train_loader)}")
print(f"  Val batches: {len(val_loader)}")
print(f"  Batch size: {batch_size}")


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def train(model, loader, optimizer, criterion, device):
    model.train()
    loss_sum, correct = 0, 0
    for x, y in tqdm(loader, desc="Training  "):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * x.size(0)
        correct  += (out.argmax(1) == y).sum().item()
    return loss_sum / len(loader.dataset), correct / len(loader.dataset)


def evaluate(model, loader, criterion, device):
    with torch.no_grad():
        model.eval()
        loss_sum, correct = 0, 0
        for x, y in tqdm(loader, desc="Validation"):
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            loss_sum += loss.item() * x.size(0)
            correct  += (out.argmax(1) == y).sum().item()
    return loss_sum / len(loader.dataset), correct / len(loader.dataset)


# =============================================================================
# TASK 4: Train CNN-ViT model with specified hyperparameters
# =============================================================================

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nTraining on device: {device}")

# Hyperparameters (TASK 4 parameters)
epochs = 5
attn_heads = 12
depth = 12
embed_dim = 768
lr = 0.001

print(f"\nTASK 4: Training CNN-ViT Hybrid Model")
print(f"  Epochs: {epochs}")
print(f"  Attention heads: {attn_heads}")
print(f"  Transformer depth: {depth}")
print(f"  Embedding dimension: {embed_dim}")
print(f"  Learning rate: {lr}")

# Create model
model = CNN_ViT_Hybrid(
    num_classes=2,
    heads=attn_heads,
    depth=depth,
    embed_dim=embed_dim
).to(device)

# Load pre-trained CNN weights
pytorch_state_dict_path = 'ai-capstone-pytorch-best-model_downloaded.pth'
if os.path.exists(pytorch_state_dict_path):
    model.cnn.load_state_dict(torch.load(pytorch_state_dict_path, map_location=device), strict=False)
    print(f"  Loaded pre-trained CNN from: {pytorch_state_dict_path}")

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Training loop
best_loss = float('inf')
tr_loss_all = []
te_loss_all = []
tr_acc_all = []
te_acc_all = []
training_time = []

print("\nStarting training...")
for epoch in range(1, epochs + 1):
    start_time = time.time()
    print(f"\nEpoch {epoch:02d}/{epochs:02d}")

    tr_loss, tr_acc = train(model, train_loader, optimizer, criterion, device)
    te_loss, te_acc = evaluate(model, val_loader, criterion, device)

    epoch_time = time.time() - start_time

    print(f"  Train Loss: {tr_loss:.4f} | Train Acc: {tr_acc:.4f}")
    print(f"  Val Loss:   {te_loss:.4f} | Val Acc:   {te_acc:.4f}")
    print(f"  Time: {epoch_time:.2f}s")

    tr_loss_all.append(tr_loss)
    te_loss_all.append(te_loss)
    tr_acc_all.append(tr_acc)
    te_acc_all.append(te_acc)
    training_time.append(epoch_time)

    # Save best model
    if te_loss < best_loss:
        best_loss = te_loss
        torch.save(model.state_dict(), 'pytorch_cnn_vit_model.pth')
        print(f"  *** Saved best model! ***")

print("\nTraining completed!")


# =============================================================================
# TASK 5: Plot validation loss comparison (model vs model_test)
# =============================================================================

# For demonstration, let's create a simpler model (model_test) with different hyperparameters
print("\nCreating model_test with depth=3, heads=6 for comparison...")

model_test = CNN_ViT_Hybrid(
    num_classes=2,
    heads=6,
    depth=3,
    embed_dim=768
).to(device)

if os.path.exists(pytorch_state_dict_path):
    model_test.cnn.load_state_dict(torch.load(pytorch_state_dict_path, map_location=device), strict=False)

criterion_test = nn.CrossEntropyLoss()
optimizer_test = torch.optim.Adam(model_test.parameters(), lr=lr)

te_loss_all_test = []
training_time_test = []

print("Training model_test...")
for epoch in range(1, epochs + 1):
    start_time = time.time()
    print(f"\nEpoch {epoch:02d}/{epochs:02d} (model_test)")

    tr_loss, tr_acc = train(model_test, train_loader, optimizer_test, criterion_test, device)
    te_loss, te_acc = evaluate(model_test, val_loader, criterion_test, device)

    te_loss_all_test.append(te_loss)
    training_time_test.append(time.time() - start_time)

    print(f"  Val Loss: {te_loss:.4f} | Val Acc: {te_acc:.4f}")

# TASK 5 ANSWER: Plot validation loss comparison
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(range(1, epochs+1), te_loss_all, label='Model (depth=12, heads=12)', marker='o')
ax.plot(range(1, epochs+1), te_loss_all_test, label='Model_test (depth=3, heads=6)', marker='s')
ax.set_title('Validation Loss Comparison', fontsize=14, fontweight='bold')
ax.set_xlabel('Epoch')
ax.set_ylabel('Validation Loss')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('pytorch_vit_val_loss_comparison.png', dpi=300)
plt.show()

print("\nTASK 5: Validation loss comparison plot saved")


# =============================================================================
# TASK 6: Plot training time comparison
# =============================================================================

# TASK 6 ANSWER:
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(range(1, epochs+1), training_time, label='Model (depth=12, heads=12)', marker='o')
ax.plot(range(1, epochs+1), training_time_test, label='Model_test (depth=3, heads=6)', marker='s')
ax.set_title('Training Time Comparison', fontsize=14, fontweight='bold')
ax.set_xlabel('Epoch')
ax.set_ylabel('Time (seconds)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('pytorch_vit_training_time_comparison.png', dpi=300)
plt.show()

print("\nTASK 6: Training time comparison plot saved")


# =============================================================================
# ADDITIONAL PLOTS
# =============================================================================

# Plot all metrics
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Training Loss
axes[0, 0].plot(range(1, epochs+1), tr_loss_all, marker='o', color='coral')
axes[0, 0].set_title('Training Loss')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].grid(True, alpha=0.3)

# Validation Loss
axes[0, 1].plot(range(1, epochs+1), te_loss_all, marker='s', color='dodgerblue')
axes[0, 1].set_title('Validation Loss')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Loss')
axes[0, 1].grid(True, alpha=0.3)

# Training Accuracy
axes[1, 0].plot(range(1, epochs+1), tr_acc_all, marker='o', color='green')
axes[1, 0].set_title('Training Accuracy')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Accuracy')
axes[1, 0].grid(True, alpha=0.3)

# Validation Accuracy
axes[1, 1].plot(range(1, epochs+1), te_acc_all, marker='s', color='purple')
axes[1, 1].set_title('Validation Accuracy')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Accuracy')
axes[1, 1].grid(True, alpha=0.3)

plt.suptitle('PyTorch CNN-ViT Training Metrics', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('pytorch_vit_all_metrics.png', dpi=300)
plt.show()


# =============================================================================
# SUMMARY OF SOLUTIONS
# =============================================================================

print("\n" + "="*70)
print("SUMMARY: Module 3 Lab 2 - All Tasks Completed")
print("="*70)
print(f"Task 1: Created train_transform with augmentation")
print(f"Task 2: Created val_transform without augmentation")
print(f"Task 3: Created train_loader and val_loader")
print(f"Task 4: Trained CNN-ViT (epochs={epochs}, heads={attn_heads}, depth={depth})")
print(f"Task 5: Plotted validation loss comparison")
print(f"Task 6: Plotted training time comparison")
print(f"\nFinal Model Validation Accuracy: {te_acc_all[-1]:.4f} ({te_acc_all[-1]*100:.2f}%)")
print(f"Best Validation Loss: {best_loss:.4f}")
print("="*70)
