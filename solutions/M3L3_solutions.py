"""
Module 3 Lab 3: Land Classification - CNN-Transformer Integration Evaluation
Solutions for all tasks (8 points total)

Copy these code blocks into the corresponding cells in the notebook:
lab-M4L1-Land-Classification-CNN-ViT-Integration-Evaluation-v1.ipynb
"""

import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                            f1_score, roc_auc_score, confusion_matrix,
                            classification_report, ConfusionMatrixDisplay)

# TensorFlow/Keras imports
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# =============================================================================
# TASK 1: Define dataset directory, dataloader, and model hyperparameters
# =============================================================================

# TASK 1 ANSWER:
dataset_path = os.path.join(".", "images_dataSAT")

# Hyperparameters for dataloaders
img_w, img_h = 64, 64
batch_size = 128
num_classes = 2
agri_class_labels = ["non-agri", "agri"]

# Hyperparameters for PyTorch CNN-ViT Hybrid model (same as training)
depth = 3
attn_heads = 6
embed_dim = 768

print("TASK 1: Configuration Set")
print("="*70)
print(f"Dataset path: {dataset_path}")
print(f"\nDataloader hyperparameters:")
print(f"  Image size: {img_w}x{img_h}")
print(f"  Batch size: {batch_size}")
print(f"  Number of classes: {num_classes}")
print(f"  Class labels: {agri_class_labels}")
print(f"\nPyTorch model hyperparameters:")
print(f"  Transformer depth: {depth}")
print(f"  Attention heads: {attn_heads}")
print(f"  Embedding dimension: {embed_dim}")


# =============================================================================
# PYTORCH MODEL ARCHITECTURE (Same as training)
# =============================================================================

class ConvNet(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, padding=2), nn.ReLU(inplace=True), nn.MaxPool2d(2), nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=5, padding=2), nn.ReLU(inplace=True), nn.MaxPool2d(2), nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=5, padding=2), nn.ReLU(inplace=True), nn.MaxPool2d(2), nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, kernel_size=5, padding=2), nn.ReLU(inplace=True), nn.MaxPool2d(2), nn.BatchNorm2d(256),
            nn.Conv2d(256, 512, kernel_size=5, padding=2), nn.ReLU(inplace=True), nn.MaxPool2d(2), nn.BatchNorm2d(512),
            nn.Conv2d(512, 1024, kernel_size=5, padding=2), nn.ReLU(inplace=True), nn.MaxPool2d(2), nn.BatchNorm2d(1024),
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


class PatchEmbed(nn.Module):
    def __init__(self, input_channel=1024, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(input_channel, embed_dim, kernel_size=1)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class MHSA(nn.Module):
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
    def __init__(self, dim, heads, mlp_ratio=4., dropout=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = MHSA(dim, heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp   = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViT(nn.Module):
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
# TASK 2: Instantiate PyTorch model
# =============================================================================

# Check device
device = "cuda" if torch.cuda.is_available() else "cpu"

# TASK 2 ANSWER:
pytorch_model = CNN_ViT_Hybrid(
    num_classes=num_classes,
    heads=attn_heads,
    depth=depth,
    embed_dim=embed_dim
).to(device)

print(f"\nTASK 2: PyTorch Model Instantiated")
print(f"  Device: {device}")
print(f"  Model: CNN_ViT_Hybrid")
print(f"  Parameters: depth={depth}, heads={attn_heads}, embed_dim={embed_dim}")

# Load pre-trained weights
pytorch_state_dict_path = "pytorch_cnn_vit_ai_capstone_model_state_dict.pth"
if os.path.exists(pytorch_state_dict_path):
    pytorch_model.load_state_dict(
        torch.load(pytorch_state_dict_path, map_location=device),
        strict=False
    )
    print(f"  Loaded weights from: {pytorch_state_dict_path}")
else:
    print(f"  Warning: Pre-trained weights not found at {pytorch_state_dict_path}")


# =============================================================================
# PYTORCH DATALOADER
# =============================================================================

test_transform = transforms.Compose([
    transforms.Resize((img_w, img_h)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

full_dataset = datasets.ImageFolder(dataset_path, transform=test_transform)
test_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False)

print(f"\nPyTorch DataLoader created:")
print(f"  Total samples: {len(full_dataset)}")
print(f"  Batches: {len(test_loader)}")


# =============================================================================
# PYTORCH MODEL INFERENCE
# =============================================================================

print("\nRunning PyTorch model inference...")
all_preds_pytorch = []
all_labels_pytorch = []
all_probs_pytorch = []

pytorch_model.eval()
with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="PyTorch Inference"):
        images = images.to(device)
        outputs = pytorch_model(images)
        preds = torch.argmax(outputs, dim=1)
        probs = torch.softmax(outputs, dim=1)[:, 1]  # Prob for class 1

        all_probs_pytorch.extend(probs.cpu())
        all_preds_pytorch.extend(preds.cpu().numpy().flatten())
        all_labels_pytorch.extend(labels.numpy())

print(f"PyTorch predictions collected: {len(all_preds_pytorch)}")


# =============================================================================
# KERAS MODEL LOADING
# =============================================================================

# Custom Keras layers (required for loading)
@tf.keras.utils.register_keras_serializable(package="Custom")
class AddPositionEmbedding(tf.keras.layers.Layer):
    def __init__(self, num_patches, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_patches = num_patches
        self.embed_dim   = embed_dim
        self.pos = self.add_weight(
            name="pos_embedding",
            shape=(1, num_patches, embed_dim),
            initializer="random_normal",
            trainable=True
        )

    def call(self, tokens):
        return tokens + self.pos

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_patches": self.num_patches,
            "embed_dim":   self.embed_dim,
        })
        return config


@tf.keras.utils.register_keras_serializable(package="Custom")
class TransformerBlockKeras(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads=8, mlp_dim=2048, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mlp_dim   = mlp_dim
        self.dropout   = dropout
        self.mha  = tf.keras.layers.MultiHeadAttention(num_heads, key_dim=embed_dim)
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.mlp = tf.keras.Sequential([
            tf.keras.layers.Dense(mlp_dim, activation="gelu"),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(embed_dim),
            tf.keras.layers.Dropout(dropout)
        ])

    def call(self, x):
        x = self.norm1(x + self.mha(x, x))
        return self.norm2(x + self.mlp(x))

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim":  self.embed_dim,
            "num_heads":  self.num_heads,
            "mlp_dim":    self.mlp_dim,
            "dropout":    self.dropout,
        })
        return config


# Load Keras model
keras_model_path = "keras_cnn_vit_ai_capstone.keras"
if os.path.exists(keras_model_path):
    keras_model = load_model(
        keras_model_path,
        custom_objects={
            "AddPositionEmbedding": AddPositionEmbedding,
            "TransformerBlock": TransformerBlockKeras
        }
    )
    print(f"\nKeras model loaded from: {keras_model_path}")
else:
    print(f"\nWarning: Keras model not found at {keras_model_path}")
    keras_model = None


# =============================================================================
# KERAS DATALOADER
# =============================================================================

if keras_model is not None:
    datagen = ImageDataGenerator(rescale=1./255)
    prediction_generator = datagen.flow_from_directory(
        dataset_path,
        target_size=(img_w, img_h),
        batch_size=batch_size,
        class_mode="binary",
        shuffle=False
    )

    print(f"\nKeras DataGenerator created:")
    print(f"  Total samples: {prediction_generator.samples}")

    # Run inference
    print("\nRunning Keras model inference...")
    all_probs_keras = keras_model.predict(prediction_generator, verbose=1)
    all_preds_keras = np.argmax(all_probs_keras, axis=1)
    all_labels_keras = prediction_generator.classes

    print(f"Keras predictions collected: {len(all_preds_keras)}")


# =============================================================================
# EVALUATION METRICS FUNCTION
# =============================================================================

def print_metrics(y_true, y_pred, y_prob, class_labels, model_name):
    """Print comprehensive evaluation metrics"""

    # Compute metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='binary')
    recall = recall_score(y_true, y_pred, average='binary')
    f1 = f1_score(y_true, y_pred, average='binary')

    # ROC-AUC
    y_prob = np.array(y_prob)
    if len(y_prob.shape) < 2:
        roc_auc = roc_auc_score(y_true, y_prob)
    elif len(y_prob.shape) == 2:
        roc_auc = roc_auc_score(y_true, y_prob[:, 1])
    else:
        roc_auc = np.nan

    # Print results
    print(f"\n{'='*70}")
    print(f"Evaluation metrics for the {model_name}")
    print(f"{'='*70}")
    print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    if not np.isnan(roc_auc):
        print(f"ROC-AUC:   {roc_auc:.4f}")

    # Classification report
    print(f"\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_labels, digits=4))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print(f"\nConfusion Matrix:")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{model_name.lower().replace(" ", "_")}_cm.png', dpi=300)
    plt.show()


# =============================================================================
# TASK 3: Print evaluation metrics for Keras model
# =============================================================================

if keras_model is not None:
    # TASK 3 ANSWER:
    print_metrics(
        y_true=all_labels_keras,
        y_pred=all_preds_keras,
        y_prob=all_probs_keras,
        class_labels=agri_class_labels,
        model_name="Keras CNN-ViT Hybrid Model"
    )
else:
    print("\nTASK 3: Skipped (Keras model not available)")


# =============================================================================
# TASK 4: Print evaluation metrics for PyTorch model
# =============================================================================

# TASK 4 ANSWER:
print_metrics(
    y_true=all_labels_pytorch,
    y_pred=all_preds_pytorch,
    y_prob=np.array(all_probs_pytorch),
    class_labels=agri_class_labels,
    model_name="PyTorch CNN-ViT Hybrid Model"
)


# =============================================================================
# MODEL COMPARISON
# =============================================================================

if keras_model is not None:
    print("\n" + "="*70)
    print("MODEL COMPARISON")
    print("="*70)

    keras_acc = accuracy_score(all_labels_keras, all_preds_keras)
    pytorch_acc = accuracy_score(all_labels_pytorch, all_preds_pytorch)

    print(f"\nAccuracy Comparison:")
    print(f"  Keras:   {keras_acc:.4f} ({keras_acc*100:.2f}%)")
    print(f"  PyTorch: {pytorch_acc:.4f} ({pytorch_acc*100:.2f}%)")
    print(f"  Difference: {abs(keras_acc - pytorch_acc):.4f}")

    # Visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    models = ['Keras CNN-ViT', 'PyTorch CNN-ViT']
    accuracies = [keras_acc, pytorch_acc]
    colors = ['skyblue', 'coral']

    bars = ax.bar(models, accuracies, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Accuracy')
    ax.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
    ax.set_ylim([0.8, 1.0])
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.4f}',
               ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300)
    plt.show()


# =============================================================================
# SUMMARY OF SOLUTIONS
# =============================================================================

print("\n" + "="*70)
print("SUMMARY: Module 3 Lab 3 (Final Evaluation) - All Tasks Completed")
print("="*70)
print(f"Task 1: Configured dataset path and hyperparameters")
print(f"Task 2: Instantiated PyTorch model (depth={depth}, heads={attn_heads})")
print(f"Task 3: Printed Keras CNN-ViT Hybrid Model metrics")
print(f"Task 4: Printed PyTorch CNN-ViT Hybrid Model metrics")
print(f"\nPyTorch Final Accuracy: {pytorch_acc:.4f} ({pytorch_acc*100:.2f}%)")
if keras_model is not None:
    print(f"Keras Final Accuracy: {keras_acc:.4f} ({keras_acc*100:.2f}%)")
print("\nCongratulations! All 9 labs completed successfully!")
print("="*70)
