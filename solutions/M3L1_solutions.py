"""
Module 3 Lab 1: Vision Transformers in Keras
Solutions for all tasks (10 points total)

Copy these code blocks into the corresponding cells in the notebook:
Lab-M3L1-Vision-Transformers-in-Keras-v1.ipynb
"""

import os
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# =============================================================================
# TASK 1: Load pre-trained CNN model and display summary
# =============================================================================

# Path to pre-trained Keras CNN model
keras_model_path = 'ai-capstone-keras-best-model-model_downloaded.keras'

# TASK 1 ANSWER:
cnn_model = load_model(keras_model_path)  # Load the CNN model

# Display model summary (uncomment to see architecture)
cnn_model.summary()

print(f"Loaded CNN model from: {keras_model_path}")
print(f"Model has {len(cnn_model.layers)} layers")
print("\nNow examine the summary to identify the feature extraction layer...")


# =============================================================================
# TASK 2: Identify feature layer name for feature extraction
# =============================================================================

# Based on the model summary from Task 1, identify the last convolutional
# layer before GlobalAveragePooling2D

# TASK 2 ANSWER:
feature_layer_name = "batch_normalization_5"

print(f"\nTASK 2: Feature extraction layer: {feature_layer_name}")
print("This is the last batch normalization layer before pooling.")


# =============================================================================
# CUSTOM KERAS LAYERS FOR ViT
# =============================================================================

@tf.keras.utils.register_keras_serializable(package="Custom")
class AddPositionEmbedding(layers.Layer):
    """
    Adds learnable positional embeddings to token sequences
    """
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
class TransformerBlock(layers.Layer):
    """
    Vision Transformer encoder block with multi-head attention and MLP
    """
    def __init__(self, embed_dim, num_heads=8, mlp_dim=2048, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mlp_dim   = mlp_dim
        self.dropout   = dropout

        # Multi-head attention
        self.mha  = layers.MultiHeadAttention(num_heads, key_dim=embed_dim)

        # Layer normalization
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)

        # MLP block
        self.mlp = tf.keras.Sequential([
            layers.Dense(mlp_dim, activation="gelu"),
            layers.Dropout(dropout),
            layers.Dense(embed_dim),
            layers.Dropout(dropout)
        ])

    def call(self, x):
        # Attention block with residual connection
        x = self.norm1(x + self.mha(x, x))
        # MLP block with residual connection
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


# =============================================================================
# HYBRID MODEL BUILDER FUNCTION
# =============================================================================

def build_cnn_vit_hybrid(cnn_model,
                        feature_layer_name,
                        num_transformer_layers=4,
                        num_heads=8,
                        mlp_dim=2048,
                        num_classes=2):
    """
    Build CNN-ViT hybrid model

    Args:
        cnn_model: Pre-trained CNN model
        feature_layer_name: Name of layer to extract features from
        num_transformer_layers: Number of transformer blocks
        num_heads: Number of attention heads
        mlp_dim: MLP hidden dimension
        num_classes: Number of output classes

    Returns:
        Hybrid CNN-ViT model
    """
    # Freeze CNN backbone (optional: set to True to fine-tune)
    cnn_model.trainable = False

    # Extract feature maps from specified layer
    features = cnn_model.get_layer(feature_layer_name).output
    H, W, C = features.shape[1], features.shape[2], features.shape[3]

    # Reshape spatial grid to token sequence
    x = layers.Reshape((H * W, C))(features)

    # Add positional embeddings
    x = AddPositionEmbedding(H * W, C)(x)

    # Stack transformer encoder blocks
    for _ in range(num_transformer_layers):
        x = TransformerBlock(C, num_heads, mlp_dim)(x)

    # Global average pooling over tokens
    x = layers.GlobalAveragePooling1D()(x)

    # Classification head
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    # Create model
    model = Model(cnn_model.layers[0].input, outputs, name="CNN_ViT_Hybrid")

    return model


# =============================================================================
# TASK 3: Define hybrid_model using build_cnn_vit_hybrid function
# =============================================================================

# Parameters for the hybrid model
num_classes = 2  # Binary classification (agri vs non-agri)

# TASK 3 ANSWER:
hybrid_model = build_cnn_vit_hybrid(
    cnn_model,
    feature_layer_name=feature_layer_name,
    num_transformer_layers=4,
    num_heads=8,
    mlp_dim=2048,
    num_classes=num_classes
)

print("\nTASK 3: Hybrid CNN-ViT model created")
print(f"  Transformer layers: 4")
print(f"  Attention heads: 8")
print(f"  MLP dimension: 2048")
print(f"  Output classes: {num_classes}")

# Display model summary
print("\nHybrid Model Architecture:")
hybrid_model.summary()


# =============================================================================
# TASK 4: Compile the hybrid_model
# =============================================================================

# TASK 4 ANSWER:
hybrid_model.compile(
    optimizer=Adam(learning_rate=1e-4),  # Lower learning rate for fine-tuning
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nTASK 4: Model compiled successfully!")
print("  Optimizer: Adam (lr=0.0001)")
print("  Loss: categorical_crossentropy")
print("  Metrics: accuracy")


# =============================================================================
# DATA GENERATORS
# =============================================================================

dataset_path = './images_dataSAT'
img_w, img_h = 64, 64
batch_size = 4  # Small batch size due to model complexity
num_classes = 2

# ImageDataGenerator with augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
    validation_split=0.2
)

# Training generator
train_gen = datagen.flow_from_directory(
    dataset_path,
    target_size=(img_w, img_h),
    batch_size=batch_size,
    class_mode="categorical",
    subset="training",
    shuffle=True
)

# Validation generator
val_gen = datagen.flow_from_directory(
    dataset_path,
    target_size=(img_w, img_h),
    batch_size=batch_size,
    class_mode="categorical",
    subset="validation",
    shuffle=True
)

print(f"\nData Generators Created:")
print(f"  Training samples: {train_gen.samples}")
print(f"  Validation samples: {val_gen.samples}")
print(f"  Batch size: {batch_size}")


# =============================================================================
# MODEL CHECKPOINT
# =============================================================================

model_name = "keras_cnn_vit.model.keras"

checkpoint_cb = ModelCheckpoint(
    filepath=model_name,
    save_weights_only=False,
    monitor='val_loss',
    mode='min',
    save_best_only=True,
    verbose=1
)

print(f"\nCheckpoint callback created:")
print(f"  Save path: {model_name}")
print(f"  Monitor: val_loss (min)")


# =============================================================================
# TASK 5: Train the hybrid model
# =============================================================================

epochs = 3
steps_per_epoch = 128  # Limit steps for computational efficiency

print(f"\nTASK 5: Training hybrid model...")
print(f"  Epochs: {epochs}")
print(f"  Steps per epoch: {steps_per_epoch}")
print("\nThis may take 30-60 minutes depending on hardware...")

# TASK 5 ANSWER:
fit = hybrid_model.fit(
    train_gen,
    epochs=epochs,
    validation_data=val_gen,
    callbacks=[checkpoint_cb],
    steps_per_epoch=steps_per_epoch
)

print("\nTraining completed!")


# =============================================================================
# VISUALIZE TRAINING HISTORY
# =============================================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot Accuracy
axes[0].plot(fit.history['accuracy'], label='Training Accuracy', marker='o')
axes[0].plot(fit.history['val_accuracy'], label='Validation Accuracy', marker='s')
axes[0].set_title('Keras CNN-ViT Hybrid: Model Accuracy', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot Loss
axes[1].plot(fit.history['loss'], label='Training Loss', marker='o', color='coral')
axes[1].plot(fit.history['val_loss'], label='Validation Loss', marker='s', color='dodgerblue')
axes[1].set_title('Keras CNN-ViT Hybrid: Model Loss', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('keras_cnn_vit_training_history.png', dpi=300)
plt.show()

print("\nTraining plots saved as 'keras_cnn_vit_training_history.png'")


# =============================================================================
# BONUS: Evaluate model
# =============================================================================

print("\nEvaluating model on validation set...")
val_loss, val_accuracy = hybrid_model.evaluate(val_gen, verbose=1)

print(f"\nFinal Validation Results:")
print(f"  Loss: {val_loss:.4f}")
print(f"  Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")


# =============================================================================
# MODEL SAVING
# =============================================================================

print("\nSaving final model...")
hybrid_model.save('keras_cnn_vit_final.keras')
print("Model saved as 'keras_cnn_vit_final.keras'")

print("\nTo download the model:")
print("  1. Right-click on the file in the file browser")
print("  2. Select 'Download'")
print("  3. Save to your local machine for submission")


# =============================================================================
# SUMMARY OF SOLUTIONS
# =============================================================================

print("\n" + "="*70)
print("SUMMARY: Module 3 Lab 1 - All Tasks Completed")
print("="*70)
print(f"Task 1: Loaded CNN model from {keras_model_path}")
print(f"Task 2: Identified feature layer: {feature_layer_name}")
print(f"Task 3: Built hybrid CNN-ViT model with 4 transformer layers")
print(f"Task 4: Compiled model with Adam(1e-4) and categorical_crossentropy")
print(f"Task 5: Trained model for {epochs} epochs with {steps_per_epoch} steps/epoch")
print(f"\nFinal Validation Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
print("="*70)
