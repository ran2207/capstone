"""
Module 2 Lab 1: Train and Evaluate a Keras-Based Classifier
Solutions for all tasks (12 points total)

Copy these code blocks into the corresponding cells in the notebook:
Lab-M2L1-Train-and-Evaluate-a-Keras-Based-Classifier-v1.ipynb
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import HeUniform

# =============================================================================
# TASK 1: Walk through dataset_path to create list fnames of all image files
# =============================================================================

dataset_path = './images_dataSAT'

# Walk through the directory tree
fnames = []
for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.endswith(('.jpg', '.jpeg', '.png')):
            full_path = os.path.join(root, file)
            fnames.append(full_path)

print(f"Total image files found: {len(fnames)}")
print(f"\nFirst 5 files:")
for fname in fnames[:5]:
    print(f"  {fname}")

# Alternative one-liner approach
fnames_alt = [os.path.join(root, file)
              for root, dirs, files in os.walk(dataset_path)
              for file in files if file.endswith(('.jpg', '.jpeg', '.png'))]

print(f"\nVerification: {len(fnames_alt)} files found using alternative method")


# =============================================================================
# TASK 2: Create validation_generator with appropriate settings
# =============================================================================

img_w, img_h = 64, 64
batch_size = 128

# Define ImageDataGenerator with augmentation and validation split
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2  # 20% for validation
)

# Create training generator
train_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(img_w, img_h),
    batch_size=batch_size,
    class_mode='categorical',  # For 2 classes with softmax
    subset='training',
    shuffle=True
)

# Create validation generator (TASK 2 ANSWER)
validation_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(img_w, img_h),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=False  # No shuffling for validation
)

print(f"\nTraining samples: {train_generator.samples}")
print(f"Validation samples: {validation_generator.samples}")
print(f"Number of classes: {train_generator.num_classes}")
print(f"Class indices: {train_generator.class_indices}")


# =============================================================================
# TASK 3: Count the total number of CNN model layers
# =============================================================================

# First, let's build the model (needed for Task 4 anyway)
# Then we'll count the layers

# CNN Architecture: 4 Conv2D blocks + 5 Dense layers

model = models.Sequential([
    # First Conv2D block
    layers.Conv2D(32, (5, 5), activation='relu', kernel_initializer=HeUniform(),
                  input_shape=(img_w, img_h, 3), padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.BatchNormalization(),

    # Second Conv2D block
    layers.Conv2D(64, (5, 5), activation='relu', kernel_initializer=HeUniform(), padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.BatchNormalization(),

    # Third Conv2D block
    layers.Conv2D(128, (5, 5), activation='relu', kernel_initializer=HeUniform(), padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.BatchNormalization(),

    # Fourth Conv2D block
    layers.Conv2D(256, (5, 5), activation='relu', kernel_initializer=HeUniform(), padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.BatchNormalization(),

    # Flatten and Dense layers
    layers.Flatten(),

    # First Dense layer
    layers.Dense(512, activation='relu', kernel_initializer=HeUniform()),
    layers.Dropout(0.4),
    layers.BatchNormalization(),

    # Second Dense layer
    layers.Dense(256, activation='relu', kernel_initializer=HeUniform()),
    layers.Dropout(0.3),
    layers.BatchNormalization(),

    # Third Dense layer
    layers.Dense(128, activation='relu', kernel_initializer=HeUniform()),
    layers.Dropout(0.2),

    # Fourth Dense layer
    layers.Dense(64, activation='relu', kernel_initializer=HeUniform()),
    layers.Dropout(0.2),

    # Fifth Dense layer (output layer)
    layers.Dense(train_generator.num_classes, activation='softmax')
])

# TASK 3 ANSWER: Count total layers
total_layers = len(model.layers)
print(f"\nTASK 3: Total number of CNN model layers: {total_layers}")

# Display model architecture
print("\nModel Architecture:")
model.summary()

# Verify layer counts
conv_layers = len([layer for layer in model.layers if isinstance(layer, layers.Conv2D)])
dense_layers = len([layer for layer in model.layers if isinstance(layer, layers.Dense)])
print(f"\nVerification:")
print(f"  Conv2D layers: {conv_layers} (Expected: 4)")
print(f"  Dense layers: {dense_layers} (Expected: 5)")


# =============================================================================
# TASK 4: Create and compile CNN with 4 Conv2D and 5 Dense layers
# (Already done above, but showing compilation here)
# =============================================================================

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\nModel compiled successfully!")
print(f"Optimizer: Adam (lr=0.001)")
print(f"Loss: categorical_crossentropy")
print(f"Metrics: accuracy")


# =============================================================================
# TASK 5: Define checkpoint callback with max accuracy monitoring
# =============================================================================

# Define model checkpoint to save best model based on validation accuracy
checkpoint_path = 'keras_cnn_best_model.keras'

checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_path,
    monitor='val_accuracy',      # Monitor validation accuracy
    mode='max',                   # Save when validation accuracy increases
    save_best_only=True,         # Only save when val_accuracy improves
    verbose=1,                    # Print message when saving
    save_weights_only=False      # Save complete model
)

print(f"\nCheckpoint callback created:")
print(f"  Filepath: {checkpoint_path}")
print(f"  Monitor: val_accuracy")
print(f"  Mode: max")
print(f"  Save best only: True")


# =============================================================================
# TRAINING THE MODEL
# =============================================================================

# Train the model (reduced epochs for demonstration)
epochs = 5  # Increase to 15-20 for full training

print(f"\nTraining model for {epochs} epochs...")
print(f"This may take 20-30 minutes on CPU...")

history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator,
    callbacks=[checkpoint_callback],
    verbose=1
)

print("\nTraining completed!")


# =============================================================================
# TASK 6: Plot training and validation loss
# =============================================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot accuracy
axes[0].plot(history.history['accuracy'], label='Training Accuracy', marker='o')
axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy', marker='s')
axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot loss (TASK 6 ANSWER)
axes[1].plot(history.history['loss'], label='Training Loss', marker='o', color='coral')
axes[1].plot(history.history['val_loss'], label='Validation Loss', marker='s', color='dodgerblue')
axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('keras_cnn_training_history.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nTraining history plots saved as 'keras_cnn_training_history.png'")


# =============================================================================
# BONUS: Evaluate model on validation set
# =============================================================================

print("\nEvaluating model on validation set...")
val_loss, val_accuracy = model.evaluate(validation_generator, verbose=1)

print(f"\nFinal Validation Results:")
print(f"  Loss: {val_loss:.4f}")
print(f"  Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")


# =============================================================================
# BONUS: Make predictions and show confusion matrix
# =============================================================================

from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Reset generator
validation_generator.reset()

# Get predictions
predictions = model.predict(validation_generator, verbose=1)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = validation_generator.classes
class_labels = list(validation_generator.class_indices.keys())

# Confusion matrix
cm = confusion_matrix(true_classes, predicted_classes)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_labels, yticklabels=class_labels)
plt.title('Confusion Matrix - Keras CNN', fontsize=14, fontweight='bold')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('keras_cnn_confusion_matrix.png', dpi=300)
plt.show()

# Classification report
print("\nClassification Report:")
print(classification_report(true_classes, predicted_classes,
                          target_names=class_labels, digits=4))


# =============================================================================
# SUMMARY OF SOLUTIONS
# =============================================================================

print("\n" + "="*70)
print("SUMMARY: Module 2 Lab 1 - All Tasks Completed")
print("="*70)
print(f"Task 1: Created fnames list with {len(fnames)} image files")
print(f"Task 2: Created validation_generator with {validation_generator.samples} samples")
print(f"Task 3: Total CNN layers = {total_layers}")
print(f"Task 4: Built CNN with {conv_layers} Conv2D + {dense_layers} Dense layers")
print(f"Task 5: Created checkpoint callback monitoring val_accuracy (max)")
print(f"Task 6: Plotted training and validation loss curves")
print(f"\nFinal Validation Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
print("="*70)
