"""
Model training script for the Smart Waste Sorting System.

This script demonstrates how to train the MobileNetV2-based model
on the TrashNet dataset or similar waste classification datasets.

Note: This script requires training data to be available.
For production use, you would need to:
1. Download and prepare the TrashNet dataset
2. Organize images into appropriate folders
3. Run this script to train the model
4. Save the trained model for use in the Flask app
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt

def create_model(num_classes=3, input_shape=(224, 224, 3)):
    """Create MobileNetV2-based model for waste classification."""
    
    # Load pre-trained MobileNetV2 without top layers
    base_model = MobileNetV2(input_shape=input_shape,
                           include_top=False,
                           weights='imagenet')
    
    # Freeze the base model initially
    base_model.trainable = False
    
    # Add custom classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    # Create the complete model
    model = Model(inputs=base_model.input, outputs=predictions)
    
    return model, base_model

def prepare_data_generators(data_dir, batch_size=32, img_size=(224, 224)):
    """Prepare data generators for training and validation."""
    
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        validation_split=0.2
    )
    
    # Only rescaling for validation
    validation_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )
    
    # Create generators
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )
    
    validation_generator = validation_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )
    
    return train_generator, validation_generator

def train_model(data_dir='data', model_save_path='models/waste_classifier.h5'):
    """Train the waste classification model."""
    
    print("Creating model...")
    model, base_model = create_model()
    
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("Model architecture:")
    model.summary()
    
    # Check if data directory exists
    if not os.path.exists(data_dir):
        print(f"Error: Data directory '{data_dir}' not found.")
        print("Please organize your training data as follows:")
        print(f"{data_dir}/")
        print("  biodegradable/")
        print("    image1.jpg")
        print("    image2.jpg")
        print("  recyclable/")
        print("    image1.jpg")
        print("    image2.jpg") 
        print("  landfill/")
        print("    image1.jpg")
        print("    image2.jpg")
        return
    
    print("Preparing data generators...")
    train_generator, validation_generator = prepare_data_generators(data_dir)
    
    # Print class indices
    print("Class indices:", train_generator.class_indices)
    
    # Set up callbacks
    callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True),
        ModelCheckpoint(model_save_path, save_best_only=True),
        ReduceLROnPlateau(factor=0.2, patience=5)
    ]
    
    # Train the model (initial training with frozen base)
    print("Starting initial training with frozen base model...")
    history = model.fit(
        train_generator,
        epochs=20,
        validation_data=validation_generator,
        callbacks=callbacks,
        verbose=1
    )
    
    # Fine-tuning: unfreeze the base model
    print("Fine-tuning: unfreezing base model...")
    base_model.trainable = True
    
    # Use a lower learning rate for fine-tuning
    model.compile(
        optimizer=Adam(learning_rate=0.0001/10),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Continue training with unfrozen base
    history_fine = model.fit(
        train_generator,
        epochs=10,
        validation_data=validation_generator,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save the final model
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")
    
    # Plot training history
    plot_training_history(history, history_fine)
    
    return model

def plot_training_history(history, history_fine=None):
    """Plot training and validation accuracy/loss."""
    
    plt.figure(figsize=(12, 4))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    
    if history_fine:
        epochs_base = len(history.history['accuracy'])
        plt.plot(range(epochs_base, epochs_base + len(history_fine.history['accuracy'])),
                history_fine.history['accuracy'], label='Fine-tuning Training Accuracy')
        plt.plot(range(epochs_base, epochs_base + len(history_fine.history['val_accuracy'])),
                history_fine.history['val_accuracy'], label='Fine-tuning Validation Accuracy')
    
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    
    if history_fine:
        epochs_base = len(history.history['loss'])
        plt.plot(range(epochs_base, epochs_base + len(history_fine.history['loss'])),
                history_fine.history['loss'], label='Fine-tuning Training Loss')
        plt.plot(range(epochs_base, epochs_base + len(history_fine.history['val_loss'])),
                history_fine.history['val_loss'], label='Fine-tuning Validation Loss')
    
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

def evaluate_model(model_path='models/waste_classifier.h5', data_dir='data'):
    """Evaluate the trained model on test data."""
    
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found.")
        return
    
    print("Loading model...")
    model = keras.models.load_model(model_path)
    
    print("Preparing test data...")
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    test_generator = test_datagen.flow_from_directory(
        data_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        shuffle=False
    )
    
    print("Evaluating model...")
    loss, accuracy = model.evaluate(test_generator)
    
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    
    return loss, accuracy

if __name__ == "__main__":
    print("Smart Waste Sorting System - Model Training")
    print("=" * 50)
    
    # Check for GPU availability
    if tf.config.list_physical_devices('GPU'):
        print("GPU is available for training")
    else:
        print("Training on CPU (GPU not available)")
    
    # Example usage
    print("\nTo train the model, you need to:")
    print("1. Prepare your dataset in the following structure:")
    print("   data/")
    print("     biodegradable/")
    print("     recyclable/")
    print("     landfill/")
    print("2. Place training images in respective folders")
    print("3. Run: python train_model.py")
    
    # Uncomment the following lines to actually train the model
    # (requires training data to be available)
    
    # train_model()
    # evaluate_model()
    
    print("\nNote: This is a template training script.")
    print("Actual training requires the TrashNet dataset or similar data.")
    print("The Flask app will work with a rule-based classifier until a trained model is available.")
