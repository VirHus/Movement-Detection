import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam

# Define paths and hyperparameters
data_dir = "HMDB51"  
image_size = (150, 150)  
batch_size = 32
epochs = 10  

# Data Augmentation for better generalization
datagen = ImageDataGenerator(
    rescale=1.0/255,  # Normalize pixel values to [0, 1]
    validation_split=0.2,  # Split the data into training and validation sets
    rotation_range=40,  # Randomly rotate images
    width_shift_range=0.2,  # Randomly shift images horizontally
    height_shift_range=0.2,  # Randomly shift images vertically
    shear_range=0.2,  # Apply shear transformations
    zoom_range=0.2,  # Apply random zoom
    horizontal_flip=True,  # Randomly flip images horizontally
    fill_mode='nearest'  # Fill missing pixels after transformations
)

# Create training and validation data generators
train_data = datagen.flow_from_directory(
    data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'  # Set this as training data
)

val_data = datagen.flow_from_directory(
    data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'  # Set this as validation data
)

# Build the model using Transfer Learning with VGG16
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(image_size[0], image_size[1], 3))
base_model.trainable = False  # Freeze the VGG16 convolutional base layers

# Define the custom classification head on top of the VGG16 base model
model = Sequential([
    base_model,  # Pre-trained VGG16 layers (frozen)
    Flatten(),  # Flatten the 3D outputs from the convolutional base to 1D
    Dense(128, activation='relu'),  # Fully connected layer with ReLU activation
    Dropout(0.5),  # Dropout regularization to prevent overfitting
    Dense(len(train_data.class_indices), activation='softmax')  # Output layer for multi-class classification
])

# Compile the model with Adam optimizer and categorical cross-entropy loss
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_data,  # Training data
    validation_data=val_data,  # Validation data
    epochs=epochs,  # Number of epochs
)

# Save the trained model and class labels for future use
model.save("HMDB51_with_vgg16_trained_model.h5")
np.save('classes.npy', list(train_data.class_indices.keys()))  # Save class labels
