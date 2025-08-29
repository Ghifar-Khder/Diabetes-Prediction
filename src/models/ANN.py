import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import os
import random

# random seeds for reproducibility
os.environ['PYTHONHASHSEED'] = '42'
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# Create directory for saved models if it doesn't exist
os.makedirs('saved-models', exist_ok=True)

# Load and prepare data
train1_df = pd.read_csv('data/data-split/train1.csv')
train2_df = pd.read_csv('data/data-split/train2.csv')

X_train = train1_df.drop('Outcome', axis=1)
y_train = train1_df['Outcome']
X_val = train2_df.drop('Outcome', axis=1)
y_val = train2_df['Outcome']

# Calculate class weights
class_counts = np.bincount(y_train)
total = len(y_train)
weight_for_0 = (1 / class_counts[0]) * (total / 2.0)
weight_for_1 = (1 / class_counts[1]) * (total / 2.0) * 1.5  
class_weight = {0: weight_for_0, 1: weight_for_1}

print(f"Class 0 weight: {weight_for_0:.2f}")
print(f"Class 1 weight: {weight_for_1:.2f}")

# Build the model layer by layer
model = Sequential()

# Input layer
model.add(Dense(32, activation='relu', kernel_regularizer=l2(0.0022), input_shape=(X_train.shape[1],)))
model.add(BatchNormalization())
model.add(Dropout(0.4))

# First hidden layer
model.add(Dense(256, activation='relu', kernel_regularizer=l2(0.0022)))
model.add(BatchNormalization())
model.add(Dropout(0.4))

# Second hidden layer
model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.0022)))
model.add(BatchNormalization())
model.add(Dropout(0.4))

# Third hidden layer
model.add(Dense(256, activation='relu', kernel_regularizer=l2(0.0022)))
model.add(BatchNormalization())

# Output layer
model.add(Dense(1, activation='sigmoid'))

# Compile the model
optimizer = Adam(learning_rate=0.005651235488796088)
model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy', 
    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

# Display model architecture
model.summary()

# Define callbacks
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    filepath='saved-models/ANN.keras',  
    monitor='val_accuracy',
    save_best_only=True,
    mode='max'
)

# Train the model with class weights
print("Training model with class weights...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=250,
    batch_size=8,
    class_weight=class_weight,  # Add class weights here
    callbacks=[early_stopping, checkpoint],
    verbose=1
)

# Load the best saved model
best_model = tf.keras.models.load_model('saved-models/ANN.keras')

# Evaluate on both training and validation sets
train_loss, train_accuracy, train_precision, train_recall = best_model.evaluate(X_train, y_train, verbose=0)
val_loss, val_accuracy, val_precision, val_recall = best_model.evaluate(X_val, y_val, verbose=0)

# Calculate F1 scores
train_f1 = 2 * (train_precision * train_recall) / (train_precision + train_recall + 1e-7)
val_f1 = 2 * (val_precision * val_recall) / (val_precision + val_recall + 1e-7)

# Simplified evaluation printout
print("\n" + "="*50)
print("MODEL PERFORMANCE EVALUATION")
print("="*50)
print(f"{'METRIC':<15} {'TRAINING':<12} {'VALIDATION':<12}")
print(f"{'Accuracy':<15} {train_accuracy*100:<10.2f}% {val_accuracy*100:<10.2f}%")
print(f"{'Loss':<15} {train_loss:<10.4f} {val_loss:<10.4f}")
print(f"{'Precision':<15} {train_precision*100:<10.2f}% {val_precision*100:<10.2f}%")
print(f"{'Recall':<15} {train_recall*100:<10.2f}% {val_recall*100:<10.2f}%")
print(f"{'F1 Score':<15} {train_f1*100:<10.2f}% {val_f1*100:<10.2f}%")
print("="*50)

print("\nModel training complete. Saved as 'saved-models/ANN.keras'")