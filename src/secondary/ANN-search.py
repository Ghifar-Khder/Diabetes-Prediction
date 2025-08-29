import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, InputLayer
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import StandardScaler
import keras_tuner as kt
import shutil
import os
# Load and prepare data
train_df = pd.read_csv('data/data-split/train1.csv')
val_df = pd.read_csv('data/data-split/train2.csv')

X_train = train_df.drop('Outcome', axis=1)
y_train = train_df['Outcome']
X_val = val_df.drop('Outcome', axis=1)
y_val = val_df['Outcome']

# Hyperparameter tuning
def build_model(hp):
    model = Sequential()
    
    # Input layer with tunable parameters 
    model.add(InputLayer(shape=(X_train.shape[1],))) 

    model.add(Dense(
        units=hp.Int('units_input', min_value=32, max_value=512, step=32),
        activation=hp.Choice('activation', ['relu', 'tanh']),
        kernel_regularizer=l2(hp.Float('l2_reg', 0.001, 0.1))
    ))
    model.add(BatchNormalization())
    model.add(Dropout(hp.Float('dropout', 0.2, 0.5)))
    
    # number of hidden layers
    for i in range(hp.Int('n_layers', 1, 5)):
        model.add(Dense(
            units=hp.Int(f'units_{i}', min_value=16, max_value=256, step=16),
            activation=hp.Choice('activation', ['relu', 'tanh'])
        ))
        model.add(BatchNormalization())
        model.add(Dropout(hp.Float('dropout', 0.2, 0.5)))
    
    # Output layer
    model.add(Dense(1, activation='sigmoid'))  
    
    # tunable learning rate
    optimizer = Adam(learning_rate=hp.Float('lr', 1e-4, 1e-2, sampling='log'))
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy', 
        metrics=['accuracy']
    )
    return model

# Clean up previous tuning directory to ensure fresh start
if os.path.exists('tuning_dir'):
    shutil.rmtree('tuning_dir')

# Set up tuner with overwrite=True to ensure fresh search
tuner = kt.RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=30,
    executions_per_trial=1,
    directory='tuning_dir',
    project_name='ann_tuning',
    overwrite=True  # ensures a fresh search each time
)

# Perform hyperparameter search
print("Starting hyperparameter search...")
tuner.search(X_train, y_train, epochs=100, validation_data=(X_val, y_val), verbose=1)

# Get best hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print("Best hyperparameters found:")
print(best_hps.values)

# Build model with best hyperparameters
model = tuner.hypermodel.build(best_hps)

# Define callbacks for training
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    filepath='final_ann_model.keras',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max'
)

# Train the model
print("Training model with best hyperparameters...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=150,
    batch_size=16,
    callbacks=[early_stopping, checkpoint],
    verbose=1
)

# Load the best saved model
best_model = tf.keras.models.load_model('final_ann_model.keras')

# Evaluate on validation set
val_loss, val_accuracy = best_model.evaluate(X_val, y_val, verbose=0)
print(f"\nBest Model Validation Accuracy: {val_accuracy * 100:.2f}%")

print("Model training complete. Saved as 'final_ann_model.keras'")