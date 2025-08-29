import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from skfuzzy.cluster import cmeans
import os
import random
import tensorflow as tf




# Set random seeds
def set_random_seeds(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    # For GPU determinism (if using GPU)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

# Set all random seeds
set_random_seeds(42)

# Load datasets
print("Loading datasets...") # detect the error
train1_df = pd.read_csv('data/data-split/train1.csv')
train2_df = pd.read_csv('data/data-split/train2.csv')

# Prepare features and targets
X_train = train1_df.drop('Outcome', axis=1)
y_train = train1_df['Outcome']
X_val = train2_df.drop('Outcome', axis=1)
y_val = train2_df['Outcome']

# Data is already scaled, so use directly
X_train_scaled = X_train.values
X_val_scaled = X_val.values

# Load base models
models_dir = 'saved-models'
models = {
    'ANN': load_model(f'{models_dir}/ANN.keras'),
    'KNN': joblib.load(f'{models_dir}/KNN.pkl'),
    'LightGBM': joblib.load(f'{models_dir}/LightGBM.pkl'),
    'RSVM': joblib.load(f'{models_dir}/RSVM.pkl'),
    'LinearSVM': joblib.load(f'{models_dir}/LinearSVM.pkl'),
}

# Get prediction probabilities from base models
def get_base_model_predictions(X, models):
    predictions = {}
    
    for name, model in models.items():
        if name == 'ANN':
            pred_proba = model.predict(X, verbose=0)
            pred_proba = pred_proba[:, 1] if pred_proba.shape[1] > 1 else pred_proba.flatten()
        else:
            if hasattr(model, 'predict_proba'):
                pred_proba = model.predict_proba(X)[:, 1]
            else:
                pred_proba = model.decision_function(X)
                pred_proba = (pred_proba - pred_proba.min()) / (pred_proba.max() - pred_proba.min() + 1e-8)
        
        predictions[name] = pred_proba
    
    return pd.DataFrame(predictions)

print("Getting base model predictions...")
train_proba = get_base_model_predictions(X_train_scaled, models)
val_proba = get_base_model_predictions(X_val_scaled, models)

# Combine original data with prediction probabilities for FCM
train_combined = np.hstack([X_train_scaled, train_proba.values])
val_combined = np.hstack([X_val_scaled, val_proba.values])

# Get cluster membership function
def get_cluster_memberships(combined_data, centers):
    memberships = []
    m = 2
    
    for point in combined_data:
        distances = [np.linalg.norm(point - center) for center in centers]
        
        if any(d == 0 for d in distances):
            u = [1.0 if d == 0 else 0.0 for d in distances]
        else:
            u = []
            for i in range(len(centers)):
                denominator = sum([(distances[i] / distances[j]) ** (2 / (m - 1)) 
                                  for j in range(len(centers))])
                u.append(1.0 / denominator)
        
        memberships.append(u)
    
    return np.array(memberships)

# Phase 1: Calculate FCM cluster centers using only train1
print("Phase 1: Calculating FCM cluster centers using train1 only...")
train_combined_matrix = train_combined.T

# Set a fixed initial state for FCM for reproducibility
np.random.seed(42)
initial_u = np.random.random((2, train_combined_matrix.shape[1]))
initial_u = initial_u / np.sum(initial_u, axis=0)

center_phase1, u, u0, d, jm, p, fpc = cmeans(
    train_combined_matrix, 
    c=2, 
    m=5, 
    error=0.0001, 
    maxiter=1000, 
    init=initial_u  # Use the fixed initial state
)

# Get cluster membership for training and validation sets using phase1 centers
train_memberships_phase1 = get_cluster_memberships(train_combined, center_phase1)
val_memberships_phase1 = get_cluster_memberships(val_combined, center_phase1)

# Prepare meta-features for ANN meta-learner for phase 1
X_meta_train_phase1 = np.hstack([X_train_scaled, train_proba.values, train_memberships_phase1])
X_meta_val_phase1 = np.hstack([X_val_scaled, val_proba.values, val_memberships_phase1])

# Create ANN meta-learner
print("Creating ANN meta-learner...")

def create_meta_learner(input_dim):
    model = Sequential([
        Dense(512, activation='relu', input_dim=input_dim, kernel_regularizer=l2(0.05)),
        Dense(512, activation='relu', kernel_regularizer=l2(0.01)),
        Dense(256, activation='relu', kernel_regularizer=l2(0.005)),
        Dense(128, activation='relu'),
        Dense(128, activation='relu'),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=Adam(),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

meta_learner_ann = create_meta_learner(X_meta_train_phase1.shape[1])

# Phase 1 training process
print("Starting phase 1 training process...")
print("Phase 1: Training on training data...")
checkpoint_path = f'{models_dir}/phase1_weights.weights.h5'

# Use a fixed seed for shuffling
history_phase1 = meta_learner_ann.fit(
    X_meta_train_phase1, y_train,
    validation_data=(X_meta_val_phase1, y_val),
    epochs=20,
    batch_size=8,
    verbose=1,
    shuffle=False  # Disable shuffling for reproducibility
)

# Save the weights after phase 1
meta_learner_ann.save_weights(checkpoint_path)
print("Saved weights after phase 1")

# Phase 2: Recalculate FCM centers with both datasets (2:1 weight for train2)
print("Phase 2: Recalculating FCM centers with weighted data (2:1 for train2)...")

# Create weighted dataset (duplicate train2 samples to give them 2x weight)
weighted_train_combined = np.vstack([train_combined, val_combined, val_combined])
weighted_train_combined_matrix = weighted_train_combined.T

# Set a fixed initial state for FCM for reproducibility
np.random.seed(42)
initial_u = np.random.random((2, weighted_train_combined_matrix.shape[1]))
initial_u = initial_u / np.sum(initial_u, axis=0)

center_final, u, u0, d, jm, p, fpc = cmeans(
    weighted_train_combined_matrix, 
    c=2, 
    m=2, 
    error=0.00000001, 
    maxiter=1000, 
    init=initial_u  # Use fixed initial state
)

# Get final cluster memberships using the weighted centers
train_memberships_final = get_cluster_memberships(train_combined, center_final)
val_memberships_final = get_cluster_memberships(val_combined, center_final)

# Prepare final meta-features for phase 2 training
X_meta_train_final = np.hstack([X_train_scaled, train_proba.values, train_memberships_final])
X_meta_val_final = np.hstack([X_val_scaled, val_proba.values, val_memberships_final])

# Phase 2: Continue training on combined data with weighted loss (5:1 ratio favoring validation set)
print("Phase 2: Fine-tuning on combined data with weighted loss...")
meta_learner_ann.load_weights(checkpoint_path)

# Combine training and validation data
X_combined = np.vstack([X_meta_train_final, X_meta_val_final])
y_combined = np.concatenate([y_train, y_val])

# Create sample weights with 5:1 ratio (validation samples get 5x weight)
sample_weights = np.ones(len(X_combined))
sample_weights[len(X_meta_train_final):] = 5.0  # Give validation samples 5x weight

history_phase2 = meta_learner_ann.fit(
    X_combined, y_combined,
    sample_weight=sample_weights,
    epochs=15,
    batch_size=4,
    verbose=1,
    shuffle=False,  # Disable shuffling for reproducibility
    validation_data=(X_meta_val_final, y_val)  # Keep validation on just the validation set to see the difference
)

# Make predictions with meta-learner on both sets
y_train_pred_meta = (meta_learner_ann.predict(X_meta_train_final, verbose=0) > 0.5).astype(int).flatten()
y_val_pred_meta = (meta_learner_ann.predict(X_meta_val_final, verbose=0) > 0.5).astype(int).flatten()

# Calculate metrics for meta-learner
train_accuracy_meta = accuracy_score(y_train, y_train_pred_meta)
train_precision_meta = precision_score(y_train, y_train_pred_meta)
train_recall_meta = recall_score(y_train, y_train_pred_meta)
train_f1_meta = f1_score(y_train, y_train_pred_meta)

val_accuracy_meta = accuracy_score(y_val, y_val_pred_meta)
val_precision_meta = precision_score(y_val, y_val_pred_meta)
val_recall_meta = recall_score(y_val, y_val_pred_meta)
val_f1_meta = f1_score(y_val, y_val_pred_meta)

# Calculate metrics for base models
def evaluate_base_model(model, X, y, name):
    if name == 'ANN':
        y_pred = (model.predict(X, verbose=0) > 0.5).astype(int).flatten()
    else:
        y_pred = model.predict(X)
    
    return {
        'Accuracy': accuracy_score(y, y_pred),
        'Precision': precision_score(y, y_pred),
        'Recall': recall_score(y, y_pred),
        'F1': f1_score(y, y_pred)
    }

# Evaluate base models on validation set
base_val_results = {}
for name, model in models.items():
    base_val_results[name] = evaluate_base_model(model, X_val_scaled, y_val, name)

# Print evaluation results
print("\n" + "="*80)
print("MODEL PERFORMANCE EVALUATION")
print("="*80)

# Base models on validation set
print("\nBASE MODELS - VALIDATION SET:")
print("-" * 60)
print(f"{'Model':<12} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10}")
print("-" * 60)
for name, results in base_val_results.items():
    print(f"{name:<12} {results['Accuracy']*100:<9.2f}% {results['Precision']*100:<9.2f}% {results['Recall']*100:<9.2f}% {results['F1']*100:<9.2f}%")

# Meta-learner results
print("\nANN META-LEARNER RESULTS (Two-phase Training with Weighted FCM):")
print("-" * 60)
print(f"{'Set':<12} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10}")
print("-" * 60)
print(f"{'Training':<12} {train_accuracy_meta*100:<9.2f}% {train_precision_meta*100:<9.2f}% {train_recall_meta*100:<9.2f}% {train_f1_meta*100:<9.2f}%")
print(f"{'Validation':<12} {val_accuracy_meta*100:<9.2f}% {val_precision_meta*100:<9.2f}% {val_recall_meta*100:<9.2f}% {val_f1_meta*100:<9.2f}%")
print("="*80)

# Save the meta-learner
os.makedirs(models_dir, exist_ok=True)
meta_learner_ann.save(f'{models_dir}/MetaLearner_ANN.keras')
print(f"\nANN meta-learner saved as '{models_dir}/MetaLearner_ANN.keras'")

# Save the final FCM cluster centers for future use (weighted 2:1 for train2)
np.save(f'{models_dir}/fcm_cluster_centers.npy', center_final)
print(f"FCM cluster centers (weighted 2:1 for train2) saved as '{models_dir}/fcm_cluster_centers.npy'")

# Clean up temporary weights file
if os.path.exists(checkpoint_path):
    os.remove(checkpoint_path)