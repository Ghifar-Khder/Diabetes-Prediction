import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import os

# Create directory for saved models if it doesn't exist
os.makedirs('saved-models', exist_ok=True)

# Load and prepare data with the new dataset paths
train_df = pd.read_csv('data/data-split/train1.csv')
val_df = pd.read_csv('data/data-split/train2.csv')

X_train = train_df.drop('Outcome', axis=1)
y_train = train_df['Outcome']
X_val = val_df.drop('Outcome', axis=1)
y_val = val_df['Outcome']

# best parameters from optimization
best_params = {
    'colsample_bytree': 1.0,
    'learning_rate': 0.01,
    'max_depth': 7,
    'min_child_samples': 3,
    'n_estimators': 400,
    'num_leaves': 25,
    'reg_alpha': 2,
    'reg_lambda': 2,
    'subsample': 0.7
}

# Create and train with best parameters
print("Training LightGBM with optimized hyperparameters...")
lgbm_model = lgb.LGBMClassifier(random_state=42, verbose=-1, **best_params)
lgbm_model.fit(X_train, y_train)

# Make predictions
y_train_pred = lgbm_model.predict(X_train)
y_val_pred = lgbm_model.predict(X_val)

# Calculate metrics
train_accuracy = accuracy_score(y_train, y_train_pred)
train_precision = precision_score(y_train, y_train_pred)
train_recall = recall_score(y_train, y_train_pred)
train_f1 = f1_score(y_train, y_train_pred)

val_accuracy = accuracy_score(y_val, y_val_pred)
val_precision = precision_score(y_val, y_val_pred)
val_recall = recall_score(y_val, y_val_pred)
val_f1 = f1_score(y_val, y_val_pred)

# Print evaluation
print("\n" + "="*60)
print("LIGHTGBM MODEL PERFORMANCE EVALUATION")
print("="*60)
print(f"{'METRIC':<15} {'TRAINING':<12} {'VALIDATION':<12}")
print(f"{'Accuracy':<15} {train_accuracy*100:<10.2f}% {val_accuracy*100:<10.2f}%")
print(f"{'Precision':<15} {train_precision*100:<10.2f}% {val_precision*100:<10.2f}%")
print(f"{'Recall':<15} {train_recall*100:<10.2f}% {val_recall*100:<10.2f}%")
print(f"{'F1 Score':<15} {train_f1*100:<10.2f}% {val_f1*100:<10.2f}%")
print("="*60)

# Save the model
joblib.dump(lgbm_model, 'saved-models/LightGBM.pkl')
print("\nOptimized LightGBM model saved as 'saved-models/LightGBM.pkl'")
