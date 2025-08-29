import pandas as pd
import numpy as np
from sklearn.svm import SVC
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

# Use the best parameters from optimization
best_params = {
    'C': 8.531678524172806,
    'gamma': 0.6951927961775606,
    'max_iter': 500,
    'tol': 0.1
}

# Create and train RBF SVM classifier with best parameters
print("Training RSVM with optimized hyperparameters...")
svm_model = SVC(kernel='rbf', random_state=42, **best_params)
svm_model.fit(X_train, y_train)

# Make predictions
y_train_pred = svm_model.predict(X_train)
y_val_pred = svm_model.predict(X_val)

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
print("RSVM MODEL PERFORMANCE EVALUATION")
print("="*60)
print(f"{'METRIC':<15} {'TRAINING':<12} {'VALIDATION':<12}")
print(f"{'Accuracy':<15} {train_accuracy*100:<10.2f}% {val_accuracy*100:<10.2f}%")
print(f"{'Precision':<15} {train_precision*100:<10.2f}% {val_precision*100:<10.2f}%")
print(f"{'Recall':<15} {train_recall*100:<10.2f}% {val_recall*100:<10.2f}%")
print(f"{'F1 Score':<15} {train_f1*100:<10.2f}% {val_f1*100:<10.2f}%")
print("="*60)

# Save the model
joblib.dump(svm_model, 'saved-models/RSVM.pkl')
print("\nOptimized RBF SVM model saved as 'saved-models/RSVM.pkl'")