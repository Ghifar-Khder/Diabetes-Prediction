import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
import time

# Create directory for saved models if it doesn't exist
os.makedirs('saved-models', exist_ok=True)

# Load and prepare data
train_df = pd.read_csv('data/data-split/train1.csv')
val_df = pd.read_csv('data/data-split/train2.csv')

X_train = train_df.drop('Outcome', axis=1)
y_train = train_df['Outcome']
X_val = val_df.drop('Outcome', axis=1)
y_val = val_df['Outcome']

# Combine train and validation for grid search with predefined split
X_combined = np.vstack((X_train, X_val))
y_combined = np.hstack((y_train, y_val))

# Create a list where training data indices are -1 and validation data indices are 0
test_fold = np.array([-1] * len(X_train) + [0] * len(X_val))
ps = PredefinedSplit(test_fold)

# Define comprehensive parameter grid for KNN
param_grid = {
    'n_neighbors': list(range(1, 101)) + [125, 150, 200],  # Very wide range
    'weights': ['uniform', 'distance'],
    'p': [1, 2, 3, 4, 5],  # Multiple Minkowski distances
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
    'leaf_size': [5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
    'metric': ['minkowski', 'chebyshev', 'manhattan', 'euclidean', 'cosine']
}

# Create KNN classifier
knn = KNeighborsClassifier()

# Perform comprehensive grid search with predefined split
print("Performing comprehensive hyperparameter optimization for KNN...")
print(f"Testing {len(param_grid['n_neighbors']) * len(param_grid['weights']) * len(param_grid['p']) * len(param_grid['algorithm']) * len(param_grid['leaf_size']) * len(param_grid['metric'])} parameter combinations...")
start_time = time.time()

grid_search = GridSearchCV(
    knn, 
    param_grid, 
    cv=ps,
    scoring='accuracy',
    n_jobs=-1,  # Use all available cores
    verbose=2,   # More detailed output
    refit=True   # Refit the best model on the entire dataset
)

grid_search.fit(X_combined, y_combined)

end_time = time.time()
print(f"\nHyperparameter optimization completed in {end_time - start_time:.2f} seconds")

# Get the best parameters
best_params = grid_search.best_params_
print(f"\nBest parameters: {best_params}")
print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

# The best model is already refit on the entire training+validation data
best_knn = grid_search.best_estimator_

# But let's also train it on just the training data for proper evaluation
best_knn_train = KNeighborsClassifier(**best_params)
best_knn_train.fit(X_train, y_train)

# Make predictions
y_train_pred = best_knn_train.predict(X_train)
y_val_pred = best_knn_train.predict(X_val)

# Calculate metrics for training set
train_accuracy = accuracy_score(y_train, y_train_pred)
train_precision = precision_score(y_train, y_train_pred)
train_recall = recall_score(y_train, y_train_pred)
train_f1 = f1_score(y_train, y_train_pred)

# Calculate metrics for validation set
val_accuracy = accuracy_score(y_val, y_val_pred)
val_precision = precision_score(y_val, y_val_pred)
val_recall = recall_score(y_val, y_val_pred)
val_f1 = f1_score(y_val, y_val_pred)

# Simplified evaluation printout
print("\n" + "="*60)
print("KNN MODEL PERFORMANCE EVALUATION")
print("="*60)
print(f"{'METRIC':<15} {'TRAINING':<12} {'VALIDATION':<12}")
print(f"{'Accuracy':<15} {train_accuracy*100:<10.2f}% {val_accuracy*100:<10.2f}%")
print(f"{'Precision':<15} {train_precision*100:<10.2f}% {val_precision*100:<10.2f}%")
print(f"{'Recall':<15} {train_recall*100:<10.2f}% {val_recall*100:<10.2f}%")
print(f"{'F1 Score':<15} {train_f1*100:<10.2f}% {val_f1*100:<10.2f}%")
print("="*60)

# Save the model
import joblib
joblib.dump(best_knn_train, 'saved-models/KNN_model.pkl')
print("\nKNN model training complete. Saved as 'saved-models/KNN_model.pkl'")

# Display top 5 parameter combinations
results = pd.DataFrame(grid_search.cv_results_)
top_results = results.nlargest(5, 'mean_test_score')
print("\nTop 5 parameter combinations:")
for i, (idx, row) in enumerate(top_results.iterrows(), 1):
    print(f"{i}. Score: {row['mean_test_score']:.4f}, Params: {row['params']}")
