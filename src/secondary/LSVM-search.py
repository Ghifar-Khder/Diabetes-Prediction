import pandas as pd
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
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

# Define optimized parameter grid for Linear SVM to avoid convergence issues
param_grid = {
    'C': np.logspace(-3, 3, 20),
    'loss': ['hinge','squared_hinge'], 
    'penalty': ['l2'],
    'dual': [True], 
    'tol': np.logspace(-3, -3, 20),
    'max_iter': [ 20000 , 30000 ],
    'fit_intercept': [True, False],
}

# Create Linear SVM classifier
svm = LinearSVC(random_state=42)

# Perform grid search with predefined split
print("Performing hyperparameter optimization for Linear SVM...")
print(f"Testing approximately {len(param_grid['C']) * len(param_grid['loss']) * len(param_grid['penalty']) * len(param_grid['dual']) * len(param_grid['tol']) * len(param_grid['max_iter']) * len(param_grid['fit_intercept'])} parameter combinations...")
start_time = time.time()

grid_search = GridSearchCV(
    svm, 
    param_grid, 
    cv=ps,
    scoring='accuracy',
    n_jobs=-1,
    verbose=2,
    refit=True
)

grid_search.fit(X_combined, y_combined)

end_time = time.time()
print(f"\nLinear SVM hyperparameter optimization completed in {end_time - start_time:.2f} seconds")

# Get the best parameters
best_params = grid_search.best_params_
print(f"\nBest parameters: {best_params}")
print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

# The best model is already refit on the entire training+validation data
best_svm = grid_search.best_estimator_

# But let's also train it on just the training data for proper evaluation
best_svm_train = LinearSVC(**best_params, random_state=42)
best_svm_train.fit(X_train, y_train)

# Make predictions
y_train_pred = best_svm_train.predict(X_train)
y_val_pred = best_svm_train.predict(X_val)

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
print("LINEAR SVM MODEL PERFORMANCE EVALUATION")
print("="*60)
print(f"{'METRIC':<15} {'TRAINING':<12} {'VALIDATION':<12}")
print(f"{'Accuracy':<15} {train_accuracy*100:<10.2f}% {val_accuracy*100:<10.2f}%")
print(f"{'Precision':<15} {train_precision*100:<10.2f}% {val_precision*100:<10.2f}%")
print(f"{'Recall':<15} {train_recall*100:<10.2f}% {val_recall*100:<10.2f}%")
print(f"{'F1 Score':<15} {train_f1*100:<10.2f}% {val_f1*100:<10.2f}%")
print("="*60)

# Save the model
joblib.dump(best_svm_train, 'saved-models/LinearSVM_model.pkl')
print("\nLinear SVM model training complete. Saved as 'saved-models/LinearSVM_model.pkl'")

# Display top 5 parameter combinations
results = pd.DataFrame(grid_search.cv_results_)
top_results = results.nlargest(5, 'mean_test_score')
print("\nTop 5 parameter combinations:")
for i, (idx, row) in enumerate(top_results.iterrows(), 1):
    print(f"{i}. Score: {row['mean_test_score']:.4f}, Params: {row['params']}")
