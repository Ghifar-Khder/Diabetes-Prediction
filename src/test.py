import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from tensorflow.keras.models import load_model
import os

# Load the processed test set
print("Loading processed test set...")
test_df = pd.read_csv('data/data-split/test_set_processed.csv')

# Prepare features and target
X_test = test_df.drop('Outcome', axis=1)
y_test = test_df['Outcome']

# Data is already scaled, so use directly
X_test_scaled = X_test.values

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

print("Getting base model predictions for test set...")
test_proba = get_base_model_predictions(X_test_scaled, models)

# Load the saved FCM cluster centers
center_final = np.load(f'{models_dir}/fcm_cluster_centers.npy')
print("Loaded FCM cluster centers")

# Combine original data with prediction probabilities for FCM
test_combined = np.hstack([X_test_scaled, test_proba.values])

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

# Get cluster memberships for test set using the final centers
test_memberships = get_cluster_memberships(test_combined, center_final)

# Prepare final meta-features for test set
X_meta_test = np.hstack([X_test_scaled, test_proba.values, test_memberships])

# Load the saved meta-learner
meta_learner_ann = load_model(f'{models_dir}/MetaLearner_ANN.keras')
print("Loaded ANN meta-learner")

# Make predictions on test set
y_test_pred_meta = (meta_learner_ann.predict(X_meta_test, verbose=0) > 0.5).astype(int).flatten()

# Calculate metrics for meta-learner
test_accuracy_meta = accuracy_score(y_test, y_test_pred_meta)
test_precision_meta = precision_score(y_test, y_test_pred_meta)
test_recall_meta = recall_score(y_test, y_test_pred_meta)
test_f1_meta = f1_score(y_test, y_test_pred_meta)

# Calculate metrics for base models on test set
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

# Evaluate base models on test set
base_test_results = {}
for name, model in models.items():
    base_test_results[name] = evaluate_base_model(model, X_test_scaled, y_test, name)

# Print evaluation results
print("\n" + "="*80)
print("MODEL PERFORMANCE EVALUATION ON TEST SET")
print("="*80)

# Base models on test set
print("\nBASE MODELS - TEST SET:")
print("-" * 60)
print(f"{'Model':<12} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10}")
print("-" * 60)
for name, results in base_test_results.items():
    print(f"{name:<12} {results['Accuracy']*100:<9.2f}% {results['Precision']*100:<9.2f}% {results['Recall']*100:<9.2f}% {results['F1']*100:<9.2f}%")

# Meta-learner results
print("\nANN META-LEARNER RESULTS ON TEST SET:")
print("-" * 60)
print(f"{'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
print("-" * 48)
print(f"{test_accuracy_meta*100:<11.2f}% {test_precision_meta*100:<11.2f}% {test_recall_meta*100:<11.2f}% {test_f1_meta*100:<11.2f}%")
print("="*80)

# Detailed classification report for meta-learner
print("\nDetailed Classification Report for Meta-Learner:")
print("-" * 60)
print(classification_report(y_test, y_test_pred_meta, target_names=['No Diabetes', 'Diabetes']))

# Confusion matrix for meta-learner
print("\nConfusion Matrix for Meta-Learner:")
print("-" * 60)
cm = confusion_matrix(y_test, y_test_pred_meta)
print(f"True Negatives: {cm[0,0]}")
print(f"False Positives: {cm[0,1]}")
print(f"False Negatives: {cm[1,0]}")
print(f"True Positives: {cm[1,1]}")

# Compare with best base model
best_base_model = max(base_test_results.items(), key=lambda x: x[1]['Accuracy'])
best_base_name, best_base_results = best_base_model

print(f"\nMeta-Learner vs Best Base Model ({best_base_name}):")
print("-" * 60)
print(f"{'Metric':<12} {'Meta-Learner':<15} {best_base_name:<15} {'Difference':<12}")
print("-" * 60)
for metric in ['Accuracy', 'Precision', 'Recall', 'F1']:
    meta_value = locals()[f'test_{metric.lower()}_meta'] * 100
    base_value = best_base_results[metric] * 100
    diff = meta_value - base_value
    print(f"{metric:<12} {meta_value:<15.2f}% {base_value:<15.2f}% {diff:>+10.2f}%")