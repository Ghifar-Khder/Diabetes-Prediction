import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
import joblib
import os
import json

# Create directories for saving models and scaler
os.makedirs('saved-models/preprocessors', exist_ok=True)
os.makedirs('data/data-split', exist_ok=True)

# Load the training and test datasets
train_df = pd.read_csv('data/data-split/main/train_set.csv')
test_df = pd.read_csv('data/data-split/main/test_set.csv')

# Display initial info about missing values
print("Initial missing values in training set:")
print(train_df.isnull().sum())
print("\nInitial missing values in test set:")
print(test_df.isnull().sum())

# 1. Remove outliers from both datasets using IQR method
print("\n=== OUTLIER REMOVAL PHASE ===")

def remove_outliers_iqr(df, bounds_dict):
    """
    Remove outliers from dataframe using precomputed bounds
    """
    df_clean = df.copy()
    outliers_indices = set()
    
    for col, bounds in bounds_dict.items():
        if col in df_clean.columns:
            lower_bound = bounds['lower_bound']
            upper_bound = bounds['upper_bound']
            col_outliers = df_clean[
                (df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)
            ].index
            outliers_indices.update(col_outliers)
    
    df_clean = df_clean.drop(index=outliers_indices)
    return df_clean

# Compute outlier bounds from training data
outlier_columns = train_df.columns.difference(['Outcome'])
outlier_bounds = {}

for col in outlier_columns:
    Q1 = train_df[col].quantile(0.25)
    Q3 = train_df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outlier_bounds[col] = {
        'Q1': float(Q1),
        'Q3': float(Q3),
        'IQR': float(IQR),
        'lower_bound': float(lower_bound),
        'upper_bound': float(upper_bound)
    }

# Remove outliers from both datasets
train_clean = remove_outliers_iqr(train_df, outlier_bounds)
test_clean = remove_outliers_iqr(test_df, outlier_bounds)

print(f"Training set shape after outlier removal: {train_clean.shape}")
print(f"Test set shape after outlier removal: {test_clean.shape}")

# Save outlier bounds for future use
with open('saved-models/preprocessors/outlier_bounds.json', 'w') as f:
    json.dump(outlier_bounds, f, indent=4)
print("Outlier bounds saved to saved-models/preprocessors/outlier_bounds.json")

# 2. Hyperparameter tuning for imputers using known test data
print("\n=== HYPERPARAMETER TUNING FOR IMPUTERS ===")

# Define parameter grids for grid search
skin_param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 4, 5],
    'subsample': [0.8, 0.9, 1.0],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 5]
}

insulin_param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.05, 0.1, 0.2, 0.3],
    'max_depth': [4, 5, 6],
    'subsample': [0.5, 0.7, 0.9],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 5]
}

# 2.1 Tune SkinThickness imputer
print("\nTuning SkinThickness imputer...")
skin_features = ['Age', 'BMI', 'BloodPressure', 'Pregnancies']
skin_target = 'SkinThickness'

# Prepare training data (known values only)
skin_train_known = train_clean.dropna(subset=[skin_target])
X_skin_train = skin_train_known[skin_features]
y_skin_train = skin_train_known[skin_target]

# Prepare test data (known values only for evaluation)
skin_test_known = test_clean.dropna(subset=[skin_target])
X_skin_test = skin_test_known[skin_features]
y_skin_test = skin_test_known[skin_target]

# Perform grid search
skin_model = GradientBoostingRegressor(random_state=42)
skin_grid_search = GridSearchCV(
    skin_model, 
    skin_param_grid, 
    cv=5, 
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=1
)
skin_grid_search.fit(X_skin_train, y_skin_train)

# Get best parameters
skin_best_params = skin_grid_search.best_params_
print(f"Best parameters for SkinThickness imputer: {skin_best_params}")

# Evaluate on test data
skin_best_model = skin_grid_search.best_estimator_
y_skin_pred = skin_best_model.predict(X_skin_test)
skin_test_mse = mean_squared_error(y_skin_test, y_skin_pred)
skin_test_r2 = r2_score(y_skin_test, y_skin_pred)

print(f"SkinThickness Imputer Test Performance:")
print(f"MSE: {skin_test_mse:.4f}")
print(f"R²: {skin_test_r2:.4f}")

# 2.2 Tune Insulin imputer
print("\nTuning Insulin imputer...")
insulin_features = ['Age', 'BMI', 'BloodPressure', 'Pregnancies', 'SkinThickness']
insulin_target = 'Insulin'

# Prepare training data (known values only)
insulin_train_known = train_clean.dropna(subset=[insulin_target])
X_insulin_train = insulin_train_known[insulin_features]
y_insulin_train = insulin_train_known[insulin_target]

# Prepare test data (known values only for evaluation)
insulin_test_known = test_clean.dropna(subset=[insulin_target])
X_insulin_test = insulin_test_known[insulin_features]
y_insulin_test = insulin_test_known[insulin_target]

# Perform grid search
insulin_model = GradientBoostingRegressor(random_state=42)
insulin_grid_search = GridSearchCV(
    insulin_model, 
    insulin_param_grid, 
    cv=5, 
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=1
)
insulin_grid_search.fit(X_insulin_train, y_insulin_train)

# Get best parameters
insulin_best_params = insulin_grid_search.best_params_
print(f"Best parameters for Insulin imputer: {insulin_best_params}")

# Evaluate on test data
insulin_best_model = insulin_grid_search.best_estimator_
y_insulin_pred = insulin_best_model.predict(X_insulin_test)
insulin_test_mse = mean_squared_error(y_insulin_test, y_insulin_pred)
insulin_test_r2 = r2_score(y_insulin_test, y_insulin_pred)

print(f"Insulin Imputer Test Performance:")
print(f"MSE: {insulin_test_mse:.4f}")
print(f"R²: {insulin_test_r2:.4f}")

# 3. Impute missing values in both datasets using best parameters
print("\n=== IMPUTATION PHASE WITH BEST PARAMETERS ===")

# Create copies for processing
train_processed = train_clean.copy()
test_processed = test_clean.copy()

# 3.1 Impute SkinThickness using best parameters
skin_missing_train = train_processed[train_processed[skin_target].isna()]
skin_missing_test = test_processed[test_processed[skin_target].isna()]

if len(skin_missing_train) > 0 or len(skin_missing_test) > 0:
    # Train final model on all available training data
    skin_final_model = GradientBoostingRegressor(**skin_best_params, random_state=42)
    skin_final_model.fit(X_skin_train, y_skin_train)
    
    # Impute missing values in training set
    if len(skin_missing_train) > 0:
        X_skin_pred_train = skin_missing_train[skin_features]
        skin_predictions_train = skin_final_model.predict(X_skin_pred_train)
        train_processed.loc[train_processed[skin_target].isna(), skin_target] = skin_predictions_train
    
    # Impute missing values in test set
    if len(skin_missing_test) > 0:
        X_skin_pred_test = skin_missing_test[skin_features]
        skin_predictions_test = skin_final_model.predict(X_skin_pred_test)
        test_processed.loc[test_processed[skin_target].isna(), skin_target] = skin_predictions_test
    
    # Save the model
    joblib.dump(skin_final_model, 'saved-models/preprocessors/skin_thickness_imputer_gb.pkl')
    print("SkinThickness imputation model saved")
else:
    print("\nNo missing SkinThickness values to impute")

# 3.2 Impute Insulin using best parameters
insulin_missing_train = train_processed[train_processed[insulin_target].isna()]
insulin_missing_test = test_processed[test_processed[insulin_target].isna()]

if len(insulin_missing_train) > 0 or len(insulin_missing_test) > 0:
    # Train final model on all available training data
    insulin_final_model = GradientBoostingRegressor(**insulin_best_params, random_state=42)
    insulin_final_model.fit(X_insulin_train, y_insulin_train)
    
    # Impute missing values in training set
    if len(insulin_missing_train) > 0:
        X_insulin_pred_train = insulin_missing_train[insulin_features]
        insulin_predictions_train = insulin_final_model.predict(X_insulin_pred_train)
        train_processed.loc[train_processed[insulin_target].isna(), insulin_target] = insulin_predictions_train
    
    # Impute missing values in test set
    if len(insulin_missing_test) > 0:
        X_insulin_pred_test = insulin_missing_test[insulin_features]
        insulin_predictions_test = insulin_final_model.predict(X_insulin_pred_test)
        test_processed.loc[test_processed[insulin_target].isna(), insulin_target] = insulin_predictions_test
    
    # Save the model
    joblib.dump(insulin_final_model, 'saved-models/preprocessors/insulin_imputer_gb.pkl')
    print("Insulin imputation model saved")
else:
    print("\nNo missing Insulin values to impute")

# Display info after imputation
print("\nMissing values after imputation:")
print("Training set:")
print(train_processed.isnull().sum())
print("\nTest set:")
print(test_processed.isnull().sum())

# 4. Apply MinMax scaling to all features except Outcome
print("\n=== SCALING PHASE ===")

# Initialize and fit the scaler on training data
scaler = MinMaxScaler()
X_train = train_processed.drop('Outcome', axis=1)
X_train_scaled = scaler.fit_transform(X_train)

# Scale test data
X_test = test_processed.drop('Outcome', axis=1)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrames
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

# Combine with targets
y_train = train_processed['Outcome']
y_test = test_processed['Outcome']

train_scaled = pd.concat([X_train_scaled, y_train.reset_index(drop=True)], axis=1)
test_scaled = pd.concat([X_test_scaled, y_test.reset_index(drop=True)], axis=1)

# Save the scaler
joblib.dump(scaler, 'saved-models/preprocessors/minmax_scaler.pkl')
print("MinMax scaler saved to saved-models/preprocessors/minmax_scaler.pkl")

# Save the processed datasets
#train_scaled.to_csv('data/data-split/train_set_processed.csv', index=False)
test_scaled.to_csv('data/data-split/test_set_processed.csv', index=False)
#print("Processed training set saved to data/data-split/train_set_processed.csv")
print("Processed test set saved to data/data-split/test_set_processed.csv")

# Split the processed training dataset into two parts: train1 and train2
print("\n=== SPLITTING PROCESSED DATA ===")
np.random.seed(42)  
split_mask = np.random.rand(len(train_scaled)) < 0.5  # 50/50 split
train1 = train_scaled[split_mask]
train2 = train_scaled[~split_mask]

# Save the split datasets
train1.to_csv('data/data-split/train1.csv', index=False)
train2.to_csv('data/data-split/train2.csv', index=False)

print(f"Train1 shape: {train1.shape}")
print(f"Train2 shape: {train2.shape}")
print("Train1 and Train2 datasets saved to data/data-split/")

# Display summary
print("\nProcessing complete!")
print(f"Original training set shape: {train_df.shape}")
print(f"Processed training set shape: {train_scaled.shape}")
print(f"Processed test set shape: {test_scaled.shape}")
print(f"Train1 shape: {train1.shape}")
print(f"Train2 shape: {train2.shape}")

# Save best parameters for future reference
best_params = {
    'SkinThickness': skin_best_params,
    'Insulin': insulin_best_params
}

with open('saved-models/preprocessors/best_imputation_params.json', 'w') as f:
    json.dump(best_params, f, indent=4)
print("Best imputation parameters saved to saved-models/preprocessors/best_imputation_params.json")