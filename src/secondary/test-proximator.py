import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_squared_error, r2_score

def remove_outliers_iqr(df, columns):
    """
    Remove outliers from specified columns using IQR method
    (Same function as in training code)
    """
    df_clean = df.copy()
    outliers_indices = set()
    
    for col in columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Find outliers
        col_outliers = df_clean[(df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)].index
        outliers_indices.update(col_outliers)
        
        print(f"{col}: {len(col_outliers)} outliers")
    
    # Remove all rows with outliers in any column
    df_clean = df_clean.drop(index=outliers_indices)
    print(f"\nTotal rows with outliers: {len(outliers_indices)}")
    print(f"Dataset shape after removing outliers: {df_clean.shape}")
    
    return df_clean

def evaluate_imputation_models():
    """
    Evaluate the performance of SkinThickness and Insulin imputation models
    on both training and validation sets without any data leakage
    """
    # Load the original datasets (without any processing)
    train_df = pd.read_csv('data/data-split/train_set.csv')
    val_df = pd.read_csv('data/data-split/validation_set.csv')
    
    # Load the saved models
    skin_model = joblib.load('saved-models/preprocessors/skin_thickness_imputer_gb.pkl')
    insulin_model = joblib.load('saved-models/preprocessors/insulin_imputer_gb.pkl')
    
    # Define the features used for each model
    skin_features = ['Age', 'BMI', 'BloodPressure', 'Pregnancies']
    insulin_features = ['Age', 'BMI', 'BloodPressure', 'Pregnancies', 'SkinThickness']
    
    # Apply the same outlier removal as in training
    print("Removing outliers from training data...")
    outlier_columns = train_df.columns.difference(['Outcome'])
    print("Detecting outliers in columns:", list(outlier_columns))
    
    train_clean = remove_outliers_iqr(train_df, outlier_columns)
    
    # Create clean copies for evaluation
    train_eval = train_clean.copy()
    val_eval = val_df.copy()  # Note: Validation set doesn't get outlier removal
    
    print("="*60)
    print("EVALUATING IMPUTATION MODELS (WITH OUTLIER REMOVAL)")
    print("="*60)
    
    # 1. First, evaluate SkinThickness model
    print("\n" + "="*30)
    print("SKIN THICKNESS MODEL EVALUATION")
    print("="*30)
    
    # Evaluate on training set (with outliers removed)
    skin_known_train = train_eval.dropna(subset=['SkinThickness'])
    if len(skin_known_train) > 0:
        X_skin_train = skin_known_train[skin_features]
        y_skin_train = skin_known_train['SkinThickness']
        
        y_skin_pred_train = skin_model.predict(X_skin_train)
        
        skin_train_r2 = r2_score(y_skin_train, y_skin_pred_train)
        skin_train_mse = mean_squared_error(y_skin_train, y_skin_pred_train)
        
        print(f"Training Set - R²: {skin_train_r2:.4f}, MSE: {skin_train_mse:.4f}, Samples: {len(skin_known_train)}")
    else:
        print("No SkinThickness data in training set for evaluation")
    
    # Evaluate on validation set (no outlier removal)
    skin_known_val = val_eval.dropna(subset=['SkinThickness'])
    if len(skin_known_val) > 0:
        X_skin_val = skin_known_val[skin_features]
        y_skin_val = skin_known_val['SkinThickness']
        
        y_skin_pred_val = skin_model.predict(X_skin_val)
        
        skin_val_r2 = r2_score(y_skin_val, y_skin_pred_val)
        skin_val_mse = mean_squared_error(y_skin_val, y_skin_pred_val)
        
        print(f"Validation Set - R²: {skin_val_r2:.4f}, MSE: {skin_val_mse:.4f}, Samples: {len(skin_known_val)}")
    else:
        print("No SkinThickness data in validation set for evaluation")
    
    # 2. Prepare data for Insulin evaluation by imputing SkinThickness first
    print("\n" + "="*30)
    print("INSULIN MODEL EVALUATION")
    print("="*30)
    
    # Create copies for insulin evaluation with SkinThickness imputation
    train_insulin_eval = train_eval.copy()
    val_insulin_eval = val_eval.copy()
    
    # Impute SkinThickness for insulin evaluation
    for df in [train_insulin_eval, val_insulin_eval]:
        skin_missing = df[df['SkinThickness'].isna()]
        if len(skin_missing) > 0:
            X_skin_pred = skin_missing[skin_features]
            skin_predictions = skin_model.predict(X_skin_pred)
            df.loc[df['SkinThickness'].isna(), 'SkinThickness'] = skin_predictions
    
    # Evaluate Insulin model on training set (with outliers removed)
    insulin_known_train = train_insulin_eval.dropna(subset=['Insulin'])
    if len(insulin_known_train) > 0:
        X_insulin_train = insulin_known_train[insulin_features]
        y_insulin_train = insulin_known_train['Insulin']
        
        y_insulin_pred_train = insulin_model.predict(X_insulin_train)
        
        insulin_train_r2 = r2_score(y_insulin_train, y_insulin_pred_train)
        insulin_train_mse = mean_squared_error(y_insulin_train, y_insulin_pred_train)
        
        print(f"Training Set - R²: {insulin_train_r2:.4f}, MSE: {insulin_train_mse:.4f}, Samples: {len(insulin_known_train)}")
    else:
        print("No Insulin data in training set for evaluation")
    
    # Evaluate Insulin model on validation set (no outlier removal)
    insulin_known_val = val_insulin_eval.dropna(subset=['Insulin'])
    if len(insulin_known_val) > 0:
        X_insulin_val = insulin_known_val[insulin_features]
        y_insulin_val = insulin_known_val['Insulin']
        
        y_insulin_pred_val = insulin_model.predict(X_insulin_val)
        
        insulin_val_r2 = r2_score(y_insulin_val, y_insulin_pred_val)
        insulin_val_mse = mean_squared_error(y_insulin_val, y_insulin_pred_val)
        
        print(f"Validation Set - R²: {insulin_val_r2:.4f}, MSE: {insulin_val_mse:.4f}, Samples: {len(insulin_known_val)}")
    else:
        print("No Insulin data in validation set for evaluation")
    
    # 3. Check for overfitting
    print("\n" + "="*30)
    print("OVERFITTING ANALYSIS")
    print("="*30)
    
    if 'insulin_train_r2' in locals() and 'insulin_val_r2' in locals():
        r2_diff = insulin_train_r2 - insulin_val_r2
        print(f"R² difference (Train - Val): {r2_diff:.4f}")
        
        if r2_diff > 0.3:
            print("→ Significant overfitting detected")
        elif r2_diff > 0.1:
            print("→ Moderate overfitting detected")
        else:
            print("→ Minimal overfitting detected")
    
    # 4. Compare with your previous results
    print("\n" + "="*30)
    print("COMPARISON WITH PREVIOUS RESULTS")
    print("="*30)
    print("Your previous independent test results:")
   
    
    if 'insulin_train_r2' in locals() and 'insulin_val_r2' in locals():
        print("\nCurrent test results:")
        print(f"Training R²: {insulin_train_r2:.4f}, Validation R²: {insulin_val_r2:.4f}")
        
        
if __name__ == "__main__":
    evaluate_imputation_models()