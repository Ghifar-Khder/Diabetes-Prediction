import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import json
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

class DiabetesPredictionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Diabetes Prediction Interface")
        self.root.geometry("1200x900")  
        self.root.resizable(True, True)
        
        # Initialize variables
        self.df = None
        self.processed_df = None
        self.current_row = None
        self.input_mode = "manual"  # Can be "dataset" or "manual"
        self.imputed_values = {}
        
        # Define the expected features (without DiabetesPedigreeFunction)
        self.expected_features = ['Pregnancies', 'Glucose', 'BloodPressure', 
                                 'SkinThickness', 'Insulin', 'BMI', 'Age']
        
        # Features that should treat 0 as missing (only Insulin and SkinThickness)
        self.zero_as_missing = ['SkinThickness', 'Insulin']
        
        # Load models and preprocessors
        self.load_models()
        
        # Create interface
        self.create_widgets()
        
    def load_models(self):
        """Load all necessary models and preprocessors"""
        try:
            # Load meta-learner
            self.meta_learner = load_model('saved-models/MetaLearner_ANN.keras')
            
            # Load base models
            self.models = {
                'ANN': load_model('saved-models/ANN.keras'),
                'KNN': joblib.load('saved-models/KNN.pkl'),
                'LightGBM': joblib.load('saved-models/LightGBM.pkl'),
                'RSVM': joblib.load('saved-models/RSVM.pkl'),
                'LinearSVM': joblib.load('saved-models/LinearSVM.pkl'),
            }
            
            # Load FCM cluster centers
            self.fcm_centers = np.load('saved-models/fcm_cluster_centers.npy')
            
            # Load preprocessors
            self.skin_imputer = joblib.load('saved-models/preprocessors/skin_thickness_imputer_gb.pkl')
            self.insulin_imputer = joblib.load('saved-models/preprocessors/insulin_imputer_gb.pkl')
            self.scaler = joblib.load('saved-models/preprocessors/minmax_scaler.pkl')
            
            # Load outlier bounds
            with open('saved-models/preprocessors/outlier_bounds.json', 'r') as f:
                self.outlier_bounds = json.load(f)
                
            print("All models loaded successfully!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load models: {str(e)}")
    
    def create_widgets(self):
        """Create the GUI widgets"""
        # Create notebook for tabs
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Create frames for tabs
        dataset_frame = ttk.Frame(notebook, padding="10")
        manual_frame = ttk.Frame(notebook, padding="10")
        
        notebook.add(dataset_frame, text="Dataset Input")
        notebook.add(manual_frame, text="Manual Input")
        
        # Dataset Input Tab
        self.create_dataset_tab(dataset_frame)
        
        # Manual Input Tab
        self.create_manual_tab(manual_frame)
        
        # Results frame (common for both tabs)
        results_frame = ttk.LabelFrame(self.root, text="Prediction Results", padding="10")
        results_frame.pack(fill='x', padx=10, pady=10)
        
        # Base models predictions
        ttk.Label(results_frame, text="Base Models:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.base_results_tree = ttk.Treeview(results_frame, columns=("Model", "Prediction", "Probability"), 
                                             show="headings", height=6)
        self.base_results_tree.heading("Model", text="Model")
        self.base_results_tree.heading("Prediction", text="Prediction")
        self.base_results_tree.heading("Probability", text="Probability")
        self.base_results_tree.column("Model", width=100)
        self.base_results_tree.column("Prediction", width=100)
        self.base_results_tree.column("Probability", width=100)
        self.base_results_tree.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5)
        
        # Meta-learner prediction
        ttk.Label(results_frame, text="Meta-Learner Prediction:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.meta_prediction = tk.StringVar()
        ttk.Label(results_frame, textvariable=self.meta_prediction).grid(row=1, column=1, sticky=tk.W, padx=5)
        
        # Actual outcome
        ttk.Label(results_frame, text="Actual Outcome:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.actual_outcome = tk.StringVar()
        ttk.Label(results_frame, textvariable=self.actual_outcome).grid(row=2, column=1, sticky=tk.W, padx=5)
        
        # Prediction correctness
        ttk.Label(results_frame, text="Prediction Correct:").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.prediction_correct = tk.StringVar()
        ttk.Label(results_frame, textvariable=self.prediction_correct).grid(row=3, column=1, sticky=tk.W, padx=5)
        
        # Imputation info
        ttk.Label(results_frame, text="Imputed Values:").grid(row=4, column=0, sticky=tk.W, pady=5)
        self.imputation_info = tk.StringVar()
        ttk.Label(results_frame, textvariable=self.imputation_info).grid(row=4, column=1, sticky=tk.W, padx=5)
        
        # Add scrollbar to base results treeview
        scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.base_results_tree.yview)
        self.base_results_tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.grid(row=0, column=2, sticky=(tk.N, tk.S))
        
        # Configure grid weights for results frame
        results_frame.columnconfigure(1, weight=1)
    
    def create_dataset_tab(self, parent):
        """Create the dataset input tab"""
        # Configure grid weights
        parent.columnconfigure(1, weight=1)
        parent.rowconfigure(3, weight=1)
        
        # File selection
        ttk.Label(parent, text="Dataset:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.file_path = tk.StringVar()
        ttk.Entry(parent, textvariable=self.file_path, width=50).grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5)
        ttk.Button(parent, text="Browse", command=self.browse_file).grid(row=0, column=2, padx=5)
        
        # Row selection
        ttk.Label(parent, text="Row number:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.row_var = tk.StringVar()
        row_frame = ttk.Frame(parent)
        row_frame.grid(row=1, column=1, columnspan=2, sticky=(tk.W, tk.E))
        
        self.row_spinbox = ttk.Spinbox(row_frame, from_=0, to=100, textvariable=self.row_var, width=10)
        self.row_spinbox.pack(side=tk.LEFT, padx=5)
        ttk.Button(row_frame, text="Load and Predict", command=self.load_and_predict).pack(side=tk.LEFT, padx=5)
        
        # Calculate metrics button
        ttk.Button(parent, text="Calculate Metrics", command=self.calculate_metrics).grid(row=1, column=3, padx=5)
        
        # Data display
        ttk.Label(parent, text="Row data:").grid(row=2, column=0, sticky=tk.W, pady=5)
        
        # Treeview for data display
        columns = ("Feature", "Value")
        self.data_tree = ttk.Treeview(parent, columns=columns, show="headings", height=10)
        self.data_tree.heading("Feature", text="Feature")
        self.data_tree.heading("Value", text="Value")
        self.data_tree.column("Feature", width=200)
        self.data_tree.column("Value", width=100)
        self.data_tree.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        # Metrics frame
        metrics_frame = ttk.LabelFrame(parent, text="Dataset Metrics", padding="5")
        metrics_frame.grid(row=3, column=3, sticky=(tk.N, tk.S, tk.E, tk.W), padx=5, pady=5)
        
        # Metrics labels
        ttk.Label(metrics_frame, text="Accuracy:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.accuracy_var = tk.StringVar(value="N/A")
        ttk.Label(metrics_frame, textvariable=self.accuracy_var).grid(row=0, column=1, sticky=tk.W, pady=2)
        
        ttk.Label(metrics_frame, text="Precision:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.precision_var = tk.StringVar(value="N/A")
        ttk.Label(metrics_frame, textvariable=self.precision_var).grid(row=1, column=1, sticky=tk.W, pady=2)
        
        ttk.Label(metrics_frame, text="Recall:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.recall_var = tk.StringVar(value="N/A")
        ttk.Label(metrics_frame, textvariable=self.recall_var).grid(row=2, column=1, sticky=tk.W, pady=2)
        
        ttk.Label(metrics_frame, text="F1 Score:").grid(row=3, column=0, sticky=tk.W, pady=2)
        self.f1_var = tk.StringVar(value="N/A")
        ttk.Label(metrics_frame, textvariable=self.f1_var).grid(row=3, column=1, sticky=tk.W, pady=2)
        
        # Add scrollbar to data treeview
        scrollbar = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=self.data_tree.yview)
        self.data_tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.grid(row=3, column=4, sticky=(tk.N, tk.S))
        
        # Configure grid weights
        parent.columnconfigure(3, weight=1)
        metrics_frame.columnconfigure(1, weight=1)
    
    def create_manual_tab(self, parent):
        """Create the manual input tab"""
        # Configure grid weights
        for i in range(len(self.expected_features)):
            parent.rowconfigure(i, weight=1)
        parent.columnconfigure(1, weight=1)
        
        # Create entry widgets for each feature
        self.manual_entries = {}
        for i, feature in enumerate(self.expected_features):
            ttk.Label(parent, text=f"{feature}:").grid(row=i, column=0, sticky=tk.W, pady=5, padx=5)
            self.manual_entries[feature] = ttk.Entry(parent, width=20)
            self.manual_entries[feature].grid(row=i, column=1, sticky=(tk.W, tk.E), pady=5, padx=5)
        
        # Predict button for manual input
        ttk.Button(parent, text="Predict", command=self.predict_manual).grid(
            row=len(self.expected_features), column=0, columnspan=2, pady=10)
    
    def browse_file(self):
        """Browse for a CSV file"""
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            self.file_path.set(file_path)
            self.load_dataset()
    
    def load_dataset(self):
        """Load the selected dataset"""
        try:
            self.df = pd.read_csv(self.file_path.get())
            messagebox.showinfo("Success", f"Dataset loaded with {len(self.df)} rows and {len(self.df.columns)} columns")
            
            # Check if all required features are present
            missing_features = [f for f in self.expected_features if f not in self.df.columns]
            if missing_features:
                messagebox.showerror("Error", f"Missing required features: {missing_features}")
                return
                
            # Process the data
            self.processed_df = self.process_new_data(self.df)
            
            # Enable row selection
            self.row_var.set("0")
            self.row_spinbox.config(to=len(self.processed_df)-1)
            
            # Reset metrics
            self.accuracy_var.set("N/A")
            self.precision_var.set("N/A")
            self.recall_var.set("N/A")
            self.f1_var.set("N/A")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load dataset: {str(e)}")
    
    def calculate_metrics(self):
        """Calculate performance metrics for the entire dataset"""
        if self.processed_df is None or 'Outcome' not in self.processed_df.columns:
            messagebox.showerror("Error", "Please load a dataset with Outcome column first")
            return
            
        try:
            # Get true labels
            y_true = self.processed_df['Outcome'].values
            
            # Get predictions for all rows
            y_pred = []
            for i in range(len(self.processed_df)):
                row = self.processed_df.iloc[i]
                X = np.array([row[self.expected_features].values])
                
                # Get base model predictions
                base_predictions = self.get_base_model_predictions(X)
                
                # Prepare data for meta-learner
                base_proba_df = pd.DataFrame([base_predictions])
                combined_data = np.hstack([X, base_proba_df.values])
                
                # Get cluster membership
                memberships = self.get_cluster_memberships(combined_data, self.fcm_centers)
                
                # Prepare meta-features
                X_meta = np.hstack([X, base_proba_df.values, memberships.reshape(1, -1)])
                
                # Make prediction with meta-learner
                meta_pred_proba = self.meta_learner.predict(X_meta, verbose=0)[0][0]
                meta_prediction = 1 if meta_pred_proba > 0.5 else 0
                y_pred.append(meta_prediction)
            
            # Calculate metrics
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
            
            # Update metrics display
            self.accuracy_var.set(f"{accuracy:.4f}")
            self.precision_var.set(f"{precision:.4f}")
            self.recall_var.set(f"{recall:.4f}")
            self.f1_var.set(f"{f1:.4f}")
            
            messagebox.showinfo("Success", "Metrics calculated successfully!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to calculate metrics: {str(e)}")
    
    def load_and_predict(self):
        """Load the selected row and make prediction"""
        self.input_mode = "dataset"
        self.load_row_data()
        self.predict()
    
    def load_row_data(self):
        """Load data for the selected row"""
        if self.processed_df is None:
            messagebox.showerror("Error", "Please load a dataset first")
            return
            
        try:
            row_idx = int(self.row_var.get())
            if row_idx < 0 or row_idx >= len(self.processed_df):
                messagebox.showerror("Error", f"Row index must be between 0 and {len(self.processed_df)-1}")
                return
                
            self.current_row = self.processed_df.iloc[row_idx].copy()
            
            # Clear previous data
            for item in self.data_tree.get_children():
                self.data_tree.delete(item)
                
            # Display the row data using the expected features
            for feature in self.expected_features:
                if feature in self.current_row.index:
                    self.data_tree.insert("", "end", values=(feature, f"{self.current_row[feature]:.4f}"))
            
            # Show actual outcome if available
            if 'Outcome' in self.current_row.index:
                self.actual_outcome.set("Diabetic" if self.current_row['Outcome'] == 1 else "Non-Diabetic")
            else:
                self.actual_outcome.set("Not available")
                
            # Clear previous predictions and imputation info
            self.meta_prediction.set("")
            self.prediction_correct.set("")
            self.imputation_info.set("")
            for item in self.base_results_tree.get_children():
                self.base_results_tree.delete(item)
                
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid row number")
    
    def convert_zeros_to_nan(self, df):
        """Convert zeros to NaN for features where zero is not a valid value"""
        df_processed = df.copy()
        
        for feature in self.zero_as_missing:
            if feature in df_processed.columns:
                df_processed[feature] = df_processed[feature].replace(0, np.nan)
        
        return df_processed
    
    def predict_manual(self):
        """Make prediction from manual input"""
        self.input_mode = "manual"
        
        # Get values from manual entries
        manual_data = {}
        self.imputed_values = {}  # Reset imputed values
        
        for feature, entry in self.manual_entries.items():
            value = entry.get().strip()
            
            if value == "":
                # Mark as missing (will be imputed)
                manual_data[feature] = np.nan
            else:
                try:
                    val = float(value)
                    # For features where zero is invalid, convert to NaN
                    if feature in self.zero_as_missing and val == 0:
                        manual_data[feature] = np.nan
                    else:
                        manual_data[feature] = val
                except ValueError:
                    messagebox.showerror("Error", f"Please enter a valid number for {feature}")
                    return
        
        # Create a DataFrame from manual input
        manual_df = pd.DataFrame([manual_data])
        
        # Process the manual input and track imputed values
        try:
            processed_manual, imputed_features = self.process_new_data_with_imputation_tracking(manual_df)
            self.current_row = processed_manual.iloc[0]
            
            # Store which features were imputed
            for feature in imputed_features:
                self.imputed_values[feature] = self.current_row[feature]
            
            # Clear previous data in dataset treeview
            for item in self.data_tree.get_children():
                self.data_tree.delete(item)
                
            # Display the manual input data with imputation info
            for feature in self.expected_features:
                value = self.current_row[feature]
                if feature in self.imputed_values:
                    self.data_tree.insert("", "end", values=(feature, f"{value:.4f} (imputed)"))
                else:
                    self.data_tree.insert("", "end", values=(feature, f"{value:.4f}"))
            
            # Show imputation info
            if self.imputed_values:
                imputed_text = ", ".join([f"{k}: {v:.4f}" for k, v in self.imputed_values.items()])
                self.imputation_info.set(imputed_text)
            else:
                self.imputation_info.set("No values imputed")
            
            # No actual outcome for manual input
            self.actual_outcome.set("Not available")
            self.prediction_correct.set("")
            
            # Clear previous predictions
            self.meta_prediction.set("")
            for item in self.base_results_tree.get_children():
                self.base_results_tree.delete(item)
                
            # Make prediction
            self.predict()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to process manual input: {str(e)}")
    
    def process_new_data_with_imputation_tracking(self, df, remove_outliers=False, scale_data=True):
        """
        Apply the same processing pipeline to new data with imputation tracking
        Returns processed data and list of imputed features
        """
        df_processed = df.copy()
        imputed_features = []
        
        # Convert zeros to NaN for features where zero is invalid
        df_processed = self.convert_zeros_to_nan(df_processed)
        
        # 1. Impute missing values and track which ones were imputed
        # Impute SkinThickness
        skin_features = ['Age', 'BMI', 'BloodPressure', 'Pregnancies']
        skin_missing = df_processed[df_processed['SkinThickness'].isna()]
        if len(skin_missing) > 0:
            X_skin_pred = skin_missing[skin_features]
            skin_predictions = self.skin_imputer.predict(X_skin_pred)
            df_processed.loc[df_processed['SkinThickness'].isna(), 'SkinThickness'] = skin_predictions
            imputed_features.append('SkinThickness')
        
        # Impute Insulin
        insulin_features = ['Age', 'BMI', 'BloodPressure', 'Pregnancies', 'SkinThickness']
        insulin_missing = df_processed[df_processed['Insulin'].isna()]
        if len(insulin_missing) > 0:
            X_insulin_pred = insulin_missing[insulin_features]
            insulin_predictions = self.insulin_imputer.predict(X_insulin_pred)
            df_processed.loc[df_processed['Insulin'].isna(), 'Insulin'] = insulin_predictions
            imputed_features.append('Insulin')
        
        # 2. Remove outliers (disabled for single row prediction)
        if remove_outliers and len(df_processed) > 1:
            outliers_indices = set()
            for col, bounds in self.outlier_bounds.items():
                if col in df_processed.columns:
                    lower_bound = bounds['lower_bound']
                    upper_bound = bounds['upper_bound']
                    col_outliers = df_processed[
                        (df_processed[col] < lower_bound) | (df_processed[col] > upper_bound)
                    ].index
                    outliers_indices.update(col_outliers)
            
            df_processed = df_processed.drop(index=outliers_indices)
        
        # 3. Scale data
        if scale_data:
            # Use only the expected features
            X = df_processed[self.expected_features]
            X_scaled = self.scaler.transform(X)
            df_processed = pd.DataFrame(X_scaled, columns=self.expected_features)
        
        return df_processed, imputed_features
    
    def process_new_data(self, df, impute_missing=True, remove_outliers=False, scale_data=True):
        """
        Apply the same processing pipeline to new data (without imputation tracking)
        """
        df_processed = df.copy()
        
        # Convert zeros to NaN for features where zero is invalid
        df_processed = self.convert_zeros_to_nan(df_processed)
        
        # 1. Impute missing values
        if impute_missing:
            # Impute SkinThickness
            skin_features = ['Age', 'BMI', 'BloodPressure', 'Pregnancies']
            skin_missing = df_processed[df_processed['SkinThickness'].isna()]
            if len(skin_missing) > 0:
                X_skin_pred = skin_missing[skin_features]
                skin_predictions = self.skin_imputer.predict(X_skin_pred)
                df_processed.loc[df_processed['SkinThickness'].isna(), 'SkinThickness'] = skin_predictions
            
            # Impute Insulin
            insulin_features = ['Age', 'BMI', 'BloodPressure', 'Pregnancies', 'SkinThickness']
            insulin_missing = df_processed[df_processed['Insulin'].isna()]
            if len(insulin_missing) > 0:
                X_insulin_pred = insulin_missing[insulin_features]
                insulin_predictions = self.insulin_imputer.predict(X_insulin_pred)
                df_processed.loc[df_processed['Insulin'].isna(), 'Insulin'] = insulin_predictions
        
        # 2. Remove outliers (disabled for single row prediction)
        if remove_outliers and len(df_processed) > 1:
            outliers_indices = set()
            for col, bounds in self.outlier_bounds.items():
                if col in df_processed.columns:
                    lower_bound = bounds['lower_bound']
                    upper_bound = bounds['upper_bound']
                    col_outliers = df_processed[
                        (df_processed[col] < lower_bound) | (df_processed[col] > upper_bound)
                    ].index
                    outliers_indices.update(col_outliers)
            
            df_processed = df_processed.drop(index=outliers_indices)
        
        # 3. Scale data
        if scale_data:
            # Separate features and target
            if 'Outcome' in df_processed.columns:
                X = df_processed[self.expected_features]  # Use only the expected features
                y = df_processed['Outcome']
                X_scaled = self.scaler.transform(X)
                X_scaled = pd.DataFrame(X_scaled, columns=self.expected_features)
                df_processed = pd.concat([X_scaled, y.reset_index(drop=True)], axis=1)
            else:
                X = df_processed[self.expected_features]  # Use only the expected features
                X_scaled = self.scaler.transform(X)
                df_processed = pd.DataFrame(X_scaled, columns=self.expected_features)
        
        return df_processed
    
    def get_base_model_predictions(self, X):
        """Get prediction probabilities from base models"""
        predictions = {}
        
        for name, model in self.models.items():
            if name == 'ANN':
                pred_proba = model.predict(X, verbose=0)
                pred_proba = pred_proba[:, 1] if pred_proba.shape[1] > 1 else pred_proba.flatten()
            else:
                if hasattr(model, 'predict_proba'):
                    pred_proba = model.predict_proba(X)[:, 1]
                else:
                    pred_proba = model.decision_function(X)
                    pred_proba = (pred_proba - pred_proba.min()) / (pred_proba.max() - pred_proba.min() + 1e-8)
            
            predictions[name] = pred_proba[0]
        
        return predictions
    
    def get_cluster_memberships(self, combined_data, centers):
        """Calculate cluster memberships for a data point"""
        m = 2
        point = combined_data[0] if len(combined_data.shape) > 1 else combined_data
        
        distances = [np.linalg.norm(point - center) for center in centers]
        
        if any(d == 0 for d in distances):
            u = [1.0 if d == 0 else 0.0 for d in distances]
        else:
            u = []
            for i in range(len(centers)):
                denominator = sum([(distances[i] / distances[j]) ** (2 / (m - 1)) 
                                  for j in range(len(centers))])
                u.append(1.0 / denominator)
        
        return np.array(u)
    
    def predict(self):
        """Make prediction for the current row"""
        if self.current_row is None:
            messagebox.showerror("Error", "No data loaded for prediction")
            return
            
        try:
            # Extract features (only the expected features)
            X = np.array([self.current_row[self.expected_features].values])
            
            # Get base model predictions
            base_predictions = self.get_base_model_predictions(X)
            
            # Clear previous base model results
            for item in self.base_results_tree.get_children():
                self.base_results_tree.delete(item)
                
            # Display base model results
            for model_name, proba in base_predictions.items():
                prediction = "Diabetic" if proba > 0.5 else "Non-Diabetic"
                self.base_results_tree.insert("", "end", values=(model_name, prediction, f"{proba:.4f}"))
            
            # Prepare data for meta-learner
            base_proba_df = pd.DataFrame([base_predictions])
            combined_data = np.hstack([X, base_proba_df.values])
            
            # Get cluster membership
            memberships = self.get_cluster_memberships(combined_data, self.fcm_centers)
            
            # Prepare meta-features
            X_meta = np.hstack([X, base_proba_df.values, memberships.reshape(1, -1)])
            
            # Make prediction with meta-learner
            meta_pred_proba = self.meta_learner.predict(X_meta, verbose=0)[0][0]
            meta_prediction = "Diabetic" if meta_pred_proba > 0.5 else "Non-Diabetic"
            
            self.meta_prediction.set(f"{meta_prediction} (Probability: {meta_pred_proba:.4f})")
            
            # Check if prediction is correct (only for dataset input with known outcome)
            if self.input_mode == "dataset" and 'Outcome' in self.current_row.index:
                actual = self.current_row['Outcome']
                predicted = 1 if meta_prediction == "Diabetic" else 0
                is_correct = "Yes" if actual == predicted else "No"
                self.prediction_correct.set(is_correct)
            
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = DiabetesPredictionApp(root)
    root.mainloop()