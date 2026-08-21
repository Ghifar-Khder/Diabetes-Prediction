"""
Diabetes Prediction — Stacked Ensemble (ANN meta-learner) Streamlit interface.

Run locally:
    streamlit run app.py

Deploy online (free): push this repo to GitHub, then create an app at
https://share.streamlit.io pointing at this file. Streamlit Cloud installs
requirements.txt and runs `streamlit run app.py` automatically — no server
setup needed.
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from tensorflow.keras.models import load_model

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODELS_DIR = "saved-models"
EXPECTED_FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure",
    "SkinThickness", "Insulin", "BMI", "Age",
]
# Features where a recorded 0 is actually a missing measurement
ZERO_AS_MISSING = ["SkinThickness", "Insulin"]

FEATURE_INFO = {
    "Pregnancies":   dict(label="Pregnancies", min=0, max=17, step=1, default=1, help="Number of times pregnant"),
    "Glucose":       dict(label="Glucose (mg/dL)", min=0, max=250, step=1, default=120, help="Plasma glucose concentration, 2h oral glucose tolerance test"),
    "BloodPressure": dict(label="Blood Pressure (mm Hg)", min=0, max=140, step=1, default=70, help="Diastolic blood pressure"),
    "SkinThickness": dict(label="Skin Thickness (mm)", min=0, max=100, step=1, default=20, help="Triceps skin fold thickness — leave at 0 / check 'unknown' to impute"),
    "Insulin":       dict(label="Insulin (mu U/mL)", min=0, max=850, step=1, default=80, help="2-Hour serum insulin — leave at 0 / check 'unknown' to impute"),
    "BMI":           dict(label="BMI", min=0.0, max=70.0, step=0.1, default=28.0, help="Body mass index"),
    "Age":           dict(label="Age (years)", min=1, max=120, step=1, default=33, help="Age in years"),
}

BASE_MODEL_NAMES = ["ANN", "KNN", "LightGBM", "RSVM", "LinearSVM"]


# --------------------------------------------------------------------------
# Model / preprocessor loading (cached — only runs once per session)
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading models…")
def load_artifacts():
    missing = [
        p for p in [
            f"{MODELS_DIR}/ANN.keras", f"{MODELS_DIR}/KNN.pkl",
            f"{MODELS_DIR}/LightGBM.pkl", f"{MODELS_DIR}/RSVM.pkl",
            f"{MODELS_DIR}/LinearSVM.pkl", f"{MODELS_DIR}/MetaLearner_ANN.keras",
            f"{MODELS_DIR}/fcm_cluster_centers.npy",
            f"{MODELS_DIR}/preprocessors/skin_thickness_imputer_gb.pkl",
            f"{MODELS_DIR}/preprocessors/insulin_imputer_gb.pkl",
            f"{MODELS_DIR}/preprocessors/minmax_scaler.pkl",
            f"{MODELS_DIR}/preprocessors/outlier_bounds.json",
        ] if not os.path.exists(p)
    ]
    if missing:
        raise FileNotFoundError(
            "Missing model files: " + ", ".join(missing) +
            ". Run this app from the repository root so 'saved-models/' resolves."
        )

    models = {
        "ANN": load_model(f"{MODELS_DIR}/ANN.keras"),
        "KNN": joblib.load(f"{MODELS_DIR}/KNN.pkl"),
        "LightGBM": joblib.load(f"{MODELS_DIR}/LightGBM.pkl"),
        "RSVM": joblib.load(f"{MODELS_DIR}/RSVM.pkl"),
        "LinearSVM": joblib.load(f"{MODELS_DIR}/LinearSVM.pkl"),
    }
    meta_learner = load_model(f"{MODELS_DIR}/MetaLearner_ANN.keras")
    fcm_centers = np.load(f"{MODELS_DIR}/fcm_cluster_centers.npy")
    skin_imputer = joblib.load(f"{MODELS_DIR}/preprocessors/skin_thickness_imputer_gb.pkl")
    insulin_imputer = joblib.load(f"{MODELS_DIR}/preprocessors/insulin_imputer_gb.pkl")
    scaler = joblib.load(f"{MODELS_DIR}/preprocessors/minmax_scaler.pkl")
    with open(f"{MODELS_DIR}/preprocessors/outlier_bounds.json") as f:
        outlier_bounds = json.load(f)

    return {
        "models": models,
        "meta_learner": meta_learner,
        "fcm_centers": fcm_centers,
        "skin_imputer": skin_imputer,
        "insulin_imputer": insulin_imputer,
        "scaler": scaler,
        "outlier_bounds": outlier_bounds,
    }


# --------------------------------------------------------------------------
# Pipeline logic (mirrors src/interface/interface.py + src/test.py)
# --------------------------------------------------------------------------
def convert_zeros_to_nan(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for feature in ZERO_AS_MISSING:
        if feature in df.columns:
            df[feature] = df[feature].replace(0, np.nan)
    return df


def impute(df: pd.DataFrame, art: dict):
    """Impute missing SkinThickness / Insulin. Returns (df, list of imputed feature names)."""
    df = df.copy()
    imputed = []

    skin_features = ["Age", "BMI", "BloodPressure", "Pregnancies"]
    skin_missing = df[df["SkinThickness"].isna()]
    if len(skin_missing) > 0:
        preds = art["skin_imputer"].predict(skin_missing[skin_features])
        df.loc[df["SkinThickness"].isna(), "SkinThickness"] = preds
        imputed.append("SkinThickness")

    insulin_features = ["Age", "BMI", "BloodPressure", "Pregnancies", "SkinThickness"]
    insulin_missing = df[df["Insulin"].isna()]
    if len(insulin_missing) > 0:
        preds = art["insulin_imputer"].predict(insulin_missing[insulin_features])
        df.loc[df["Insulin"].isna(), "Insulin"] = preds
        imputed.append("Insulin")

    return df, imputed


def scale(df: pd.DataFrame, art: dict) -> np.ndarray:
    return art["scaler"].transform(df[EXPECTED_FEATURES])


def get_base_model_predictions(X_scaled: np.ndarray, art: dict) -> pd.DataFrame:
    """X_scaled: (n_samples, n_features). Returns DataFrame of per-model probabilities."""
    preds = {}
    for name, model in art["models"].items():
        if name == "ANN":
            p = model.predict(X_scaled, verbose=0)
            p = p[:, 1] if p.shape[1] > 1 else p.flatten()
        elif hasattr(model, "predict_proba"):
            p = model.predict_proba(X_scaled)[:, 1]
        else:
            p = model.decision_function(X_scaled)
            p = (p - p.min()) / (p.max() - p.min() + 1e-8)
        preds[name] = p
    return pd.DataFrame(preds)


def get_cluster_memberships(combined: np.ndarray, centers: np.ndarray) -> np.ndarray:
    m = 2
    memberships = []
    for point in combined:
        distances = [np.linalg.norm(point - c) for c in centers]
        if any(d == 0 for d in distances):
            u = [1.0 if d == 0 else 0.0 for d in distances]
        else:
            u = []
            for i in range(len(centers)):
                denom = sum((distances[i] / distances[j]) ** (2 / (m - 1)) for j in range(len(centers)))
                u.append(1.0 / denom)
        memberships.append(u)
    return np.array(memberships)


def run_pipeline(df_raw: pd.DataFrame, art: dict):
    """
    Full pipeline from raw (unscaled) feature values to final prediction.
    df_raw must contain EXPECTED_FEATURES, may contain 0s for missing values.
    Returns: base_proba_df, meta_proba (array), meta_pred (array 0/1), imputed_features
    """
    df = convert_zeros_to_nan(df_raw[EXPECTED_FEATURES])
    df, imputed_features = impute(df, art)
    X_scaled = scale(df, art)

    base_proba = get_base_model_predictions(X_scaled, art)
    combined = np.hstack([X_scaled, base_proba.values])
    memberships = get_cluster_memberships(combined, art["fcm_centers"])
    X_meta = np.hstack([X_scaled, base_proba.values, memberships])

    meta_proba = art["meta_learner"].predict(X_meta, verbose=0).flatten()
    meta_pred = (meta_proba > 0.5).astype(int)

    return base_proba, meta_proba, meta_pred, imputed_features, df


# --------------------------------------------------------------------------
# UI helpers
# --------------------------------------------------------------------------
def verdict_banner(proba: float):
    pct = proba * 100
    if proba > 0.5:
        st.error(f"### 🔴 Diabetic — predicted probability {pct:.1f}%")
    else:
        st.success(f"### 🟢 Non-Diabetic — predicted probability {pct:.1f}%")
    st.progress(min(max(proba, 0.0), 1.0))


def base_model_chart(base_proba_row: pd.Series):
    colors = ["#e74c3c" if v > 0.5 else "#2ecc71" for v in base_proba_row.values]
    fig = go.Figure(
        go.Bar(
            x=base_proba_row.values,
            y=base_proba_row.index,
            orientation="h",
            marker_color=colors,
            text=[f"{v:.1%}" for v in base_proba_row.values],
            textposition="outside",
        )
    )
    fig.add_vline(x=0.5, line_dash="dash", line_color="gray")
    fig.update_layout(
        xaxis=dict(range=[0, 1], tickformat=".0%", title="Predicted probability of diabetes"),
        yaxis_title=None,
        height=260,
        margin=dict(l=10, r=10, t=10, b=30),
    )
    return fig


def imputation_note(imputed_features, row: pd.Series):
    if imputed_features:
        vals = ", ".join(f"**{f}** → {row[f]:.1f}" for f in imputed_features)
        st.info(f"ℹ️ Missing values were estimated by the trained imputers: {vals}")
    else:
        st.caption("No values needed imputation.")


# --------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------
st.title("🩺 Diabetes Risk Predictor")
st.caption(
    "Fuzzy C-Means enhanced stacked ensemble (ANN · KNN · LightGBM · RBF-SVM · Linear-SVM) "
    "with an ANN meta-learner, trained on the Pima Indians Diabetes dataset."
)

try:
    ART = load_artifacts()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

tab_manual, tab_batch, tab_about = st.tabs(["🧍 Single prediction", "📄 Batch dataset", "ℹ️ About"])

# ---------------- Tab 1: manual single-patient prediction ----------------
with tab_manual:
    st.subheader("Enter patient values")
    st.caption("Check 'Unknown' for Skin Thickness or Insulin if that measurement wasn't taken — the app will estimate it the same way the training pipeline does.")

    col1, col2 = st.columns(2)
    values = {}
    unknown_flags = {}
    columns_cycle = [col1, col2]
    for i, feature in enumerate(EXPECTED_FEATURES):
        info = FEATURE_INFO[feature]
        target_col = columns_cycle[i % 2]
        with target_col:
            if feature in ZERO_AS_MISSING:
                sub1, sub2 = st.columns([3, 1])
                unknown = sub2.checkbox("Unknown", key=f"unk_{feature}", help=f"Treat {feature} as missing")
                val = sub1.number_input(
                    info["label"], min_value=info["min"], max_value=info["max"],
                    value=info["default"], step=info["step"], help=info["help"],
                    disabled=unknown,
                )
                values[feature] = 0 if unknown else val
                unknown_flags[feature] = unknown
            else:
                values[feature] = st.number_input(
                    info["label"], min_value=info["min"], max_value=info["max"],
                    value=info["default"], step=info["step"], help=info["help"],
                )

    predict_clicked = st.button("Predict", type="primary", use_container_width=True)

    if predict_clicked:
        input_df = pd.DataFrame([values])
        with st.spinner("Running the ensemble…"):
            base_proba, meta_proba, meta_pred, imputed_features, processed = run_pipeline(input_df, ART)

        st.divider()
        verdict_banner(float(meta_proba[0]))
        imputation_note(imputed_features, processed.iloc[0])

        st.markdown("##### Base model votes")
        st.plotly_chart(base_model_chart(base_proba.iloc[0]), use_container_width=True)

        with st.expander("Show processed (scaled) feature values"):
            st.dataframe(processed.round(4), use_container_width=True, hide_index=True)

# ---------------- Tab 2: batch dataset ----------------
with tab_batch:
    st.subheader("Upload a CSV and predict on any row, or evaluate the whole file")
    st.caption(f"Required columns: {', '.join(EXPECTED_FEATURES)}. An optional 'Outcome' column (0/1) enables accuracy metrics.")

    uploaded = st.file_uploader("Upload dataset (.csv)", type=["csv"])

    if uploaded is not None:
        try:
            raw_df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            st.stop()

        missing_cols = [f for f in EXPECTED_FEATURES if f not in raw_df.columns]
        if missing_cols:
            st.error(f"Uploaded file is missing required columns: {missing_cols}")
            st.stop()

        st.success(f"Loaded {len(raw_df)} rows.")
        has_outcome = "Outcome" in raw_df.columns

        left, right = st.columns([1, 1])
        with left:
            st.markdown("##### Predict a single row")
            row_idx = st.number_input("Row index", min_value=0, max_value=len(raw_df) - 1, value=0, step=1)
            st.dataframe(raw_df.loc[[row_idx], EXPECTED_FEATURES], use_container_width=True, hide_index=True)
            row_predict_clicked = st.button("Predict this row", use_container_width=True)

            if row_predict_clicked:
                base_proba, meta_proba, meta_pred, imputed_features, processed = run_pipeline(
                    raw_df.iloc[[row_idx]], ART
                )
                verdict_banner(float(meta_proba[0]))
                imputation_note(imputed_features, processed.iloc[0])
                st.plotly_chart(base_model_chart(base_proba.iloc[0]), use_container_width=True)
                if has_outcome:
                    actual = int(raw_df.iloc[row_idx]["Outcome"])
                    correct = actual == int(meta_pred[0])
                    st.markdown(
                        f"**Actual outcome:** {'Diabetic' if actual == 1 else 'Non-Diabetic'} — "
                        f"{'✅ prediction matches' if correct else '❌ prediction differs'}"
                    )

        with right:
            st.markdown("##### Evaluate full dataset")
            if not has_outcome:
                st.caption("Add an 'Outcome' column to your CSV to compute accuracy metrics.")
            eval_clicked = st.button(
                "Run predictions on all rows", use_container_width=True, disabled=not has_outcome
            )
            if eval_clicked:
                with st.spinner(f"Predicting {len(raw_df)} rows…"):
                    base_proba, meta_proba, meta_pred, imputed_features, processed = run_pipeline(raw_df, ART)
                    y_true = raw_df["Outcome"].values
                    y_pred = meta_pred

                acc = accuracy_score(y_true, y_pred)
                prec = precision_score(y_true, y_pred)
                rec = recall_score(y_true, y_pred)
                f1 = f1_score(y_true, y_pred)

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Accuracy", f"{acc:.1%}")
                m2.metric("Precision", f"{prec:.1%}")
                m3.metric("Recall", f"{rec:.1%}")
                m4.metric("F1", f"{f1:.1%}")

                cm = confusion_matrix(y_true, y_pred)
                cm_fig = go.Figure(
                    data=go.Heatmap(
                        z=cm,
                        x=["Pred: Non-Diabetic", "Pred: Diabetic"],
                        y=["Actual: Non-Diabetic", "Actual: Diabetic"],
                        colorscale="Blues",
                        text=cm,
                        texttemplate="%{text}",
                        showscale=False,
                    )
                )
                cm_fig.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10), title="Confusion matrix")
                st.plotly_chart(cm_fig, use_container_width=True)

                results_df = raw_df.copy()
                results_df["Predicted_Probability"] = meta_proba
                results_df["Predicted_Outcome"] = y_pred
                st.download_button(
                    "Download predictions as CSV",
                    results_df.to_csv(index=False).encode("utf-8"),
                    file_name="predictions.csv",
                    mime="text/csv",
                )

# ---------------- Tab 3: about ----------------
with tab_about:
    st.markdown(
        """
### How this works
1. **Preprocessing** — zero-valued Skin Thickness / Insulin readings are treated as missing and
   estimated with the same Gradient-Boosting imputers used during training, then all seven
   features are Min-Max scaled.
2. **Base models** — five classifiers (ANN, KNN, LightGBM, RBF-SVM, Linear-SVM) each output a
   diabetes probability.
3. **Fuzzy C-Means membership** — the scaled features plus the five base-model probabilities are
   compared against saved cluster centers to compute soft cluster memberships.
4. **Meta-learner** — an ANN takes the scaled features, base-model probabilities, and cluster
   memberships and produces the final prediction.

This app is a Streamlit front end for the
[Diabetes-Prediction](https://github.com/Ghifar-Khder/Diabetes-Prediction) repository — it
loads the same `saved-models/` artifacts used in `src/test.py` and
`src/interface/interface.py`, so predictions are identical to those from the desktop
(Tkinter) interface.

**Disclaimer:** this tool is for educational/demonstration purposes only and is not a medical
diagnostic device.
"""
    )