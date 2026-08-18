"""
INFERENCE PIPELINE - Production ML Model Serving with Feature Consistency
=========================================================================

This module provides the core inference functionality for the Telco Churn prediction model.
It ensures that serving-time feature transformations exactly match training-time transformations,
which is CRITICAL for model accuracy in production.

Key Responsibilities:
1. Load MLflow-logged model and feature metadata from training
2. Apply identical feature transformations as used during training
3. Ensure correct feature ordering for model input
4. Convert model predictions to user-friendly output

CRITICAL PATTERN: Training/Serving Consistency
- Uses fixed BINARY_MAP for deterministic binary encoding
- Applies same one-hot encoding with drop_first=True
- Maintains exact feature column order from training
- Handles missing/new categorical values gracefully

Production Deployment:
- MODEL_DIR points to containerized model artifacts
- Feature schema loaded from training-time artifacts
- Optimized for single-row inference (real-time serving)
"""

import os
import glob
from pathlib import Path

import pandas as pd
import yaml
import mlflow

# === MODEL LOADING CONFIGURATION ===
# IMPORTANT: This path is set during Docker container build
# In development: uses local MLflow artifacts
# In production: uses model copied to container at build time
MODEL_DIR = "/app/model"

try:
    # Load the trained model in MLflow pyfunc format
    # This ensures compatibility regardless of the underlying ML library
    model = mlflow.pyfunc.load_model(MODEL_DIR)
    print(f"✅ Model loaded successfully from {MODEL_DIR}")
except Exception as e:
    print(f"❌ Failed to load model from {MODEL_DIR}: {e}")
    # Fallback for local development (OPTIONAL)
    try:
        local_model_paths = glob.glob("./mlruns/*/models/m-*/artifacts")
        if local_model_paths:
            latest_model = max(local_model_paths, key=os.path.getmtime)

            #convert the filesystem path into a URI
            #MLflow's load_model() accepts model locations in URI-like formats, such as:s3://...,http://...,file:///C:/...
            local_path = Path(latest_model).resolve() #This converts the model path into an absolute path.
            model = mlflow.pyfunc.load_model(local_path.as_uri())  #as_uri() converts the local filesystem path into a proper file:// URI so MLflow doesn't misparse a Windows drive letter (C:\...) as a URI scheme.
            MODEL_DIR = str(local_path)
            print(f"✅ Fallback: Loaded model from {MODEL_DIR}")
        else:
            raise Exception("No model found in local mlruns")
    except Exception as fallback_error:
        raise Exception(f"Failed to load model: {e}. Fallback failed: {fallback_error}")

# === FEATURE SCHEMA LOADING ===
# CRITICAL: Load the exact feature column order used during training,from the SAME RUN that produced the loaded model. 
# Multiple training runs can each leave behind their own model + feature_columns.txt, 
# so we must not pair a model from one run with feature columns from a different run
# (that reintroduces train/serve skew even though loading "succeeds").
try:
    feature_file = os.path.join(MODEL_DIR, "feature_columns.txt")

    if not os.path.exists(feature_file):
        # Not co-located with the model artifacts (common when it was logged
        # via mlflow.log_artifact() against the *run*, not the logged model).
        mlmodel_path = os.path.join(MODEL_DIR, "MLmodel")
        run_id = None

        if os.path.exists(mlmodel_path):
            with open(mlmodel_path) as f:
                mlmodel_meta = yaml.safe_load(f)
            run_id = mlmodel_meta.get("run_id")

        if run_id:
            # MODEL_DIR looks like: .../mlruns/<exp_id>/models/m-<hash>/artifacts
            # The run's own artifacts live at: .../mlruns/<exp_id>/<run_id>/artifacts
            model_dir_path = Path(MODEL_DIR) #Converts the model directory into a Path object.
            #find where the "models" directory appears in the path.
            models_idx = model_dir_path.parts.index("models") if "models" in model_dir_path.parts else None

            ##Uses everything before "models" as the MLflow experiment root.
            if models_idx is not None:
                exp_root = Path(*model_dir_path.parts[:models_idx]) #takes everything in the path before models and turn it back into a Path.
                candidate = exp_root / run_id / "artifacts" / "feature_columns.txt" #Constructs the expected path to feature_columns.txt for the run_id.
                if candidate.exists():
                    feature_file = str(candidate)
                    print(f"ℹ️  Using feature_columns.txt from matching run {run_id}: {feature_file}")

        if not os.path.exists(feature_file):
            # Last-resort fallback: search anywhere under mlruns/. This is NOT
            # guaranteed to match the loaded model's training run — only used
            # when the run_id could not be resolved from MLmodel above.
            candidates = glob.glob("./mlruns/**/feature_columns.txt", recursive=True) #recursive=True means it searches through nested directories.
            if candidates:
                feature_file = max(candidates, key=os.path.getmtime)
                print(
                    f"⚠️  Could not confirm run match; using most recent "
                    f"feature_columns.txt found: {feature_file}"
                )

    with open(feature_file) as f:
        FEATURE_COLS = [ln.strip() for ln in f if ln.strip()] 
        #Keeps only lines where the stripped result isn't empty.
        #ln.strip() Removes whitespace from the beginning and end, including the newline \n.
    print(f"✅ Loaded {len(FEATURE_COLS)} feature columns from training")
except Exception as e:
    raise Exception(f"Failed to load feature columns: {e}")

# === FEATURE TRANSFORMATION CONSTANTS ===
# CRITICAL: These mappings must exactly match those used in training
# Any changes here will cause train/serve skew and degrade model performance

# Deterministic binary feature mappings (consistent with training)
BINARY_MAP = {
    "gender": {"Female": 0, "Male": 1},           # Demographics
    "Partner": {"No": 0, "Yes": 1},               # Has partner
    "Dependents": {"No": 0, "Yes": 1},            # Has dependents
    "PhoneService": {"No": 0, "Yes": 1},          # Phone service
    "PaperlessBilling": {"No": 0, "Yes": 1},      # Billing preference
}

# Numeric columns that need type coercion
NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]

def _serve_transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply identical feature transformations as used during model training.

    This function is CRITICAL for production ML - it ensures that features are
    transformed exactly as they were during training to prevent train/serve skew.

    Transformation Pipeline:
    1. Clean column names and handle data types
    2. Apply deterministic binary encoding (using BINARY_MAP)
    3. One-hot encode remaining categorical features
    4. Convert boolean columns to integers
    5. Align features with training schema and order

    Args:
        df: Single-row DataFrame with raw customer data

    Returns:
        DataFrame with features transformed and ordered for model input

    IMPORTANT: Any changes to this function must be reflected in training
    feature engineering to maintain consistency.
    """
    df = df.copy()

    # Clean column names (remove any whitespace)
    df.columns = df.columns.str.strip()

    # === STEP 1: Numeric Type Coercion ===
    # Ensure numeric columns are properly typed (handle string inputs)
    for c in NUMERIC_COLS:
        if c in df.columns:
            # Convert to numeric, replacing invalid values with NaN
            df[c] = pd.to_numeric(df[c], errors="coerce")
            # Fill NaN with 0 (same as training preprocessing)
            df[c] = df[c].fillna(0)

    # === STEP 2: Binary Feature Encoding ===
    # Apply deterministic mappings for binary features
    # CRITICAL: Must use exact same mappings as training
    for c, mapping in BINARY_MAP.items():
        if c in df.columns:
            df[c] = (
                df[c]
                .astype(str)                    # Convert to string
                .str.strip()                    # Remove whitespace
                .map(mapping)                   # Apply binary mapping
                .astype("Int64")                # Handle NaN values : pandas' nullable integer type - It allows 0,1 and NaN to coexist.
                .fillna(0)                      # Fill unknown values with 0
                .astype(int)                    # Final integer conversion
            )

    # === STEP 3: One-Hot Encoding for Remaining Categorical Features ===
    # Find remaining object/categorical columns (not in BINARY_MAP)
    obj_cols = [c for c in df.select_dtypes(include=["object"]).columns]
    if obj_cols:
        # Apply one-hot encoding with drop_first=True (same as training)
        # This prevents multicollinearity by dropping the first category. The dropped category becomes the reference/baseline category.
        df = pd.get_dummies(df, columns=obj_cols, drop_first=True)

    # === STEP 4: Boolean to Integer Conversion ===
    # Convert any boolean columns to integers 
    bool_cols = df.select_dtypes(include=["bool"]).columns
    if len(bool_cols) > 0:
        df[bool_cols] = df[bool_cols].astype(int)

    # === STEP 5: Feature Alignment with Training Schema ===
    # CRITICAL: Ensure features are in exact same order as training
    # Missing features get filled with 0, extra features are dropped
    df = df.reindex(columns=FEATURE_COLS, fill_value=0)
    #So this line effectively says: "Make this DataFrame look exactly like the feature matrix the model saw during training". That's crucial for production ML

    return df

def predict(input_dict: dict) -> str:
    """
    Main prediction function for customer churn inference.

    This function provides the complete inference pipeline from raw customer data
    to business-friendly prediction output. It's called by both the FastAPI endpoint
    and the Gradio interface to ensure consistent predictions.

    Pipeline:
    1. Convert input dictionary to DataFrame
    2. Apply feature transformations (identical to training)
    3. Generate model prediction using loaded model
    4. Convert prediction to user-friendly string

    Args:
        input_dict: Dictionary containing raw customer data with keys matching
                   the CustomerData schema (18 features total)

    Returns:
        Human-readable prediction string:
        - "Likely to churn" for high-risk customers (model prediction = 1)
        - "Not likely to churn" for low-risk customers (model prediction = 0)

    Example:
        >>> customer_data = {
        ...     "gender": "Female", "tenure": 1, "Contract": "Month-to-month",
        ...     "MonthlyCharges": 85.0, ... # other features
        ... }
        >>> predict(customer_data)
        "Likely to churn"
    """

    # === STEP 1: Convert Input to DataFrame ===
    #input_dict is one dictionary representing one customer.
    # Wrapping it in [] creates a list containing one dictionary (list containing 1 row)
    # Because pd.DataFrame() expects multiple rows
    # Create single-row DataFrame for pandas transformations
    df = pd.DataFrame([input_dict])

    # === STEP 2: Apply Feature Transformations ===
    # Use the same transformation pipeline as training
    df_enc = _serve_transform(df)

    # === STEP 3: Generate Model Prediction ===
    # Call the loaded MLflow model for inference
    # The model returns predictions in various formats depending on the ML library
    try:
        preds = model.predict(df_enc)

        # Normalize prediction output to consistent format
        if hasattr(preds, "tolist"): #asks: Does this object have an attribute/method called tolist?
            preds = preds.tolist()  # Convert numpy array to list

        # Extract single prediction value (for single-row input)
        if isinstance(preds, (list, tuple)) and len(preds) == 1: #asking: Is preds a list or tuple with exactly one element?
            result = preds[0] #Extract the actual prediction (0 or 1) from the list/tuple
        else:
            result = preds 
            # If the model returns something that isn't a single-item list/tuple, the code simply keeps it unchanged.
            #If the model returned a single scalar (e.g., 0 or 1), use it directly

    except Exception as e:
        raise Exception(f"Model prediction failed: {e}") #more descriptive error message

    # === STEP 4: Convert to Business-Friendly Output ===
    # Convert binary prediction (0/1) to actionable business language
    if result == 1:
        return "Likely to churn"      # High risk - needs intervention
    else:
        return "Not likely to churn"  # Low risk - maintain normal service


# TO EXECUTE THIS FILE:
# python .\src\serving\inference.py