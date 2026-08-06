
"""
LightGBM Training Pipeline
----------------------------------------------------
Pipeline:
1. Load Data
2. Validate Data
3. Preprocess Data
4. Feature Engineering
5. Train/Test Split
6. Train LightGBM
7. Evaluate
8. Log with MLflow
9. Save Model
----------------------------------------------------
"""

import sys
import time
import json
import argparse
import joblib
from pathlib import Path
import pandas as pd
import mlflow
import mlflow.sklearn

from lightgbm import LGBMClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
)

# --------------------------------------------------
# Allow importing from src/
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

# --------------------------------------------------
# Local modules
# --------------------------------------------------

from src.data.load_data import load_data
from src.data.preprocess_data import preprocess_data
from src.features.build_features import build_features
from src.utils.validate_data import validate_telco_data
from src.models.tune import tune_model


# --------------------------------------------------
# Main Pipeline
# --------------------------------------------------

def main(args):

    # ----------------------------------------------
    # MLflow Setup (Windows Compatible)
    # ----------------------------------------------

    tracking_uri = (
        args.mlflow_uri
        if args.mlflow_uri
        else f"file:///{(PROJECT_ROOT / 'mlruns').as_posix()}"
    )

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(args.experiment)

    with mlflow.start_run():

        mlflow.log_param("model", "LightGBM")
        mlflow.log_param("threshold", args.threshold)
        mlflow.log_param("test_size", args.test_size)

        # ------------------------------------------
        # Load Dataset
        # ------------------------------------------

        print("=" * 60)
        print("Loading dataset...")
        print("=" * 60)

        df = load_data(args.input)

        print(f"Rows    : {df.shape[0]}")
        print(f"Columns : {df.shape[1]}")

        # ------------------------------------------
        # Validate Dataset
        # ------------------------------------------

        print("\nRunning data validation...")

        # Convert TotalCharges to numeric (recommended)
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

        valid, failed = validate_telco_data(df)

        mlflow.log_metric(
            "data_validation_passed",
            int(valid),
        )

        if not valid:

            mlflow.log_text(
                json.dumps(failed, indent=2),
                "failed_expectations.json",
            )

            raise ValueError(
                f"Validation failed:\n{failed}"
            )

        print("Validation passed.")

        # ------------------------------------------
        # Preprocessing
        # ------------------------------------------

        print("\nPreprocessing data...")

        df = preprocess_data(df)

        processed_dir = PROJECT_ROOT / "data" / "processed"
        processed_dir.mkdir(parents=True, exist_ok=True)

        processed_file = (
            processed_dir /
            "telco_churn_processed.csv"
        )

        df.to_csv(
            processed_file,
            index=False,
        )

        print(
            f"Processed dataset saved:\n{processed_file}"
        )

        # ------------------------------------------
        # Feature Engineering
        # ------------------------------------------

        target = args.target

        if target not in df.columns:
            raise ValueError(
                f"Target column '{target}' not found."
            )

        print("\nBuilding features...")

        df_enc = build_features(
            df,
            target_col=target,
        )

        bool_cols = df_enc.select_dtypes(
            include=["bool"]
        ).columns

        for col in bool_cols:
            df_enc[col] = df_enc[col].astype(int)

        print(
            f"Final feature count: "
            f"{df_enc.shape[1]-1}" #Why subtract 1? One column is the target
        )
            # ------------------------------------------
        # Save Feature Metadata
        # ------------------------------------------

        artifacts_dir = PROJECT_ROOT / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        feature_columns = list(
            df_enc.drop(columns=[target]).columns
        )

        feature_json = artifacts_dir / "feature_columns.json"

        with open(feature_json, "w") as f:
            json.dump(feature_columns, f, indent=2)

        preprocessing_artifact = {
            "feature_columns": feature_columns,
            "target": target,
        }

        preprocessing_file = (
            artifacts_dir / "preprocessing.pkl"
        )

        joblib.dump(
            preprocessing_artifact,
            preprocessing_file,
        )

        mlflow.log_artifact(str(preprocessing_file))
        mlflow.log_text(
            "\n".join(feature_columns),
            artifact_file="feature_columns.txt",
        )

        print(
            f"Saved {len(feature_columns)} feature columns."
        )

        # ------------------------------------------
        # Train / Test Split
        # ------------------------------------------

        print("\nSplitting dataset...")

        X = df_enc.drop(columns=[target])
        y = df_enc[target]

        X_train, X_test, y_train,y_test = train_test_split(
            X,
            y,
            test_size=args.test_size,
            stratify=y,
            random_state=42,
        )

        print(
            f"Training samples : {len(X_train)}")

        print(
            f"Testing samples  : {len(X_test)}")

        # ------------------------------------------
        # Handle Class Imbalance
        # ------------------------------------------

        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

        mlflow.log_param(
            "scale_pos_weight",
            scale_pos_weight,
        )

        print(
            f"Scale Pos Weight : "
            f"{scale_pos_weight:.2f}"
        )

        # ------------------------------------------
        # LightGBM Model
        # ------------------------------------------

        print("\nInitializing LightGBM...")
        

        best_params = {
            "n_estimators": 463,
            "learning_rate": 0.13587526633713798,
            "max_depth": 3,
            "num_leaves": 122,
            "min_child_samples": 10,
            "colsample_bytree": 0.5995415565396105,
            "subsample": 0.905495456894141,
            "reg_alpha": 4.974993568902758,
            "reg_lambda": 2.0560190920864265,
            "boosting_type": "gbdt",
            "objective": "binary",
            "random_state": 42,
            "n_jobs": -1,
            "verbosity": -1,
            "scale_pos_weight": 2.768561872909699
            }
        
        print("\nBest Hyperparameters")

        for key, value in best_params.items():
            print(f"{key}: {value}")

        

        # Save best parameters
        best_params_file = artifacts_dir / "best_params.json"
        
        with open(best_params_file, "w") as f:
            json.dump(best_params, f, indent=4)
        
        print(f"Best parameters saved to: {best_params_file}")

        # Log tuned parameters
        mlflow.log_params(best_params)
        mlflow.log_artifact(str(best_params_file))
        

        # ------------------------------------------
        # Train Final Model
        # ------------------------------------------

        print("\nTraining final LightGBM model...")

        train_start = time.time()

        model = LGBMClassifier(**best_params)

        model.fit(
        X_train,
        y_train,
        )

        train_time = time.time() - train_start

        mlflow.log_metric(
        "train_time_seconds",
        train_time,
        )

        print(
            f"Training completed "
            f"in {train_time:.2f} sec"
        )

        # ------------------------------------------
        # Model Prediction
        # ------------------------------------------

        print("\nEvaluating model...")

        prediction_start = time.time()

        # Predict probabilities
        probabilities = model.predict_proba(X_test)[:, 1]

        # Apply threshold
        predictions = (
            probabilities >= args.threshold
        ).astype(int)

        prediction_time = (
            time.time() - prediction_start
        )

        mlflow.log_metric(
            "prediction_time_seconds",
            prediction_time,
        )

        # ------------------------------------------
        # Evaluation Metrics
        # ------------------------------------------

        precision = precision_score(
            y_test,
            predictions,
        )

        recall = recall_score(
            y_test,
            predictions,
        )

        f1 = f1_score(
            y_test,
            predictions,
        )

        roc_auc = roc_auc_score(
            y_test,
            probabilities,
        )

        mlflow.log_metric(
            "precision",
            precision,
        )

        mlflow.log_metric(
            "recall",
            recall,
        )

        mlflow.log_metric(
            "f1_score",
            f1,
        )

        mlflow.log_metric(
            "roc_auc",
            roc_auc,
        )

        # ------------------------------------------
        # Display Results
        # ------------------------------------------

        print("\n" + "=" * 60)
        print("MODEL PERFORMANCE")
        print("=" * 60)

        print(f"Precision : {precision:.4f}")
        print(f"Recall    : {recall:.4f}")
        print(f"F1 Score  : {f1:.4f}")
        print(f"ROC AUC   : {roc_auc:.4f}")

        print("\nClassification Report")
        print("-" * 60)

        print(
            classification_report(
                y_test,
                predictions,
                digits=4,
            )
        )

        # ------------------------------------------
        # Log Additional Metrics
        # ------------------------------------------

        mlflow.log_metric(
            "training_samples",
            len(X_train),
        )

        mlflow.log_metric(
            "testing_samples",
            len(X_test),
        )

        mlflow.log_metric(
            "num_features",
            X_train.shape[1],
        )

        mlflow.log_metric(
            "positive_class_train",
            int((y_train == 1).sum()),
        )

        mlflow.log_metric(
            "negative_class_train",
            int((y_train == 0).sum()),
        )

        print("\nMetrics successfully logged to MLflow.")

        # ------------------------------------------
        # Save Model to MLflow
        # ------------------------------------------

        print("\nSaving model to MLflow...")

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
        )

        print("Model successfully logged.")

        # ------------------------------------------
        # Save Local Backup (Optional)
        # ------------------------------------------

        local_model_dir = PROJECT_ROOT / "artifacts"
        local_model_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        local_model_path = (
            local_model_dir / "lightgbm_model.pkl"
        )

        joblib.dump(
            model,
            local_model_path,
        )

        print(
            f"Local model saved to:\n{local_model_path}"
        )

        # ------------------------------------------
        # Final Summary
        # ------------------------------------------

        print("\n" + "=" * 60)
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print("=" * 60)

        print(f"Training Time      : {train_time:.2f} sec")
        print(f"Prediction Time    : {prediction_time:.4f} sec")
        print(f"Training Samples   : {len(X_train)}")
        print(f"Testing Samples    : {len(X_test)}")
        print(f"Number of Features : {X_train.shape[1]}")
        print(f"Precision          : {precision:.4f}")
        print(f"Recall             : {recall:.4f}")
        print(f"F1 Score           : {f1:.4f}")
        print(f"ROC-AUC            : {roc_auc:.4f}")

        print("=" * 60)


# --------------------------------------------------
# Entry Point
# --------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Customer Churn Training Pipeline using LightGBM"
    )

    parser.add_argument(
        "--input",
        required=True,
        type=str,
        help="Path to raw CSV dataset",
    )

    parser.add_argument(
        "--target",
        default="Churn",
        type=str,
        help="Target column name",
    )

    parser.add_argument(
        "--threshold",
        default=0.30,
        type=float,
        help="Classification threshold",
    )

    parser.add_argument(
        "--test_size",
        default=0.20,
        type=float,
        help="Test split ratio",
    )

    parser.add_argument(
        "--experiment",
        default="Telco Churn",
        type=str,
        help="MLflow experiment name",
    )

    parser.add_argument(
        "--mlflow_uri",
        default=None,
        type=str,
        help="Optional MLflow tracking URI",
    )

    args = parser.parse_args() #Reads the command-line arguments and stores them in an object.

    main(args) #Calls your main() function and passes the parsed arguments.


# Use this below to run the pipeline:
# python scripts/run_pipeline_final.py --input data/raw/TelcoCustomerChurn.csv --target Churn
