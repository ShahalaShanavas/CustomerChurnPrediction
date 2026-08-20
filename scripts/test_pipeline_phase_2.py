import sys
from pathlib import Path

import mlflow
import pandas as pd

# -------------------------------------------------
# Project Setup
# -------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from models.train import train_model
from models.evaluate import evaluate_model


def main():
    print("=" * 60)
    print("PHASE 2 : MODEL TRAINING & EVALUATION")
    print("=" * 60)

    # -------------------------------------------------
    # Configure MLflow
    # -------------------------------------------------
    mlruns_dir = PROJECT_ROOT / "mlruns"
    mlruns_dir.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(mlruns_dir.resolve().as_uri())
    mlflow.set_experiment("CustomerChurnPrediction")

    print(f"\nMLflow Tracking URI : {mlflow.get_tracking_uri()}")

    # -------------------------------------------------
    # Load Processed Dataset
    # -------------------------------------------------
    data_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "telco_churn_processed.csv"
    )

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found:\n{data_path}")

    df = pd.read_csv(data_path)

    print(f"\nDataset Shape : {df.shape}")
    print("\nTarget Distribution:")
    print(df["Churn"].value_counts())

    # -------------------------------------------------
    # Train Model
    # -------------------------------------------------
    print("\nTraining LightGBM model...\n")

    try:
        model, X_test, y_test = train_model(
            df=df,
            target_col="Churn",
            tune=True,
        )

        print("\n✓ Model training completed.")

    except Exception as e:
        print("\n✗ Training failed!")
        raise RuntimeError(f"Training failed: {e}") from e

    # -------------------------------------------------
    # Evaluate Model
    # -------------------------------------------------
    print("\n" + "=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)

    evaluate_model(
        model=model,
        X_test=X_test,
        y_test=y_test,
    )

    print("\n✓ Pipeline completed successfully.")


if __name__ == "__main__":
    main()