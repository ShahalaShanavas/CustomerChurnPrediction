import mlflow
import mlflow.lightgbm
import pandas as pd

from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score

from .tune import tune_model


def train_model(df: pd.DataFrame, target_col: str, tune: bool = True):
    """
    Trains a LightGBM model and logs it with MLflow.

    Args:
        df (pd.DataFrame): Feature dataset.
        target_col (str): Name of the target column.
        tune (bool): Whether to tune hyperparameters using Optuna.
    """

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # -----------------------------
    # Hyperparameter Tuning
    # -----------------------------
    if tune:
        print("Running Optuna hyperparameter tuning...")
        best_params = tune_model(X_train, y_train)
    else:
        best_params = {
            "n_estimators": 300,
            "learning_rate": 0.1,
            "max_depth": 6,
            "random_state": 42,
            "n_jobs": -1,
            "objective": "binary"
        }

    # Ensure required parameters exist
    best_params["random_state"] = 42
    best_params["n_jobs"] = -1
    best_params["objective"] = "binary"
    best_params["class_weight"] = {0:1, 1:(y_train==0).sum()/(y_train==1).sum()}

    model = LGBMClassifier(**best_params)

    with mlflow.start_run():

        # Log Parameters
        mlflow.log_params(best_params)

        # Train
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("recall", rec)

        # Log model
        mlflow.lightgbm.log_model(
            model,
            artifact_path="model"
        )

        # Log dataset
        train_ds = mlflow.data.from_pandas(
            df,
            source="training_data"
        )
        mlflow.log_input(train_ds, context="training")

        print(f"Accuracy : {acc:.4f}")
        print(f"Recall   : {rec:.4f}")

    return model, X_test, y_test