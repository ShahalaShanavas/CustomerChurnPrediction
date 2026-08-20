import optuna
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_score


def tune_model(X, y):
    """
    Tunes a LightGBM model using Optuna.

    Args:
        X (pd.DataFrame): Features.
        y (pd.Series): Target.
    """

    def objective(trial):
        params = {"n_estimators": trial.suggest_int("n_estimators", 300, 800), 
                  "learning_rate": trial.suggest_float("learning_rate", 0.01,0.2),
                  "max_depth" : trial.suggest_int("max_depth", 3,10),
                  "num_leaves" : trial.suggest_int("num_leaves", 20,150),
                  "min_child_samples" : trial.suggest_int("min_child_samples", 10,100), 
                  "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0), 
                  "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                  "reg_alpha": trial.suggest_float("reg_alpha", 0, 5), 
                  "reg_lambda": trial.suggest_float("reg_lambda", 0, 5), 
                  "random_state": 42,
                  "n_jobs": -1
                  }

        model = LGBMClassifier(**params)

        scores = cross_val_score(
            model,
            X,
            y,
            cv=3,
            scoring="recall",
            n_jobs=-1
        )

        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30)

    print("Best Params:", study.best_params)
    print("Best Recall:", study.best_value)

    return study.best_params