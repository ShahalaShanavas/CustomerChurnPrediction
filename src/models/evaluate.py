from sklearn.metrics import classification_report, confusion_matrix


def evaluate_model(model, X_test, y_test):
    """
    Evaluates a LightGBM model on test data.

    Args:
        model: Trained LGBMClassifier model.
        X_test: Test features.
        y_test: Test labels.
    """
    y_pred = model.predict(X_test)

    print("Classification Report:\n")
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix:\n")
    print(confusion_matrix(y_test, y_pred))