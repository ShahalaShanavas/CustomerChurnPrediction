import pandas as pd




def preprocess_data(
    df: pd.DataFrame,
    target_col: str = "Churn",
) -> pd.DataFrame:
    """
    Preprocess the Telco Customer Churn dataset.

    Steps:
    1. Trim whitespace from column names.
    2. Remove customer ID columns.
    3. Convert target ('Yes'/'No') to binary (1/0).
    4. Convert TotalCharges to numeric.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    target_col : str, default="Churn"
        Name of the target column.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe.
    """
    df = df.copy()

    # Clean column names
    df.columns = df.columns.str.strip()

    # Remove ID columns
    # drop ids if present
    for col in ["customerID", "CustomerID", "customer_id"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Encode target column
    if target_col in df.columns and df[target_col].dtype == "object":
        df[target_col] = (
            df[target_col]
            .str.strip()
            .map({"No": 0, "Yes": 1})
        )

    # Convert TotalCharges to numeric
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(
            df["TotalCharges"],
            errors="coerce",
        )

    
    

    return df