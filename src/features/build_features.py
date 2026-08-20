import pandas as pd


def _map_binary_series(s: pd.Series) -> pd.Series:
    """
    Convert a binary categorical column into 0/1.

    Deterministic mappings:
        Yes/No      -> No=0, Yes=1
        Male/Female -> Female=0, Male=1

    Any other binary feature is mapped alphabetically:
        First value  -> 0
        Second value -> 1

    Returns a Pandas nullable integer (Int64).
    """

    # Get unique non-null values
    vals = list(s.dropna().unique())

    # Convert to string only for comparison
    valset = {str(v) for v in vals}

    # -----------------------------
    # Hardcoded mappings
    # -----------------------------
    if valset == {"Yes", "No"}:
        return s.map({"No": 0, "Yes": 1}).astype("Int64")

    if valset == {"Male", "Female"}:
        return s.map({"Female": 0, "Male": 1}).astype("Int64")

    # -----------------------------
    # Generic mapping
    # -----------------------------
    if len(vals) == 2:

        # Sort alphabetically for deterministic mapping
        sorted_vals = sorted(vals)

        mapping = {
            sorted_vals[0]: 0,
            sorted_vals[1]: 1
        }

        return s.map(mapping).astype("Int64")

    # Return unchanged if not binary
    return s


def build_features(
    df: pd.DataFrame,
    target_col: str = "Churn"
) -> pd.DataFrame:
    """
    Complete feature engineering pipeline.

    Steps
    -----
    1. Copy data
    2. Identify categorical columns
    3. Separate binary and multi-category columns
    4. Binary encode
    5. One-hot encode
    6. Convert booleans to integers
    7. Convert nullable integers to standard integers

    Returns
    -------
    Processed DataFrame ready for ML.
    """

    # ---------------------------------------------------
    # Step 1 : Copy dataframe
    # ---------------------------------------------------
    df = df.copy()

    print(f"Starting feature engineering on {df.shape[1]} columns")

    # ---------------------------------------------------
    # Step 2 : Identify feature types
    # ---------------------------------------------------

    # Find all categorical columns except target
    obj_cols = [
        c
        for c in df.select_dtypes(include="object").columns
        if c != target_col
    ]

    # Find numeric columns
    numeric_cols = df.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    print(
        f"Found {len(obj_cols)} categorical "
        f"and {len(numeric_cols)} numeric columns"
    )

    # ---------------------------------------------------
    # Step 3 : Split categorical columns
    # ---------------------------------------------------

    binary_cols = [
        c
        for c in obj_cols
        if df[c].dropna().nunique() == 2
    ]

    multi_cols = [
        c
        for c in obj_cols
        if df[c].dropna().nunique() > 2
    ]

    print(
        f"Binary features: {len(binary_cols)} | "
        f"Multi-category features: {len(multi_cols)}"
    )

    # ---------------------------------------------------
    # Step 4 : Binary encoding
    # ---------------------------------------------------

    for c in binary_cols:

        original_dtype = df[c].dtype

        # Convert binary values into 0/1
        df[c] = _map_binary_series(df[c])

        print(
            f"{c}: {original_dtype} -> binary (0/1)"
        )

    # ---------------------------------------------------
    # Step 5 : One-Hot Encoding
    # ---------------------------------------------------

    if multi_cols:

        print(
            f"Applying one-hot encoding "
            f"to {len(multi_cols)} columns..."
        )

        original_shape = df.shape

        # Convert multi-category columns into dummy variables
        df = pd.get_dummies(
            df,
            columns=multi_cols,
            drop_first=True
        )

        # Number of dummy columns created
        new_features = (
            df.shape[1]
            - original_shape[1]
            + len(multi_cols)
        )

        print(
            f"Created {new_features} new features"
        )

    # ---------------------------------------------------
    # Step 6 : Convert boolean columns
    # ---------------------------------------------------

    # Some dummy columns may be boolean
    bool_cols = df.select_dtypes(include="bool").columns

    if len(bool_cols):

        df[bool_cols] = df[bool_cols].astype(int)

        print(
            f"Converted {len(bool_cols)} "
            f"boolean columns to integers"
        )

    # ---------------------------------------------------
    # Step 7 : Convert nullable integers
    # ---------------------------------------------------

    for c in binary_cols:

        if pd.api.types.is_integer_dtype(df[c]):

            # Replace missing values
            df[c] = df[c].fillna(0)

            # Convert Int64 -> int64
            df[c] = df[c].astype(int)

    print(
        f"Feature engineering complete : "
        f"{df.shape[1]} final features"
    )

    return df