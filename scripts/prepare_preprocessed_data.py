import os
import sys
import pandas as pd

# Make src importable
sys.path.append(os.path.abspath("src"))

from data.load_data import load_data
from data.preprocess_data import preprocess_data
from features.build_features import build_features

# Configuration
DATA_PATH = "C:/Users/thoufik rifayi/Desktop/Projects/CustomerChurnPrediction/data/raw/TelcoCustomerChurn.csv"
OUTPUT_PATH = "C:/Users/thoufik rifayi/Desktop/Projects/CustomerChurnPrediction/data/processed/telco_churn_processed.csv"
TARGET_COL = "Churn"


def main():
    print("=== Creating Processed Dataset ===")

    # Step 1: Load data
    df = load_data(DATA_PATH)

    # Step 2: Preprocess
    df_clean = preprocess_data(df, target_col=TARGET_COL)

    # Step 3: Feature engineering
    df_processed = build_features(df_clean, target_col=TARGET_COL)

    # Ensure output folder exists
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    # Save processed dataset
    df_processed.to_csv(OUTPUT_PATH, index=False)

    print(f"Processed dataset saved to: {OUTPUT_PATH}")
    print(f"Shape: {df_processed.shape}")


if __name__ == "__main__":
    main()