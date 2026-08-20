import os
import pandas as pd
import sys

sys.path.append(os.path.abspath("src")) 
#adds the absolute path of the src directory to Python's module search path(sys.path), allowing python to search modules in that directory(src) and  to import modules from that directory.

#Import project functions
from data.load_data import load_data #Imports the function load_data() from src/data/load_data.py
from data.preprocess_data import preprocess_data #Imports the preprocessing function.
from features.build_features import build_features #Imports the feature engineering function.

#Configuration

DATA_PATH = "C:/Users/thoufik rifayi/Desktop/Projects/CustomerChurnPrediction/data/raw/TelcoCustomerChurn.csv" #Stores the location of the CSV file.
TARGET_COL = "Churn" #Stores the name of the target column.

def main():
    print("=== Testing Phase 1: Load → Preprocess → Build Features ===")

    #Step 1 — Load Data
    print("\n[1] Loading Data....")
    df=load_data(DATA_PATH)
    print(f"Data loaded. Shape: {df.shape}")
    print(df.head(3))

    #Step 2 — Preprocessing
    print("\n[2] Preprocesing Data....")
    df_clean  = preprocess_data(df,target_col=TARGET_COL)
    print(f"Data after Preprocessing. Shape: {df_clean.shape}")
    print(df_clean.head(3))

    #Step 3 — Feature Engineering
    print("\n[3] Building features....")
    df_features = build_features(df_clean, target_col=TARGET_COL)
    print(f"Data after feature engineering. Shape: {df_features.shape}")
    print(df_features.head(3))

    print("\n✅ Phase 1 pipeline completed successfully!")


if __name__ == "__main__" :
    main()











