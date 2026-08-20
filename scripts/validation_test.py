import pandas as pd
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Local modules - Core pipeline components
from src.data.load_data import load_data                 
from src.utils.validate_data import validate_telco_data

df = load_data("data/raw/TelcoCustomerChurn.csv")

# Convert TotalCharges to numeric (recommended)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Run validation
is_valid, failed_checks = validate_telco_data(df)

if is_valid:
    print("Dataset is valid. Proceed with preprocessing.")
else:
    print("Dataset validation failed.")
    print(failed_checks)