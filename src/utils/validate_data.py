import pandas as pd
import great_expectations as ge
from typing import Tuple, List

def validate_telco_data(df) -> Tuple[bool,List[str]]:
    

    """
    Comprehensive data validation for Telco Customer Churn dataset using Great Expectations.

    Return type hint: a tuple containing a boolean and a list of strings.
    
    This function implements critical data quality checks that must pass before model training.
    It validates data integrity, business logic constraints, and statistical properties
    that the ML model expects.

    """

    print("Starting data validation with Great Expectations...")

    #### Convert DataFrame to Great Expectations dataset ###

    ge_df = ge.dataset.PandasDataset(df)

    print("Validating schema and required columns...")

    ### Checks that the column customerID is present.###
    #Required for identifying customers. If missing, validation fails.

    ge_df.expect_column_to_exist("customerID")

    ### Check customerID has no nulls ###
    #Ensures every row has a customer ID. Fails if any value is NaN or null.

    ge_df.expect_column_values_to_not_be_null("customerID")

    ### Check other required columns exist ###
    #If any column is missing, validation will fail.

    ## Core demographic features
    ge_df.expect_column_to_exist("gender") 
    ge_df.expect_column_to_exist("Partner")
    ge_df.expect_column_to_exist("Dependents")

    ## Service features (critical for churn analysis)
    ge_df.expect_column_to_exist("PhoneService")
    ge_df.expect_column_to_exist("InternetService")
    ge_df.expect_column_to_exist("Contract")

    ## Financial features (key churn predictors)
    ge_df.expect_column_to_exist("tenure")
    ge_df.expect_column_to_exist("MonthlyCharges")
    ge_df.expect_column_to_exist("TotalCharges")

    # ==============================
    # Business logic validation section
    # ==============================

    print("Validating business logic constraints...")


    ### Validate gender values ###
    ge_df.expect_column_values_to_be_in_set("gender", ["Male", "Female"])

    ### Validate Yes/No columns ###
    ge_df.expect_column_values_to_be_in_set("Partner",["Yes","No"])
    ge_df.expect_column_values_to_be_in_set("Dependents", ["Yes", "No"])
    ge_df.expect_column_values_to_be_in_set("PhoneService", ["Yes", "No"])

    ### Validate contract values ###
    ge_df.expect_column_values_to_be_in_set(
        "Contract", 
        ["Month-to-month", "One year", "Two year"]
    )

    ### Validate internet service values ###
    ge_df.expect_column_values_to_be_in_set(
        "InternetService",
        ["DSL", "Fiber optic", "No"]
    )

    # ==============================
    # Numeric range validation section 
    # ==============================

    print("Validating numeric ranges and business constraints...")

    ### Tenure must be non-negative 
    ge_df.expect_column_values_to_be_between("tenure", min_value=0)

    ### Monthly charges must be positive (business logic - no free service)
    ge_df.expect_column_values_to_be_between("MonthlyCharges", min_value=0)
    
    ### Total charges should be non-negative (business logic)
    ge_df.expect_column_values_to_be_between("TotalCharges", min_value=0)
    
    # ==============================
    # Statistical validation section
    # ==============================

    print("Validating statistical properties...")

    # Tenure should be reasonable (max ~10 years = 120 months for telecom)
    ge_df.expect_column_values_to_be_between("tenure", min_value=0, max_value=120)

    # Monthly charges should be within reasonable business range
    ge_df.expect_column_values_to_be_between("MonthlyCharges", min_value=0, max_value=200)

    # No missing values in critical numeric features  
    ge_df.expect_column_values_to_not_be_null("tenure")
    ge_df.expect_column_values_to_not_be_null("MonthlyCharges")

    # ==============================
    # Data consistency section
    # ==============================

    print("Validating data consistency...")

    ### Cross-column consistency check###
    # Total charges should generally be >= Monthly charges (except for very new customers)
    # This is a business logic check to catch data entry errors
    ge_df.expect_column_pair_values_A_to_be_greater_than_B(
        column_A="TotalCharges",
        column_B="MonthlyCharges",
        or_equal=True,
        mostly=0.95)
    #Allows equality. At least 95% of rows must satisfy the rule. Allows up to 5% of rows to fail without failing the entire expectation.

    # ==============================
    #Run validation suite
    # ==============================

    print("Running complete validation suite...")
    results = ge_df.validate() #Executes all expectations added earlier.

    #Prepare list for failed expectations
    #Creates an empty list. This will store failed expectation names.
    failed_expectations = []
    for r in results["results"]:
        if not r["success"]:
            expectation_type = r["expectation_config"]["expectation_type"]
            failed_expectations.append(expectation_type)

    # ==============================
    #Print validation Summary
    # ==============================

    ### Count validation results ###
    total_checks = len(results["results"])
    passed_checks = sum(1 for r in results["results"] if r["success"])
    failed_checks = total_checks - passed_checks

    ### Check overall validation success ###
    if results["success"]:
        print(f"Data validation PASSED: {passed_checks}/{total_checks} checks successful")
    else:
        print(f"Data validation FAILED: {failed_checks}/{total_checks} checks failed")
        print(f"Failed expectations: {failed_expectations}")
    
    return results["success"], failed_expectations