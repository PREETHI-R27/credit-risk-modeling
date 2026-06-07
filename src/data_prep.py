"""Data preparation and preprocessing utilities for Credit Risk Modeling."""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import re


def load_and_clean_data(filepath='../data/german_credit.csv'):
    """Load and perform initial cleaning."""
    df = pd.read_csv(filepath)
    # No missing values in German dataset
    return df


def engineer_features(df):
    """Create domain-relevant engineered features."""
    df = df.copy()

    # Credit-to-Age ratio (proxy for credit burden vs stability)
    df['Credit_to_Age_Ratio'] = df['Credit_amount'] / df['Age_in_years']

    # Debt burden proxy
    df['Debt_Burden'] = df['Credit_amount'] * (df['Installment_rate_in_percentage_of_disposable_income'] / 100)

    # Purpose categories
    purpose_risk_map = {
        'A40': 'car_new', 'A41': 'car_used', 'A42': 'furniture_equipment',
        'A43': 'radio_television', 'A44': 'domestic_appliances', 'A45': 'repairs',
        'A46': 'education', 'A47': 'vacation', 'A48': 'retraining', 'A49': 'business', 'A410': 'others'
    }
    df['Purpose_Category'] = df['Purpose'].map(purpose_risk_map).fillna('others')

    # Employment length
    emp_map = {
        'A71': 'unemployed', 'A72': '<1_year', 'A73': '1-4_years',
        'A74': '4-7_years', 'A75': '>=7_years'
    }
    df['Employment_Length'] = df['Present_employment_since'].map(emp_map).fillna('unknown')

    # Checking account status
    checking_map = {
        'A11': 'negative', 'A12': '0-200', 'A13': '>=200', 'A14': 'no_account'
    }
    df['Checking_Account_Status'] = df['Status_of_existing_checking_account'].map(checking_map).fillna('unknown')

    # Credit history category
    credit_hist_map = {
        'A30': 'no_credits', 'A31': 'all_paid_duly', 'A32': 'existing_paid_duly',
        'A33': 'delay_past', 'A34': 'critical_account'
    }
    df['Credit_History_Category'] = df['Credit_history'].map(credit_hist_map).fillna('unknown')

    return df


def encode_features(df):
    """One-hot encode categorical features."""
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    return df_encoded


def preprocess_pipeline(filepath='../data/german_credit.csv', test_size=0.2, random_state=42):
    """End-to-end preprocessing pipeline."""
    # Load
    df = load_and_clean_data(filepath)

    # Target
    y = df['Credit_risk']
    X = df.drop('Credit_risk', axis=1)

    # Feature engineering
    X = engineer_features(X)

    # Encoding
    X = encode_features(X)

    # Sanitize feature names for XGBoost (no [, ] or <)
    regex = re.compile(r"\[|\]|<", re.IGNORECASE)
    X.columns = [regex.sub("_", col) if any(x in str(col) for x in ['[', ']', '<']) else col for col in X.columns]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Scale
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns, index=X_test.index
    )

    # SMOTE
    smote = SMOTE(random_state=random_state)
    X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)

    # Ensure X_train_res is a DataFrame with original column names
    if not isinstance(X_train_res, pd.DataFrame):
        X_train_res = pd.DataFrame(X_train_res, columns=X_train_scaled.columns)

    return X_train_res, X_test_scaled, y_train_res, y_test, scaler


if __name__ == "__main__":
    print("Testing Data Preparation Pipeline...")
    try:
        X_train, X_test, y_train, y_test, scaler = preprocess_pipeline('../data/german_credit.csv')
        print(f"Success! Training set shape: {X_train.shape}")
        print(f"Resampled Target Distribution:\n{y_train.value_counts()}")
    except Exception as e:
        print(f"Error during testing: {e}")
