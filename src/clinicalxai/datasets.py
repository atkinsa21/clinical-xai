"""
Load and preprocess datasets for clinical machine learning tasks.
This module provides functions to load the dataset, split it into
training and testing sets, and prepare it for use in machine learning models.
"""

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

def load_diabetes_dataset():
    """Load Kaggle diabetes binary classification data from a CSV file and split it into training and testing sets.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
        A tuple containing the training features, testing features, training labels, and testing labels.
    """
    data_path = (
        Path(__file__).resolve().parent
        / "datasets"
        / "diabetes_binary_health_indicators_BRFSS2015.csv"
    )
    df_raw = pd.read_csv(data_path)
    target = "Diabetes_binary"

    df_copy = df_raw.copy()
    X = df_copy.drop(columns=target)
    y = df_copy[target]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test
