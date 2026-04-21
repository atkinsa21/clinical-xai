import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

DATA_PATH = (
    Path(__file__).resolve().parent
    / "datasets"
    / "diabetes_binary_health_indicators_BRFSS2015.csv"
)
DF = pd.read_csv(DATA_PATH)
TARGET_COLUMN = "Diabetes_binary"


def load_diabetes_dataset():
    """Load Kaggle diabetes binary classification data from a CSV file and split it into training and testing sets.

    Returns:
        X_train (pd.DataFrame): Training features.
        X_test (pd.DataFrame): Testing features.
        y_train (pd.Series): Training target variable.
        y_test (pd.Series): Testing target variable.
    """
    df = DF.copy()
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test
