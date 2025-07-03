import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def clean_missing_values(df, strategy="drop"):
    """
    Handle missing values in the dataset.
    strategy: 'drop' or 'fill'
    """
    if strategy == "drop":
        df = df.dropna()
    elif strategy == "fill":
        df = df.fillna(df.mean(numeric_only=True))
    return df

def scale_features(X, numeric_features):
    """
    Returns a pipeline to scale numeric features.
    """
    scaler = StandardScaler()
    X_scaled = X.copy()
    X_scaled[numeric_features] = scaler.fit_transform(X[numeric_features])
    return X_scaled, scaler

def create_preprocessing_pipeline(numeric_features, categorical_features):
    """
