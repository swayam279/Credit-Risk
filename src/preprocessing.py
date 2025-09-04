# In your src/preprocessing.py file

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def preprocess_data(df):
    """
    Preprocesses the Home Credit Default Risk dataset.

    This function performs the following steps:
    1. Selects numerical and categorical features.
    2. Imputes missing numerical values with the median.
    3. Scales numerical features using StandardScaler.
    4. One-hot encodes categorical features.
    5. Combines the processed numerical and categorical features.

    Args:
        df (pd.DataFrame): The input dataframe (e.g., application_train).

    Returns:
        pd.DataFrame: A fully preprocessed dataframe ready for modeling.
        list: A list of the final feature names.
    """
    # Isolate numerical and categorical columns
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    numeric_cols.remove('TARGET')
    numeric_cols.remove('SK_ID_CURR')
    
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

    # Create copies to avoid modifying original data
    df_numeric = df[numeric_cols].copy()
    df_categorical = df[categorical_cols].copy()

    # Impute and scale numerical data
    imputer = SimpleImputer(strategy='median')
    df_numeric_imputed = pd.DataFrame(imputer.fit_transform(df_numeric), columns=numeric_cols)
    
    scaler = StandardScaler()
    df_numeric_scaled = pd.DataFrame(scaler.fit_transform(df_numeric_imputed), columns=numeric_cols)

    # One-hot encode categorical data
    df_categorical_encoded = pd.get_dummies(df_categorical, handle_unknown='ignore')
    
    # Combine processed data
    processed_df = pd.concat([df_numeric_scaled, df_categorical_encoded], axis=1)
    
    return processed_df, processed_df.columns.tolist()