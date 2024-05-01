import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd


def cap_extreme_outliers_iqr(data):
    """
    Caps extreme outliers from a dataframe based on the IQR method.
    Args:
    data (DataFrame): Input dataframe from which to cap outliers.

    Returns:
    DataFrame: New dataframe with extreme outliers capped.
    """
    for column in data.columns:
        Q1 = np.percentile(data[column], 25)
        Q3 = np.percentile(data[column], 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Cap extreme outliers
        data[column] = np.where(data[column] > upper_bound, upper_bound, data[column])
        data[column] = np.where(data[column] < lower_bound, lower_bound, data[column])

    return data

def cap_with_reference(data, reference_data):
    """
    Caps extreme outliers from a dataframe based on the IQR method derived from a reference dataset.
    
    Args:
    data (DataFrame): Input dataframe to cap outliers.
    reference_data (DataFrame): Dataset from which to derive IQR-based capping bounds.

    Returns:
    DataFrame: New dataframe with outliers capped.
    """
    for column in data.columns:
        Q1 = np.percentile(reference_data[column], 25)
        Q3 = np.percentile(reference_data[column], 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Cap extreme outliers
        data[column] = np.where(data[column] > upper_bound, upper_bound, data[column])
        data[column] = np.where(data[column] < lower_bound, lower_bound, data[column])

    return data


def scale_features(X_train, X_test):
    """
    Scales the training and test datasets using standardization and converts them back to DataFrame with original columns.

    Parameters:
    - X_train (DataFrame): Training data features to scale
    - X_test (DataFrame): Test data features to scale

    Returns:
    - X_train_scaled (DataFrame): Scaled training data as DataFrame
    - X_test_scaled (DataFrame): Scaled test data as DataFrame
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Recreate DataFrames to preserve column labels
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

    return X_train_scaled, X_test_scaled