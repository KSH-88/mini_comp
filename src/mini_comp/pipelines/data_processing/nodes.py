from typing import Tuple

import pandas as pd


def split_cities(initial_data_train_labels: pd.DataFrame, initial_data_train_features: pd.DataFrame, initial_data_test_features: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the initial data into separate dataframes for San Juan and Iquitos.

    Args:
        initial_data_train_labels (pd.DataFrame): The initial data train labels dataframe.
        initial_data_train_features (pd.DataFrame): The initial data train features dataframe.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]: A tuple containing the following dataframes:
            - sj_train_features: The train features dataframe for San Juan.
            - sj_train_labels: The train labels dataframe for San Juan.
            - iq_train_features: The train features dataframe for Iquitos.
            - iq_train_labels: The train labels dataframe for Iquitos.
    """

    # Seperate data for San Juan
    sj_train_features = initial_data_train_features.loc['sj']
    sj_train_labels = initial_data_train_labels.loc['sj']

    # Separate data for Iquitos
    iq_train_features = initial_data_train_features.loc['iq']
    iq_train_labels = initial_data_train_labels.loc['iq']

    sj_test_features = initial_data_test_features.loc['sj']
    iq_test_features = initial_data_test_features.loc['iq']

    return sj_train_features, sj_train_labels, iq_train_features, iq_train_labels, sj_test_features, iq_test_features


def preprocess_cities(sj_train_features: pd.DataFrame, iq_train_features: pd.DataFrame, sj_train_labels: pd.DataFrame, iq_train_labels: pd.DataFrame,
                      sj_test_features: pd.DataFrame, iq_test_features: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Preprocesses the city train features and labels.

    Args:
        sj_train_features (pd.DataFrame): The San Juan train features.
        iq_train_features (pd.DataFrame): The Iquitos train features.
        sj_train_labels (pd.DataFrame): The San Juan train labels.
        iq_train_labels (pd.DataFrame): The Iquitos train labels.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]: A tuple containing the preprocessed San Juan train data and labels, and the preprocessed Iquitos train data and labels.
    """

    # Fill missing values with the mean of the column
    sj_train_features_imputed = sj_train_features.fillna(
        method='ffill').round(2)
    iq_train_features_imputed = iq_train_features.fillna(
        method='ffill').round(2)

    sj_test_features_imputed = sj_test_features.fillna(method='ffill').round(2)
    iq_test_features_imputed = iq_test_features.fillna(method='ffill').round(2)

    # select features we want
    features = ['reanalysis_specific_humidity_g_per_kg',
                'reanalysis_dew_point_temp_k',
                'station_avg_temp_c',
                'station_min_temp_c']

    sj_train_features = sj_train_features_imputed[features]
    iq_train_features = iq_train_features_imputed[features]
    sj_unseen_test = sj_test_features_imputed[features]
    iq_unseen_test = iq_test_features_imputed[features]

    # Join the features and labels
    sj_train = sj_train_features.join(sj_train_labels)
    iq_train = iq_train_features.join(iq_train_labels)

    return sj_train, iq_train, sj_unseen_test, iq_unseen_test
