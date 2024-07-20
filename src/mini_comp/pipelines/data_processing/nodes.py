from typing import Tuple

import pandas as pd
import numpy as np


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


def cycle_transform(data: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Transform a cyclical feature into two features: sin and cos of the feature.

    Args:
        data (pd.DataFrame): The data to transform.
        column (str): The column to transform.

    Returns:
        pd.DataFrame: The transformed data.
    """
    # data = data.reset_index()
    data[column + '_sin'] = np.sin(2 * np.pi *
                                   data[column] / data[column].max())
    data[column + '_cos'] = np.cos(2 * np.pi *
                                   data[column] / data[column].max())
    # data = data.set_index(['year', 'weekofyear'])
    # data.drop(column, axis=1, inplace=True)

    return data


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

    sj_train_features_imputed = sj_train_features_imputed.reset_index()
    iq_train_features_imputed = iq_train_features_imputed.reset_index()
    sj_test_features_imputed = sj_test_features_imputed.reset_index()
    iq_test_features_imputed = iq_test_features_imputed.reset_index()

    sj_train_features_imputed['weekofyearcopy'] = sj_train_features_imputed['weekofyear']
    iq_train_features_imputed['weekofyearcopy'] = iq_train_features_imputed['weekofyear']
    sj_test_features_imputed['weekofyearcopy'] = sj_test_features_imputed['weekofyear']
    iq_test_features_imputed['weekofyearcopy'] = iq_test_features_imputed['weekofyear']

    # Transform the weekofyear feature into two features
    sj_train_features_imputed = cycle_transform(
        sj_train_features_imputed, 'weekofyear')
    iq_train_features_imputed = cycle_transform(
        iq_train_features_imputed, 'weekofyear')
    sj_test_features_imputed = cycle_transform(
        sj_test_features_imputed, 'weekofyear')
    iq_test_features_imputed = cycle_transform(
        iq_test_features_imputed, 'weekofyear')

    # select features we want
    features = ['reanalysis_specific_humidity_g_per_kg',
                'reanalysis_dew_point_temp_k',
                'station_avg_temp_c',
                'station_min_temp_c',
                'reanalysis_relative_humidity_percent',
                'reanalysis_tdtr_k',
                'weekofyearcopy',
                'weekofyear_sin',
                'weekofyear_cos']

    sj_train_features_imputed = sj_train_features_imputed.set_index(
        ['year', 'weekofyear'])
    iq_train_features_imputed = iq_train_features_imputed.set_index(
        ['year', 'weekofyear'])
    sj_test_features_imputed = sj_test_features_imputed.set_index(
        ['year', 'weekofyear'])
    iq_test_features_imputed = iq_test_features_imputed.set_index(
        ['year', 'weekofyear'])

    sj_train_features = sj_train_features_imputed[features]
    iq_train_features = iq_train_features_imputed[features]
    sj_unseen_test = sj_test_features_imputed[features]
    iq_unseen_test = iq_test_features_imputed[features]

    # Join the features and labels
    sj_train = sj_train_features.join(sj_train_labels)
    iq_train = iq_train_features.join(iq_train_labels)

    sj_train['total_cases'] = np.log(1 + sj_train['total_cases'])
    iq_train['total_cases'] = np.log(1 + iq_train['total_cases'])

    print(sj_train.head())
    print(iq_train.head())

    return sj_train, iq_train, sj_unseen_test, iq_unseen_test
