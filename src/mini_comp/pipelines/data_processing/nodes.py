from typing import Dict, Tuple

import pandas as pd
from pyspark.sql import Column
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql.functions import regexp_replace
from pyspark.sql.types import DoubleType


def _is_true(x: Column) -> Column:
    return x == "t"


def _parse_percentage(x: Column) -> Column:
    x = regexp_replace(x, "%", "")
    x = x.cast("float") / 100
    return x


def _parse_money(x: Column) -> Column:
    x = regexp_replace(x, "[$£€]", "")
    x = regexp_replace(x, ",", "")
    x = x.cast(DoubleType())
    return x


def preprocess_companies(companies: SparkDataFrame) -> Tuple[SparkDataFrame, Dict]:
    """Preprocesses the data for companies.

    Args:
        companies: Raw data.
    Returns:
        Preprocessed data, with `company_rating` converted to a float and
        `iata_approved` converted to boolean.
    """
    companies = companies.withColumn(
        "iata_approved", _is_true(companies.iata_approved))
    companies = companies.withColumn(
        "company_rating", _parse_percentage(companies.company_rating))

    # Drop columns that aren't used for model training
    companies = companies.drop('company_location', 'total_fleet_count')
    return companies, {"columns": companies.columns, "data_type": "companies"}


def load_shuttles_to_csv(shuttles: pd.DataFrame) -> pd.DataFrame:
    """Load shuttles to csv because it's not possible to load excel directly into spark.
    """
    return shuttles


def preprocess_shuttles(shuttles: SparkDataFrame) -> SparkDataFrame:
    """Preprocesses the data for shuttles.

    Args:
        shuttles: Raw data.
    Returns:
        Preprocessed data, with `price` converted to a float and `d_check_complete`,
        `moon_clearance_complete` converted to boolean.
    """
    shuttles = shuttles.withColumn(
        "d_check_complete", _is_true(shuttles.d_check_complete))
    shuttles = shuttles.withColumn(
        "moon_clearance_complete", _is_true(shuttles.moon_clearance_complete))
    shuttles = shuttles.withColumn("price", _parse_money(shuttles.price))

    # Drop columns that aren't used for model training
    shuttles = shuttles.drop(
        'shuttle_location', 'engine_type', 'engine_vendor', 'cancellation_policy')
    return shuttles


def preprocess_reviews(reviews: SparkDataFrame) -> SparkDataFrame:
    # Drop columns that aren't used for model training
    reviews = reviews.drop('review_scores_comfort', 'review_scores_amenities', 'review_scores_trip', 'review_scores_crew',
                           'review_scores_location', 'review_scores_price', 'number_of_reviews', 'reviews_per_month')
    return reviews


def create_model_input_table(
    shuttles: SparkDataFrame, companies: SparkDataFrame, reviews: SparkDataFrame
) -> SparkDataFrame:
    """Combines all data to create a model input table.

    Args:
        shuttles: Preprocessed data for shuttles.
        companies: Preprocessed data for companies.
        reviews: Raw data for reviews.
    Returns:
        Model input table.

    """
    # Rename columns to prevent duplicates
    shuttles = shuttles.withColumnRenamed("id", "shuttle_id")
    companies = companies.withColumnRenamed("id", "company_id")

    rated_shuttles = shuttles.join(reviews, "shuttle_id", how="left")
    model_input_table = rated_shuttles.join(
        companies, "company_id", how="left")
    model_input_table = model_input_table.dropna()
    return model_input_table


def load_data(initial_data_train_labels: pd.DataFrame, initial_data_train_features: pd.DataFrame) -> pd.DataFrame:
    """
    Load the initial data train labels and features.

    Args:
        initial_data_train_labels (pd.DataFrame): The initial data train labels.
        initial_data_train_features (pd.DataFrame): The initial data train features.

    Returns:
        pd.DataFrame: The loaded data train labels and features.
    """

    return initial_data_train_labels, initial_data_train_features


def split_cities(initial_data_train_labels: pd.DataFrame, initial_data_train_features: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Seperate data for San Juan
    sj_train_features = initial_data_train_features.loc['sj']
    sj_train_labels = initial_data_train_labels.loc['sj']

    # Separate data for Iquitos
    iq_train_features = initial_data_train_features.loc['iq']
    iq_train_labels = initial_data_train_labels.loc['iq']

    return sj_train_features, sj_train_labels, iq_train_features, iq_train_labels


def impute_features(sj_train_features: pd.DataFrame, iq_train_features: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Fill missing values with the mean of the column
    sj_train_features = sj_train_features.fillna(method='ffill')
    iq_train_features = iq_train_features.fillna(method='ffill')

    return sj_train_features, iq_train_features


def select_features(sj_train_features: pd.DataFrame, iq_train_features: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # select features we want
    features = ['reanalysis_specific_humidity_g_per_kg',
                'reanalysis_dew_point_temp_k',
                'station_avg_temp_c',
                'station_min_temp_c']

    sj_train_features = sj_train_features[features]
    iq_train_features = iq_train_features[features]

    return sj_train_features, iq_train_features


def join_features_labels(sj_train_features: pd.DataFrame, sj_train_labels: pd.DataFrame, iq_train_features: pd.DataFrame, iq_train_labels: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Join the features and labels
    sj_train = sj_train_features.join(sj_train_labels)
    iq_train = iq_train_features.join(iq_train_labels)

    return sj_train, iq_train
