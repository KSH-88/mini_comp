import logging
from typing import Dict, Tuple

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import max_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from statsmodels.tools import eval_measures
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import numpy as np


def split_data(data: pd.DataFrame) -> Tuple:
    """Splits data into features and targets training and test sets.

    Args:
        data: Data containing features and target.
        splitsize: 800 for sj and 400 for iq
    Returns:
        Split data.
    """
    #train = data.head(splitsize)
    #test = data.tail(data.shape[0] - splitsize)

    train, test = train_test_split(
        data, test_size = 0.2, shuffle = False, stratify = None
    )
    return train, test


def get_best_model(train, test):
    # Step 1: specify the form of the model
    model_formula = "total_cases ~ 1 + " \
                    "reanalysis_specific_humidity_g_per_kg + " \
                    "reanalysis_dew_point_temp_k + " \
                    "station_min_temp_c + " \
                    "station_avg_temp_c"
    
    grid = 10 ** np.arange(-8, -3, dtype=np.float64)
                    
    best_alpha = []
    best_score = 1000
        
    # Step 2: Find the best hyper parameter, alpha
    for alpha in grid:
        model = smf.glm(formula=model_formula,
                        data=train,
                        family=sm.families.NegativeBinomial(alpha=alpha))

        results = model.fit()
        predictions = results.predict(test).astype(int)
        score = eval_measures.meanabs(predictions, test.total_cases)

        if score < best_score:
            best_alpha = alpha
            best_score = score

    print('best alpha = ', best_alpha)
    print('best score = ', best_score)
            
    # Step 3: refit on entire dataset
    full_dataset = pd.concat([train, test])
    model = smf.glm(formula=model_formula,
                    data=full_dataset,
                    family=sm.families.NegativeBinomial(alpha=best_alpha))

    fitted_model = model.fit()
    return fitted_model


def plot_fitted_values(train, fitted_model):
    figs, axes = plt.subplots(nrows=1, ncols=1)

    # plot 
    train['fitted'] = fitted_model.fittedvalues
    train.fitted.plot(ax=axes, label="Predictions")
    train.total_cases.plot(ax=axes, label="Actual")

    plt.suptitle("Dengue Predicted Cases vs. Actual Cases")
    plt.legend()
    return figs


def write_output_file(sj_test, iq_test, sj_best_model, iq_best_model):
    sj_predictions = sj_best_model.predict(sj_test).astype(int)
    iq_predictions = iq_best_model.predict(iq_test).astype(int)
    submission = [sj_test, iq_test]
    submission = pd.concat(submission)
    submission.total_cases = np.concatenate([sj_predictions, iq_predictions])
    return submission


def wrapper_split_data(sj_data, iq_data):
    sj_train, sj_test = split_data(sj_data)
    iq_train, iq_test = split_data(iq_data)
    return sj_train, sj_test, iq_train, iq_test


def wrapper_get_best_model(sj_train, sj_test, iq_train, iq_test):
    sj_fitted_model = get_best_model(sj_train, sj_test)
    iq_fitted_model = get_best_model(iq_train, iq_test)
    return sj_fitted_model, iq_fitted_model


def wrapper_plot_fitted_values(sj_train, sj_fitted_model, iq_train, iq_fitted_model):
    sj_figs = plot_fitted_values(sj_train, sj_fitted_model)
    iq_figs = plot_fitted_values(iq_train, iq_fitted_model)
    return sj_figs, iq_figs

