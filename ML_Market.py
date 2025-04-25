'''
This file contains the training and prediction process of various ML models for each market.

The main functions in this file are:
- `training_year`: Trains models and selects the best hyperparameter for forecasting for a certain year.
- `training_years`: Applies the `training_year` function for a list of years.
- `training_parallelize`: Parallelizes the training process using multiprocessing.
- `training_whole`: Executes the whole training process for a specific market.

Note: This code assumes the existence of the following modules: pandas, numpy, sklearn, multiprocessing, functools, os, time, joblib, etc.

'''

import pandas as pd
import numpy as np
from sklearn.linear_model import HuberRegressor, LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from multiprocessing import cpu_count, Pool
from functools import partial
import os
import time
from joblib import dump
from sklearn.metrics import mean_squared_error
from os.path import join

from Load_Data import load_data, in_output, split_data, load_markets

# your local path to store the data and results
your_path = '/data01/AI_Finance/'

# variables used in ols-3 model, see GKX's paper
selected_columns = ['mvel1', 'bm', 'mom_12']

# For a certain year, train models and select the best one hyperparameter used for forecasting
def training_year(add_year, market, model_name, final_data, train_split, valid_split):
    parameters_path = join(your_path, 'model_parameters')
    os.makedirs(parameters_path, exist_ok=True)
    basedir = join(parameters_path, market, model_name)
    os.makedirs(basedir, exist_ok=True)

    cur_year = valid_split + add_year
    model_path = join(basedir, f'year{cur_year}.joblib')

    train_data, valid_data, test_data = split_data(final_data, train_split, valid_split, add_year)
    train_x, train_y = in_output(train_data)
    valid_x, valid_y = in_output(valid_data)
    test_x, test_y = in_output(test_data)

    # ols with 3 predictors
    if model_name == 'ols-3':
        train_x = train_x[selected_columns]
        valid_x = valid_x[selected_columns]
        test_x = test_x[selected_columns]
        best_model = LinearRegression()
        best_model.fit(train_x, train_y)
    
    # ols with 3 predictors and huber regression
    elif model_name == 'ols-3+h':
        train_x = train_x[selected_columns]
        valid_x = valid_x[selected_columns]
        test_x = test_x[selected_columns]
        min_mse = np.inf
        for epsilon in [3.09]:
            tmp_model = HuberRegressor(epsilon=epsilon)
            tmp_model.fit(train_x, train_y)
            # Determining the parameter of model using the validation sets
            pred_valid = tmp_model.predict(valid_x)
            tmp_mse = mean_squared_error(valid_y, pred_valid)
            if tmp_mse < min_mse:
                min_mse = tmp_mse
                best_model = tmp_model

    # ols with all predictors
    elif model_name == 'linear':
        best_model = LinearRegression()
        best_model.fit(train_x, train_y)

    # ols with all predictors and huber regression
    elif model_name == 'linear+h':
        # huber regression
        min_mse = np.inf
        for epsilon in np.linspace(2, 4, 5):
            tmp_model = HuberRegressor(epsilon=epsilon)
            tmp_model.fit(train_x, train_y)
            # Determining the parameter of model using the validation sets
            pred_valid = tmp_model.predict(valid_x)
            tmp_mse = mean_squared_error(valid_y, pred_valid)
            if tmp_mse < min_mse:
                min_mse = tmp_mse
                best_model = tmp_model

    # lasso with all predictors
    elif model_name == 'lasso':
        min_mse = np.inf
        for alpha in np.logspace(-3, 3, 10):
            tmp_model = Lasso(alpha)
            tmp_model.fit(train_x, train_y)
            pred_valid = tmp_model.predict(valid_x)
            tmp_mse = mean_squared_error(valid_y, pred_valid)
            if tmp_mse < min_mse:
                min_mse = tmp_mse
                best_model = tmp_model
    
    # ridge with all predictors
    elif model_name == 'ridge':
        min_mse = np.inf
        for alpha in np.logspace(-3, 3, 10):
            tmp_model = Ridge(alpha)
            tmp_model.fit(train_x, train_y)
            pred_valid = tmp_model.predict(valid_x)
            tmp_mse = mean_squared_error(valid_y, pred_valid)
            if tmp_mse < min_mse:
                min_mse = tmp_mse
                best_model = tmp_model
    
    # elastic net with all predictors
    elif model_name == 'enet':
        min_mse = np.inf
        for alpha in np.logspace(-4, -1, 4):
            tmp_model = ElasticNet(alpha, l1_ratio=0.5)
            tmp_model.fit(train_x, train_y)
            pred_valid = tmp_model.predict(valid_x)
            tmp_mse = mean_squared_error(valid_y, pred_valid)
            if tmp_mse < min_mse:
                min_mse = tmp_mse
                best_model = tmp_model

    # random forest with all predictors
    elif model_name == 'rf':
        min_mse = np.inf
        for maxdp in [2, 4, 6]:
            for max_ft in [3, 5, 10]:
                tmp_model = RandomForestRegressor(max_depth=maxdp,
                                                  n_estimators=300,
                                                  max_features=max_ft)
                tmp_model.fit(train_x, train_y)
                pred_valid = tmp_model.predict(valid_x)
                tmp_mse = mean_squared_error(valid_y, pred_valid)
                if tmp_mse < min_mse:
                    min_mse = tmp_mse
                    best_model = tmp_model
    
    # gradient boosting with all predictors
    elif model_name == 'gbrt+h':
        min_mse = np.inf
        for maxdp in [1, 2]:
            for lr in [0.01, 0.1]:
                for n_estimators in [1000]:
                    tmp_model = GradientBoostingRegressor(max_depth=maxdp,
                                                          learning_rate=lr,
                                                          n_estimators=n_estimators,
                                                          loss='huber',
                                                          alpha=0.999)
                    tmp_model.fit(train_x, train_y)
                    pred_valid = tmp_model.predict(valid_x)
                    tmp_mse = mean_squared_error(valid_y, pred_valid)
                    if tmp_mse < min_mse:
                        min_mse = tmp_mse
                        best_model = tmp_model
    else:
        best_model = LinearRegression()
        best_model.fit(train_x, train_y)

    dump(best_model, model_path)

    # predict
    y_pred_train = best_model.predict(train_x)
    y_pred_train = pd.concat([train_data[['PERMNO', 'DATE', 'TARGET']], pd.DataFrame(y_pred_train, columns=['pred'])], axis=1)

    y_pred_valid = best_model.predict(valid_x)
    y_pred_valid = pd.concat([valid_data[['PERMNO', 'DATE', 'TARGET']], pd.DataFrame(y_pred_valid, columns=['pred'])], axis=1)

    y_pred_test = best_model.predict(test_x)
    y_pred_test = pd.concat([test_data[['PERMNO', 'DATE', 'TARGET']], pd.DataFrame(y_pred_test, columns=['pred'])], axis=1)

    forecasts_path = join(your_path, 'forecasts')
    os.makedirs(forecasts_path, exist_ok=True)
    save_path = join(forecasts_path, market, model_name, f'Year{valid_split + add_year}')
    os.makedirs(save_path, exist_ok=True)
    y_pred_train.to_csv(join(save_path, 'train_pred.csv'))
    y_pred_valid.to_csv(join(save_path, 'valid_pred.csv'))
    y_pred_test.to_csv(join(save_path, 'test_pred.csv'))
    return y_pred_test


# for a list of years, apply training_year function
def training_years(add_years_list, market, model_name, final_data, train_split, valid_split):
    df_l = []
    for add_year in add_years_list:
        tmp_y_pred = training_year(add_year, market, model_name, final_data, train_split, valid_split)
        df_l.append(tmp_y_pred)
    return pd.concat(df_l)


# parallel process if needed
def training_parallelize(market, model_name, final_data, train_split, valid_split, end_year):
    cores = cpu_count()  # Number of CPU cores on your system
    partitions = min(cores, end_year-valid_split) - 1  # Define as many partitions as you want
    data_split = np.array_split(range(1, end_year + 1 - valid_split), partitions)
    pool = Pool(partitions)
    partial_func = partial(training_years,
                           market=market,
                           model_name=model_name,
                           final_data=final_data,
                           train_split=train_split,
                           valid_split=valid_split)
    pool_results = pd.concat(pool.map(partial_func, data_split))
    pool.close()
    pool.join()
    return pool_results


# The Whole Process
def training_whole(market):
    print('# ' * 20 + market + ' #' * 20)

    # split data
    final_data, start_year, train_split, valid_split, end_year, _ = load_data(market)

    # test target
    test_y = final_data[str(valid_split + 1) < final_data['DATE']][['PERMNO', 'DATE', 'TARGET']]
    test_y.reset_index(drop=True, inplace=True)
    print("Shape of Predition Data: %d" % test_y.shape[0])

    # attempts on various models
    model_name_l = ['ols-3', 'linear', 'lasso', 'ridge', 'rf', 'gbrt+h']
    for model_name in model_name_l:
        print('* ' * 10 + model_name.upper() + ' *' * 10)
        # if needed, parallelize the process
        # pred_df = training_parallelize(market, model_name, final_data, train_split, valid_split, end_year)
        # otherwise, sequentially train the models
        pred_df = training_years(range(1, end_year + 1 - valid_split), market, model_name, final_data, train_split, valid_split)
        forecast = pd.merge(test_y, pred_df, on=['PERMNO', 'DATE'])

        # save and upload data
        forecast.to_csv(join(your_path, 'forecasts', market, 'f{model_name}_pred.csv'))


if __name__ == '__main__':
    markets_l = load_markets(your_path)
    # train market-specific models for all markets, including the USA
    for market in markets_l:
        training_whole(market)
