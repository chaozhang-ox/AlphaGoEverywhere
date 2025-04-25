'''
Train market-specific models enhanced by the USA factors, USA characteristics gaps, and local factors, as additional features
Similar to the ML_Market.py, but with more predictors
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

from Load_Data import variables, in_output, split_data, load_markets


# your local path to store the data and results
your_path = '/data01/AI_Finance/'

# variables used in ols-3 model, see GKX's paper
ols_3_columns = ['mvel1', 'bm', 'mom_12']

# additional feature types
USnameF = 'USVWF'  # US Value Weighted Factor
USnameG = 'USGapQ' # US Gap
LnameF = 'LVWF'    # Local Value Weighted Factor

# additional feature names
influent_features = [i for i in variables if i not in ['PERMNO', 'DATE', 'TARGET']]
variables += [i + '-' + USnameF for i in influent_features]
variables += [i + '-' + USnameG for i in influent_features]
variables += [i + '-' + LnameF for i in influent_features]


# maybe not all characteristics are useful, so select the most important features based on the market-specific model's performance
def load_variables_importance(year, ext_market, num_selected=10):
    basedir = join(your_path, 'result_analysis')

    ssd_df_l = []

    linear_model_l = ['linear', 'lasso', 'ridge']
    tree_model_l = ['rf', 'gbrt+h']
    nn_model_l = ['nn%d' % i for i in range(1, 6)]

    # select_model_name_l = linear_model_l + tree_model_l + nn_model_l
    select_model_name_l = linear_model_l + nn_model_l

    for model_name in select_model_name_l:
        local_path = join(basedir, ext_market, model_name + '_ssd.csv')
        ssd_df = pd.read_csv(local_path, index_col=0)
        ssd_df_l.append(ssd_df.loc[year])

    ssd_df_year = pd.concat(ssd_df_l, axis=1).T
    ssd_df_year.index = select_model_name_l

    selected_df = ssd_df_year.loc[nn_model_l].mean().sort_values(ascending=False)
    vars_sort = selected_df.index.tolist()
    vars_sort = [i for i in vars_sort if '_ia' not in i]

    selected_features = vars_sort[:num_selected]
    print(selected_features)
    return selected_features


def load_data_enhanced(market):
    local_path = join(your_path, f'{market}_USLVWF+USLGapQ.csv')
    final_data = pd.read_csv(local_path)
    final_data = final_data[variables]
    print(final_data)
    sum_df = pd.read_csv(join(your_path, 'Data', 'Basic_Info.csv'), index_col=0)
    start_year = sum_df.loc[market, 'Start']
    end_year = sum_df.loc[market, 'End']
    train_split = sum_df.loc[market, 'Train']
    valid_split = sum_df.loc[market, 'Valid']
    batch_size = sum_df.loc[market, 'BatchSize']

    final_data = final_data.replace([np.inf, -np.inf], np.nan)
    final_data.dropna(inplace=True, how='any')
    final_data = final_data[final_data['DATE'] <= str(end_year + 1)]
    final_data = final_data[final_data['DATE'] > str(start_year)]
    final_data.reset_index(drop=True, inplace=True)
    input_rows, input_size = final_data.shape
    print("Input Rows: %d" % input_rows)
    return final_data, start_year, train_split, valid_split, end_year, batch_size


# For a certain year, train models and select the best one hyperparameter used for forecasting
def model_year(add_year, market, model_name, final_data, train_split, valid_split):
    parameters_path = join(your_path, 'model_parameters_enhanced')
    os.makedirs(parameters_path, exist_ok=True)
    basedir = join(parameters_path, market, model_name)
    os.makedirs(basedir, exist_ok=True)

    cur_year = valid_split + add_year
    model_path = os.path.join(basedir, 'year%d.joblib' % cur_year)

    train_data, valid_data, test_data = split_data(final_data, train_split, valid_split, add_year)
    train_x, train_y = in_output(train_data, cur_year, 'USA')
    valid_x, valid_y = in_output(valid_data, cur_year, 'USA')
    test_x, test_y = in_output(test_data, cur_year, 'USA')

    if model_name == 'ols-3':
        train_x = train_x[ols_3_columns]
        valid_x = valid_x[ols_3_columns]
        test_x = test_x[ols_3_columns]
        best_model = LinearRegression()
        best_model.fit(train_x, train_y)
    elif model_name == 'ols-3+h':
        train_x = train_x[ols_3_columns]
        valid_x = valid_x[ols_3_columns]
        test_x = test_x[ols_3_columns]
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
    elif model_name == 'linear':
        best_model = LinearRegression()
        best_model.fit(train_x, train_y)
    elif model_name == 'linear+h':
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
    elif model_name == 'lasso':
        min_mse = np.inf
        for alpha in np.logspace(-3, 3, 10):
            tmp_model = Lasso(alpha)
            tmp_model.fit(train_x, train_y)
            # Determining the parameter of model using the validation sets
            pred_valid = tmp_model.predict(valid_x)
            tmp_mse = mean_squared_error(valid_y, pred_valid)
            if tmp_mse < min_mse:
                min_mse = tmp_mse
                best_model = tmp_model
    elif model_name == 'ridge':
        min_mse = np.inf
        for alpha in np.logspace(-3, 3, 10):
            tmp_model = Ridge(alpha)
            tmp_model.fit(train_x, train_y)
            # Determining the parameter of model using the validation sets
            pred_valid = tmp_model.predict(valid_x)
            tmp_mse = mean_squared_error(valid_y, pred_valid)
            if tmp_mse < min_mse:
                min_mse = tmp_mse
                best_model = tmp_model
    elif model_name == 'enet':
        min_mse = np.inf
        for alpha in np.logspace(-4, -1, 4):
            tmp_model = ElasticNet(alpha, l1_ratio=0.5)
            tmp_model.fit(train_x, train_y)
            # Determining the parameter of model using the validation sets
            pred_valid = tmp_model.predict(valid_x)
            tmp_mse = mean_squared_error(valid_y, pred_valid)
            if tmp_mse < min_mse:
                min_mse = tmp_mse
                best_model = tmp_model
    elif model_name == 'rf':
        min_mse = np.inf
        for maxdp in [2, 4, 6]:
            for max_ft in [3, 5, 10, 20, 30, 40]:
                tmp_model = RandomForestRegressor(max_depth=maxdp,
                                                  n_estimators=300,
                                                  max_features=max_ft)
                tmp_model.fit(train_x, train_y)
                # Determining the parameter of model using the validation sets
                pred_valid = tmp_model.predict(valid_x)
                tmp_mse = mean_squared_error(valid_y, pred_valid)
                if tmp_mse < min_mse:
                    min_mse = tmp_mse
                    best_model = tmp_model
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
                    # Determining the parameter of model using the validation sets
                    pred_valid = tmp_model.predict(valid_x)
                    tmp_mse = mean_squared_error(valid_y, pred_valid)
                    if tmp_mse < min_mse:
                        min_mse = tmp_mse
                        best_model = tmp_model
    else:
        best_model = LinearRegression()
        best_model.fit(train_x, train_y)

    dump(best_model, model_path)
    print("Year %d finished!" % (valid_split + add_year))

    # predict for train and validation data
    y_pred_train = best_model.predict(train_x)
    y_pred_train = pd.concat(
        [train_data[['PERMNO', 'DATE', 'TARGET']], pd.DataFrame(y_pred_train, columns=['pred'])],
        axis=1)

    y_pred_valid = best_model.predict(valid_x)
    y_pred_valid = pd.concat(
        [valid_data[['PERMNO', 'DATE', 'TARGET']], pd.DataFrame(y_pred_valid, columns=['pred'])],
        axis=1)

    # predict for test data
    y_pred_test = best_model.predict(test_x)
    y_pred_test = pd.concat(
        [test_data[['PERMNO', 'DATE', 'TARGET']], pd.DataFrame(y_pred_test, columns=['pred'])],
        axis=1)

    forecasts_path = join(your_path, 'forecasts_enhanced')
    os.makedirs(forecasts_path, exist_ok=True)
    save_path = join(forecasts_path, market, model_name, f'Year{valid_split + add_year}')
    os.makedirs(save_path, exist_ok=True)
    y_pred_train.to_csv(join(save_path, 'train_pred.csv'))
    y_pred_valid.to_csv(join(save_path, 'valid_pred.csv'))
    y_pred_test.to_csv(join(save_path, 'test_pred.csv'))
    return y_pred_test


# for a list of years, apply model_year function
def model_years(add_years_list, market, model_name, final_data, train_split, valid_split):
    df_l = []
    for add_year in add_years_list:
        tmp_y_pred = model_year(add_year, market, model_name, final_data, train_split, valid_split)
        df_l.append(tmp_y_pred)
    return pd.concat(df_l)


# parallel process
def pred_parallelize(market, model_name, final_data, train_split, valid_split, end_year):
    cores = cpu_count()  # Number of CPU cores on your system
    partitions = min(cores, end_year - valid_split) - 1  # Define as many partitions as you want
    data_split = np.array_split(range(1, end_year + 1 - valid_split), partitions)
    pool = Pool(partitions)
    partial_func = partial(model_years,
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
    final_data, start_year, train_split, valid_split, end_year, _ = load_data_enhanced(market)
    print(final_data)

    # actual and predicted test target
    test_y = final_data[str(valid_split + 1) < final_data['DATE']][['PERMNO', 'DATE', 'TARGET']]
    test_y.reset_index(drop=True, inplace=True)
    print("Shape of Predition Data: %d" % test_y.shape[0])
    # attempts on various models
    model_name_l = ['ols-3', 'linear', 'lasso', 'ridge', 'rf', 'gbrt+h']
    for model_name in model_name_l:
        print('* ' * 10 + model_name.upper() + ' *' * 10)
        pred_df = model_years(range(1, end_year + 1 - valid_split), market, model_name, final_data, train_split, valid_split)
        forecast = pd.merge(test_y, pred_df, on=['PERMNO', 'DATE'])
        forecast.to_csv(join(your_path, 'forecasts_enhanced', market, 'f{model_name}_pred.csv'))


if __name__ == '__main__':
    markets_l = load_markets(your_path)
    for market in markets_l:
        training_whole(market)