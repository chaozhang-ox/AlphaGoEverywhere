'''
Train market-specific models enhanced by the USA factors, USA characteristics gaps, and local factors, as additional features
Similar to the NN_Market.py, but with more predictors
'''

import pandas as pd
import numpy as np
import os
import time
import torch
import torch.nn as nn
import random
from os.path import join

from NN_set import *
from Load_Data import variables, in_output, split_data, load_markets


# your local path to store the data and results
your_path = '/data01/AI_Finance/'

# additional feature types
USnameF = 'USVWF'  # US Value Weighted Factor
USnameG = 'USGapQ' # US Gap
LnameF = 'LVWF'    # Local Value Weighted Factor

# additional feature names
influent_features = [i for i in variables if i not in ['PERMNO', 'DATE', 'TARGET']]
variables += [i + '-' + USnameF for i in influent_features]
variables += [i + '-' + USnameG for i in influent_features]
variables += [i + '-' + LnameF for i in influent_features]


# GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

model_name_l = ['nn%d' % i for i in range(1, 6)]

# hyper_parameter for Neural Network
epoch_num = 100
lambda1_l = [1e-5, 1e-2]
lr_l = [1e-3, 1e-2]
num_nn = 10
p = 5
amp = 10


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

def pred_test(month_data, best_model_l, year, market):
    test_x, _ = in_output(month_data, year, market)
    month_data.reset_index(drop=True, inplace=True)
    # predict for test data
    y_pred_l = [best_model(torch.from_numpy(test_x.values).float().to(device)) for best_model in best_model_l]
    y_pred = torch.cat(y_pred_l, dim=1).mean(dim=1)
    y_pred = y_pred.cpu().detach().numpy()
    y_pred = pd.concat([month_data[['PERMNO', 'DATE', 'TARGET']], pd.DataFrame(y_pred, columns=['pred'])], axis=1)
    return y_pred


# For a certain year, train models and select the best one hyperparameter used for forecasting
def model_year(add_year, market, model_name, final_data, train_split, valid_split, batch_size):
    parameters_path = join(your_path, 'model_parameters_enhanced')
    os.makedirs(parameters_path, exist_ok=True)
    basedir = join(parameters_path, market, model_name)
    os.makedirs(basedir, exist_ok=True)

    cur_year = valid_split + add_year
    train_data, valid_data, test_data = split_data(final_data, train_split, valid_split, add_year)
    train_x, train_y = in_output(train_data, cur_year, 'USA')
    valid_x, valid_y = in_output(valid_data, cur_year, 'USA')
    print(train_x)

    h = int(train_x.shape[0] / batch_size + 1)

    # Train Model
    best_model_l = []
    # Convert numpy arrays to torch tensors
    inputs = torch.from_numpy(train_x.values).float()
    targets = torch.from_numpy(train_y.values).float()
    targets = targets.view(len(targets), -1)
    valid_x = torch.from_numpy(valid_x.values).float().to(device)
    valid_y = torch.from_numpy(valid_y.values).float()
    valid_y = valid_y.view(len(valid_y), -1).to(device)

    for seed in range(num_nn):
        torch.manual_seed(seed)
        print("--- Model %d Starts" % seed)

        best_valid_error = np.inf
        for learning_rate in lr_l:
            for lambda1 in lambda1_l:
                # GPU Parallel
                tmp_model = eval(model_name.upper())(inputs.shape[1])
                tmp_model.to(device)

                tmp_model.apply(weights_init)

                # loss func and optimizer
                optimizer = torch.optim.Adam(tmp_model.parameters(), lr=learning_rate)
                criterion = torch.nn.MSELoss()
                # early stopping and maximum number of epoches
                epoch = 0
                j = 0
                tmp_valid_error = np.inf
                while epoch < epoch_num and j < p:
                    permutation = torch.randperm(len(inputs))
                    for _ in range(h):
                        i = random.randint(0, len(inputs) - batch_size)
                        indices = permutation[i:i + batch_size]
                        batch_x, batch_y = inputs[indices].to(device), targets[indices].to(device)
                        # l1 penalty
                        l1_regularization = torch.tensor(0.0).to(device)
                        for param in tmp_model.parameters():
                            l1_regularization += torch.norm(param, 1)
                        # Forward pass
                        outputs = tmp_model(batch_x)
                        loss = criterion(outputs, batch_y) + lambda1 * l1_regularization
                        loss = loss.to(device)
                        # Backward and optimize
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                    # calculate loss function value on validation dataset for each epoch
                    epoch += 1

                    pred_valid = tmp_model(valid_x)
                    loss = criterion(pred_valid, valid_y)
                    valid_error_new = loss.cpu().detach().numpy()
                    if valid_error_new < tmp_valid_error:
                        j = 0
                        tmp_valid_error = valid_error_new
                        tmp_best_model = tmp_model
                    else:
                        j += 1
                # pick up the best hyperparameter set
                if tmp_valid_error < best_valid_error:
                    best_valid_error = tmp_valid_error
                    best_model = tmp_best_model
                    torch.save(best_model.state_dict(), os.path.join(basedir, "year%d_seed%d" % (cur_year, seed)))
                else:
                    pass
        best_model_l.append(best_model)

    print("Year %d finished!" % cur_year)

    # Due to the properties of BN, to avoid messing up the chronological order in test data,
    # so predict test_x sequentially, i.e. month by month
    train_data['DATE'] = pd.to_datetime(train_data['DATE'], format='%Y-%m-%d')
    train_data['date_bk'] = train_data['DATE']
    train_months_l = [g for n, g in train_data.set_index('date_bk').groupby(pd.Grouper(key='DATE', freq='M')) if len(g) >= 10]
    train_y_pred_l = [pred_test(month_data, best_model_l, cur_year, 'USA') for month_data in train_months_l]
    y_pred_train = pd.concat(train_y_pred_l)

    valid_data['DATE'] = pd.to_datetime(valid_data['DATE'], format='%Y-%m-%d')
    valid_data['date_bk'] = valid_data['DATE']
    valid_months_l = [g for n, g in valid_data.set_index('date_bk').groupby(pd.Grouper(key='DATE', freq='M')) if len(g) >= 10]
    valid_y_pred_l = [pred_test(month_data, best_model_l, cur_year, 'USA') for month_data in valid_months_l]
    y_pred_valid = pd.concat(valid_y_pred_l)

    test_data['DATE'] = pd.to_datetime(test_data['DATE'], format='%Y-%m-%d')
    test_data['date_bk'] = test_data['DATE']
    test_months_l = [g for n, g in test_data.set_index('date_bk').groupby(pd.Grouper(key='DATE', freq='M')) if len(g) >= 10]
    test_y_pred_l = [pred_test(month_data, best_model_l, cur_year, 'USA') for month_data in test_months_l]
    y_pred_test = pd.concat(test_y_pred_l)

    forecasts_path = join(your_path, 'forecasts_enhanced')
    os.makedirs(forecasts_path, exist_ok=True)
    save_path = join(forecasts_path, market, model_name, "Year%d" % (valid_split + add_year))
    os.makedirs(save_path, exist_ok=True)
    y_pred_train.to_csv(join(save_path, 'train_pred.csv'))
    y_pred_valid.to_csv(join(save_path, 'valid_pred.csv'))
    y_pred_test.to_csv(join(save_path, 'test_pred.csv'))

    return y_pred_test


# for a list of years, apply model_year function
def model_years(add_years_list, market, model_name, final_data, train_split, valid_split, batch_size):
    df_l = []
    for add_year in add_years_list:
        tmp_y_pred = model_year(add_year, market, model_name, final_data, train_split, valid_split, batch_size)
        df_l.append(tmp_y_pred)
    return pd.concat(df_l)


# The Whole Process
def training_whole(market, filename):
    print('# ' * 20 + market + '*' + filename + ' #' * 20)

    # split data
    final_data, start_year, train_split, valid_split, end_year, batch_size = load_data_enhanced(market)

    # actual and predicted test target
    test_y = final_data[str(valid_split + 1) < final_data['DATE']][['PERMNO', 'DATE', 'TARGET']]
    test_y.reset_index(drop=True, inplace=True)
    print("Shape of Prediction Data: %d" % test_y.shape[0])

    # attempts on various models
    for model_name in model_name_l:
        print('* ' * 10 + model_name.upper() + ' *' * 10)
        pred_df = model_years(range(1, end_year + 1 - valid_split), market, model_name, final_data, train_split, valid_split, batch_size)
        pred_df['DATE'] = pred_df['DATE'].astype('str')
        forecast = pd.merge(test_y, pred_df, on=['PERMNO', 'DATE'])
        forecast.to_csv(join(your_path, 'forecasts_enhanced', market, 'f{model_name}_pred.csv'))


if __name__ == '__main__':
    markets_l = load_markets(your_path)
    for market in markets_l:
        training_whole(market)