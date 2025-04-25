'''
This file contains the training and prediction process of the neural network models for each market.

Similar to the ML_Market.py.

Functions:
- `predict_test`: Predicts the test data by taking the average of the predictions from the "best" models trained from multiple random seeds.
- `training_year`: Trains multiple models with different random seeds for a certain year.
- `training_years`: Applies the `training_year` function for a list of years.
- `training_whole`: Executes the whole training process for a specific market.
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
from Load_Data import load_data, in_output, split_data, load_markets

# your local path to store the data and results
your_path = '/data01/AI_Finance'

# GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


# hyper_parameter for Neural Network
epoch_num = 100 # maximum number of epoches
lambda1_l = [1e-5, 1e-4, 1e-3] # l1 penalty
learning_rate = 1e-2 # learning rate
num_nn = 10  # number of neural networks for ensemble
patience = 5 # patience for early stopping


# predict for test data by taking the average of the predictions from the "best" models trained from multiple random seeds, 
# where for each seed, the "best" model is selected from different hyperparameters based on the validation data
def predict_test(month_data, best_model_l):
    test_x, _ = in_output(month_data)
    month_data.reset_index(drop=True, inplace=True)
    y_pred_l = [best_model(torch.from_numpy(test_x.values).float().to(device)) for best_model in best_model_l]
    y_pred = torch.cat(y_pred_l, dim=1).mean(dim=1)
    y_pred = y_pred.cpu().detach().numpy()
    # concat the prediction with the original target for later comparison
    y_pred = pd.concat([month_data[['PERMNO', 'DATE', 'TARGET']], pd.DataFrame(y_pred, columns=['pred'])], axis=1)
    return y_pred


# For a certain year, train num_nn models.
def training_year(add_year, market, model_name, final_data, train_split, valid_split, batch_size):
    parameters_path = join(your_path, 'model_parameters')
    os.makedirs(parameters_path, exist_ok=True)
    basedir = join(parameters_path, market, model_name)
    os.makedirs(basedir, exist_ok=True)

    train_data, valid_data, test_data = split_data(final_data, train_split, valid_split, add_year)
    train_x, train_y = in_output(train_data)
    valid_x, valid_y = in_output(valid_data)

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
        for lambda1 in lambda1_l:
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
            while epoch < epoch_num and j < patience:
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
                torch.save(best_model.state_dict(), os.path.join(basedir, f'year{valid_split+add_year}_seed{seed}'))
            else:
                pass
        best_model_l.append(best_model)

    # Due to the properties of Batch Normlization, to avoid messing up the chronological order in test data,
    # so predict test_x sequentially, i.e. month by month
    # compute the forecast for each month in training, validation, testing
    train_data['DATE'] = pd.to_datetime(train_data['DATE'], format='%Y-%m-%d')
    train_data['date_bk'] = train_data['DATE']
    train_months_l = [g for n, g in train_data.set_index('date_bk').groupby(pd.Grouper(key='DATE', freq='M')) if len(g) >= 10]
    train_y_pred_l = [predict_test(month_data, best_model_l) for month_data in train_months_l]
    y_pred_train = pd.concat(train_y_pred_l)

    valid_data['DATE'] = pd.to_datetime(valid_data['DATE'], format='%Y-%m-%d')
    valid_data['date_bk'] = valid_data['DATE']
    valid_months_l = [g for n, g in valid_data.set_index('date_bk').groupby(pd.Grouper(key='DATE', freq='M')) if len(g) >= 10]
    valid_y_pred_l = [predict_test(month_data, best_model_l) for month_data in valid_months_l]
    y_pred_valid = pd.concat(valid_y_pred_l)

    test_data['DATE'] = pd.to_datetime(test_data['DATE'], format='%Y-%m-%d')
    test_data['date_bk'] = test_data['DATE']
    test_months_l = [g for n, g in test_data.set_index('date_bk').groupby(pd.Grouper(key='DATE', freq='M')) if len(g) >= 10]
    test_y_pred_l = [predict_test(month_data, best_model_l) for month_data in test_months_l]
    y_predict_test = pd.concat(test_y_pred_l)

    # save the forecasts
    forecasts_path = join(your_path, 'forecasts')
    os.makedirs(forecasts_path, exist_ok=True)
    save_path = join(forecasts_path, market, model_name, f'Year{valid_split + add_year}')
    os.makedirs(save_path, exist_ok=True)
    y_pred_train.to_csv(join(save_path, 'train_pred.csv'))
    y_pred_valid.to_csv(join(save_path, 'valid_pred.csv'))
    y_predict_test.to_csv(join(save_path, 'test_pred.csv'))

    return y_predict_test


# for a list of years, apply training_year function
def training_years(add_years_list, market, model_name, final_data, train_split, valid_split, batch_size):
    df_l = []
    for add_year in add_years_list:
        tmp_y_pred = training_year(add_year, market, model_name, final_data, train_split, valid_split, batch_size)
        df_l.append(tmp_y_pred)
    return pd.concat(df_l)


# The Whole Process
def training_whole(market):
    print('# ' * 20 + market + '*' +  + ' #' * 20)

    # split data
    final_data, start_year, train_split, valid_split, end_year, batch_size = load_data(your_path, market)

    # test target
    test_y = final_data[str(valid_split + 1) < final_data['DATE']][['PERMNO', 'DATE', 'TARGET']]
    test_y.reset_index(drop=True, inplace=True)
    print("Shape of Prediction Data: %d" % test_y.shape[0])

    # NNs with various layers
    model_name_l = ['nn%d' % i for i in range(1, 6)]

    for model_name in model_name_l:
        print('* ' * 10 + model_name.upper() + ' *' * 10)
        pred_df = training_years(range(1, end_year + 1 - valid_split), market, model_name, final_data, train_split, valid_split, batch_size)
        pred_df['DATE'] = pred_df['DATE'].astype('str')
        forecast = pd.merge(test_y, pred_df, on=['PERMNO', 'DATE'])
        # save data
        forecast.to_csv(join(your_path, 'forecasts', market, 'f{model_name}_pred.csv'))


if __name__ == '__main__':
    markets_l = load_markets(your_path)
    # train market-specific models for all markets, including the USA
    for market in markets_l:
        training_whole(market)
