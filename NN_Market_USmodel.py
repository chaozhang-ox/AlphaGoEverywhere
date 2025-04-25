'''
Predict the international markets using the USA model.

similar description of file structure as in the ML_Market_USmodel.py
'''

import pandas as pd
import os
from os.path import join
import torch
import torch.nn as nn

from NN_set import *
from Load_Data import load_data, in_output, load_markets

# your local path to store the data and results
your_path = '/data01/AI_Finance/'

# GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# number of neural networks for ensemble
num_nn = 10 

# same reason as in NN_Market.py
# due to the presence of Batch Normalization, we predict test_x sequentially, i.e. month by month to avoid messing up the chronological order
def predict_market_month_USmodel(month_data, best_model_l):
    test_x, _ = in_output(month_data)
    month_data.reset_index(drop=True, inplace=True)
    y_pred_l = [best_model(torch.from_numpy(test_x.values).float().to(device)) for best_model in best_model_l]
    y_pred = torch.cat(y_pred_l, dim=1).mean(dim=1)
    y_pred = y_pred.cpu().detach().numpy()
    y_pred = pd.concat([month_data[['PERMNO', 'DATE']], pd.DataFrame(y_pred, columns=['pred'])], axis=1)
    return y_pred


# predict for a certain year using the USA estimated model, no fine-tuning
def predict_market_year_USmodel(year, model_name, final_data):
    test_data = final_data[final_data['DATE'] < str(year + 1)]
    test_data = test_data[test_data['DATE'] >= str(year)]
    test_data.reset_index(drop=True, inplace=True)
    test_data['DATE'] = pd.to_datetime(test_data['DATE'], format='%Y-%m-%d')
    test_data['date_bk'] = test_data['DATE']
    months_l = [g for n, g in test_data.set_index('date_bk').groupby(pd.Grouper(key='DATE', freq='M')) if len(g) >= 10]
    basedir = join(your_path, 'model_parameters', 'USA', model_name)
    best_model_l = []
    for seed in range(num_nn):
        model = torch.load(join(basedir, f'year{year}_seed{seed}.pt')).cuda()
        best_model_l.append(model)

    if len(months_l) > 0:
        y_pred_l = [predict_market_month_USmodel(month_data, best_model_l) for month_data in months_l]
        y_pred = pd.concat(y_pred_l)
        return y_pred
    else:
        return None


# for a list of years
def predict_market_USmodel(model_name, final_data, split_way, local_dir):
    test_y = final_data[['PERMNO', 'DATE', 'TARGET']]
    df_l = []
    start_year, train_split, valid_split, end_year = split_way
    for year in range(valid_split+1, end_year+1):
        tmp_y_pred = predict_market_year_USmodel(year, model_name, final_data)
        df_l.append(tmp_y_pred)
    pred_df = pd.concat(df_l)
    pred_df['DATE'] = pred_df['DATE'].astype('str')
    forecast = pd.merge(test_y, pred_df, on=['PERMNO', 'DATE'])

    local_path = os.path.join(local_dir, f'{model_name}_pred.csv')
    forecast.to_csv(local_path)


def predict_whole(market):
    final_data, start_year, train_split, valid_split, end_year, batch_size = load_data(market)
    split_way = (start_year, train_split, valid_split, end_year)

    # new directory to store the forecasts, as we are using the USA model
    local_dir = join(your_path, 'forecasts_USmodel', market)
    os.makedirs(local_dir, exist_ok=True)

    model_name_l = ['nn%d' % i for i in range(1, 6)]

    for model_name in model_name_l:
        print('* ' * 7 + model_name.upper() + ' *' * 7)
        predict_market_USmodel(model_name, final_data, split_way, local_dir)


if __name__ == "__main__":
    markets_l = load_markets(your_path)
    # use the USA model to predict the international markets, so we exclude the US from the markets list
    international_markets = [i for i in markets_l if i != 'USA']
    for market in international_markets:
        predict_whole(market)