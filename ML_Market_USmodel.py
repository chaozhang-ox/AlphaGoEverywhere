'''
Predict the international markets using the USA model.

The script consists of the following functions:
- `predict_market_year_USmodel(cur_year, model_name, final_data)`: Predicts the market for a certain year using the USA estimated model without fine-tuning.
- `predict_market_USmodel(model_name, final_data, split_way, local_dir)`: Predicts the market for a list of years using the USA model. The period aligns with the raw testing period of the specific international markets.
- `predict_whole(market)`: Predicts the given international market using the USA model.

To use this script, set the `your_path` variable to the local path where you store the data and results. Then, run the script. You can change the new 'forecasts' directory in the `predict_whole` function.
'''

import pandas as pd
import os
from joblib import load
from os.path import join

from Load_Data import load_data, in_output, load_markets

# your local path to store the data and results
your_path = '/data01/AI_Finance/'

# variables used in ols-3 model, see GKX's paper
selected_columns = ['mvel1', 'bm', 'mom_12']


# predict the market for a certain year using the USA estimated model, no fine-tuning
def predict_market_year_USmodel(cur_year, model_name, final_data):
    test_data = final_data[final_data['DATE'] < str(cur_year + 1)]
    test_data = test_data[test_data['DATE'] >= str(cur_year)]
    test_data.reset_index(drop=True, inplace=True)
    if len(test_data) > 0:
        test_x, _ = in_output(test_data)

        if 'ols-3' in model_name:
            test_x = test_x[selected_columns]

        basedir = join(your_path, 'model_parameters', 'USA', model_name)
        model_path = join(basedir, f'year{cur_year}.joblib')
        model = load(model_path)
        y_pred = model.predict(test_x)
        y_pred = pd.concat([test_data[['PERMNO', 'DATE']], pd.DataFrame(y_pred, columns=['pred'])], axis=1)
        return y_pred
    else:
        pass


# for a list of years
def predict_market_USmodel(model_name, final_data, split_way, local_dir):
    test_y = final_data[['PERMNO', 'DATE', 'TARGET']]
    df_l = []
    start_year, train_split, valid_split, end_year = split_way
    for cur_year in range(valid_split+1, end_year+1):
        tmp_y_pred = predict_market_year_USmodel(cur_year, model_name, final_data)
        df_l.append(tmp_y_pred)

    pred_df = pd.concat(df_l)
    forecast = pd.merge(test_y, pred_df, on=['PERMNO', 'DATE'])

    local_path = os.path.join(local_dir, f'{model_name}_pred.csv')
    forecast.to_csv(local_path)


def predict_whole(market):
    # load the specific market data
    final_data, start_year, train_split, valid_split, end_year, _ = load_data(market)
    split_way = (start_year, train_split, valid_split, end_year)

    # new directory to store the forecasts, as we are using the USA model
    local_dir = join(your_path, 'forecasts_USmodel', market)
    os.makedirs(local_dir, exist_ok=True)

    model_name_l = ['ols-3', 'linear', 'lasso', 'ridge', 'rf', 'gbrt+h']

    for model_name in model_name_l:
        print('* ' * 7 + model_name.upper() + ' *' * 7)
        predict_market_USmodel(model_name, final_data, split_way, local_dir)


if __name__ == '__main__':
    markets_l = load_markets(your_path)
    # use the USA model to predict the international markets, so we exclude the US from the markets list
    international_markets = [i for i in markets_l if i != 'USA']
    for market in international_markets:
        predict_whole(market)