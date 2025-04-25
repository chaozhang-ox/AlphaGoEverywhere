"""
This file contains the functions to load the data for the ML and NN models.
"""
import pandas as pd
import numpy as np
from os.path import join
from SetUp import *


# load the US macroeconomic data
# fill in the missing values with the previous values to avoid look-ahead bias
def load_US_macro(your_path):
    macro_df = pd.read_csv(join(your_path, 'Data', 'US_Macro.csv'), thousands=',')
    cors_g = [i for i in macro_df.columns if i not in ['DATE']]
    macro_df[cors_g] = macro_df[cors_g].astype(float)
    macro_df.fillna(method='ffill', inplace=True)
    return macro_df


# load the data with 36 independent variables
# split the data into training, validation, and testing sets based on the start and end years of the market
# return the data, start year, training split, validation split, end year, and batch size
def load_data(your_path, market):
    local_path = join(your_path, 'Data', f'{market}_norm.csv')
    final_data = pd.read_csv(local_path)
    
    final_data = final_data[variables]

    final_data['TARGET'] = final_data['TARGET'].astype(float) * 100 # return in percentage

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


# distinguish independent and dependent variables
def in_output(sample_data):
    sample_x = sample_data[sample_data.columns[~sample_data.columns.isin(['TARGET', 'PERMNO', 'DATE'])]]
    sample_y = sample_data['TARGET']
    return sample_x, sample_y


# split data into training, validation, and testing sets
# we increase the training set forward by one year each time
# we move the validation and testing sets forward by one year each time
def split_data(final_data, train_split, valid_split, add_year):
    train_data = final_data[final_data['DATE'] <= str(train_split+add_year)]
    train_data.reset_index(drop=True, inplace=True)

    tmp_data_valid = final_data[str(train_split+add_year) < final_data['DATE']]
    valid_data = tmp_data_valid[tmp_data_valid['DATE'] <= str(valid_split+add_year)]
    valid_data.reset_index(drop=True, inplace=True)

    tmp_data_test = final_data[str(valid_split+add_year) < final_data['DATE']]
    test_data = tmp_data_test[tmp_data_test['DATE'] <= str(valid_split+add_year+1)]
    test_data.reset_index(drop=True, inplace=True)
    return train_data, valid_data, test_data


# load markets, used for the individual market model
def load_markets(your_path):
    sum_df = pd.read_csv(join(your_path, 'Data', 'Basic_Info.csv'), index_col=0)
    markets_l = sum_df.index
    return markets_l


# load the raw size data, used for calculating the value-weighted portfolio
def load_rawsize(your_path, market):
    raw_size = pd.read_csv(join(your_path, 'Data', f'{market}_rawsize.csv'))
    raw_size = raw_size[['PERMNO', 'DATE', 'size']]
    raw_size['DATE'] = pd.to_datetime(raw_size['DATE'], format='%Y-%m-%d')
    return raw_size, raw_size.shape[0]


# load raw data
def load_raw_data(your_path, market):
    # load data
    if market == 'USA':
        raw_data = pd.read_sas(join(your_path, 'Raw_Data', "rpsdata_rfs_1960.sas7bdat"))
        raw_data.rename(columns=USA_columns_dic, inplace=True)
        raw_data['PERMNO'] = raw_data['PERMNO'].apply(lambda x: str(int(x)))
        raw_data['DATE'] = raw_data['DATE'].apply(lambda dt: dt.replace(day=1))
    elif market == 'China':
        raw_data = pd.read_csv(join(your_path, 'Raw_Data', "China_m.csv"))
        raw_data.rename(columns=China_columns_dic, inplace=True)
        raw_data['PERMNO'] = raw_data['PERMNO'].apply(lambda x: str(x).zfill(6))
        raw_data['DATE'] = pd.to_datetime(raw_data['DATE'], format='%Ym%m')
    else:
        raw_data = pd.read_csv(join(your_path, 'Raw_Data', "%s_m.csv" % market))
        raw_data.rename(columns=Others_columns_dic, inplace=True)
        raw_data['DATE'] = pd.to_datetime(raw_data['DATE'], format='%Ym%m')

    raw_data = raw_data[variables]
    raw_data = raw_data.replace([np.inf, -np.inf], np.nan)

    final_data = raw_data[variables]
    final_data['size'] = np.exp(final_data['mvel1'])

    sum_df = pd.read_csv(join(your_path, 'Data', 'Basic_Info.csv'), index_col=0)
    start_year = sum_df.loc[market, 'Start']
    end_year = sum_df.loc[market, 'End']

    final_data = final_data.replace([np.inf, -np.inf], np.nan)
    final_data.dropna(inplace=True, how='any')
    final_data = final_data[final_data['DATE'] <= str(end_year + 1)]
    final_data = final_data[final_data['DATE'] > str(start_year)]
    final_data.reset_index(drop=True, inplace=True)
    print(final_data.shape[0])
    return final_data, start_year, end_year
