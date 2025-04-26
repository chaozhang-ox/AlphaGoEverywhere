'''
Normalize characteristics
'''

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from functools import partial
import os
from os.path import join

from Load_Data import *
from SetUp import *


your_path = '/data01/AI_Finance'
feature_path = join(your_path, 'Data')


# fill NA values with median, or drop rows with NA values
def Handle_NA(df, method):
    if method == 'median':
        # group by month
        df['date_bk'] = df['DATE']
        months = df.groupby(pd.Grouper(key='date_bk', freq='M'))
        prep_data = months.apply(lambda x: x.fillna(x.median()))
        prep_data.sort_index(inplace=True)
        del prep_data['date_bk']
    else:
        prep_data = df.dropna(how='any')

    prep_data.reset_index(drop=True, inplace=True)
    return prep_data


# rank one column
def rank_column(sr):
    assert not sr.isnull().any()
    result = sr.rank()
    result -= result.mean()
    result /= ((len(result)-1) / 2)
    return result


# rank all columns
def rank_all(df):
    if df.shape[0] == 0:
        return df
    else:
        need_process = df.columns[~df.columns.isin(['PERMNO', 'TARGET', 'DATE'])]
        for clm in need_process:
            df[clm] = rank_column(df[clm])
        return df


# check if there are enough rows in each month
# We require at least 100 stocks for at least 3 years
def Rows_Greater_Level(months, level=100):
    rows_month = months.apply(len)
    start_month = rows_month.index[rows_month >= level].min().replace(day=1)
    end_month = rows_month.index[rows_month >= level].max()
    return start_month, end_month, (rows_month >= level).sum()


# load the data and rank-normalize it
def Rankise(market, method):
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
    clms = raw_data.columns[~raw_data.columns.isin(['TARGET'])]
    df = raw_data[clms]

    # Handle NA cases
    prep_data = Handle_NA(df, 'median')

    raw_data_na = pd.merge(raw_data[['PERMNO', 'DATE', 'TARGET']], prep_data, on=['PERMNO', 'DATE'])
    raw_data_na.dropna(inplace=True, how='any')

    # group by month
    months = raw_data_na.groupby(pd.Grouper(key='DATE', freq='M'))

    # To confirm we have more than 100 observations in each month
    start_month, end_month, continous_months = Rows_Greater_Level(months)

    # enough data for 3 years
    if continous_months >= 36:
        # retain NA
        prep_data = months.apply(partial(rank_all))
        prep_data.sort_index(inplace=True)

        prep_data = prep_data[prep_data['DATE'] >= start_month]
        end_month = min(end_month, pd.to_datetime('2017-12-01'))
        prep_data = prep_data[prep_data['DATE'] <= end_month]
        prep_data.dropna(inplace=True, how='any')
        prep_data.reset_index(drop=True, inplace=True)

        print(prep_data.shape)
        # save data
        local_path = join(feature_path, "%s_rankNA.csv" % market) 
        prep_data.to_csv(local_path, index=False)
    else:
        print('Warning: %s should be removed from list!' % market)


if __name__ == '__main__':
    markets_l = load_markets(your_path)
    for market in markets_l:
        print(market)
        Rankise(market, 'median')