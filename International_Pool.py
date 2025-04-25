'''
Intergrate all standardlised market data into one international data
'''

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from functools import partial
import os
from os.path import join

from Load_Data import *


# your local path to store the data and results
your_path = '/data01/AI_Finance/'


# decentre one column
def decentre_column(sr):
    assert not sr.isnull().any()
    result = sr - sr.mean()
    return result


def mydecentre(df):
    df['TARGET_Decentre'] = decentre_column(df['TARGET'])
    return df


def International_Pool():
    ### Merge standardlised data
    df_l = []
    markets_l = load_markets(your_path)
    for market in markets_l:
        print(market)
        tmp_df = load_data(your_path, market)
        tmp_df['PERMNO'] = market + "+" + tmp_df['PERMNO'].astype('str')
        df_l.append(tmp_df)

    df = pd.concat(df_l)
    df.reset_index(drop=True, inplace=True)

    del df['TARGET_Decentre']

    df['DATE'] = pd.to_datetime(df['DATE'])
    # group by month
    months = df.groupby(pd.Grouper(key='DATE', freq='M'))
    prep_data = months.apply(partial(mydecentre))
    prep_data.sort_index(inplace=True)

    prep_data.dropna(inplace=True, how='any')
    prep_data.reset_index(drop=True, inplace=True)

    prep_data.to_csv(join(your_path, 'Data', 'World_norm.csv'))


def International_Size():
    ### Merge raw size data
    df_l = []
    llen_l = []
    markets_l = load_markets(your_path)
    for market in markets_l[1:]:
        print(market)
        tmp_df, llen = load_rawsize(market)
        print(llen)
        tmp_df['PERMNO'] = market + "+" + tmp_df['PERMNO'].astype('str')
        df_l.append(tmp_df)
        llen_l.append(llen)

    df = pd.concat(df_l)
    df.reset_index(drop=True, inplace=True)

    local_path = join(your_path, 'features', 'World_rawsize.csv')
    df.to_csv(local_path, index=False)


def Get_Dummy(data_df):
    one_hot = pd.get_dummies(data_df['Market'], drop_first=True)
    # Drop column as it is now encoded
    data_df = data_df.drop('Market', axis=1)
    # Join the encoded df
    data_df_dummy = data_df.join(one_hot)
    return data_df_dummy


def International_Pool_Dummy():
    ### Merge standardlised data with dummy variables representing the corresponding market
    df_l = []
    markets_l = load_markets(your_path)
    for market in markets_l[1:-3]:
        print(market)
        tmp_df = load_data(market)
        tmp_df['PERMNO'] = market + "+" + tmp_df['PERMNO'].astype('str')
        tmp_df['Market'] = market
        df_l.append(tmp_df)

    df = pd.concat(df_l)
    df.reset_index(drop=True, inplace=True)
    df_dummy = Get_Dummy(df)
    print(df)
    print(df_dummy)

    local_path = join(your_path, 'features', 'WorldDummy_decenTARGET.csv')
    df_dummy.to_csv(local_path)


def International_Size_Dummy():
    ### Merge raw size data with dummy variables representing the corresponding market
    df_l = []
    llen_l = []
    markets_l = load_markets(your_path)
    for market in markets_l[1:-3]:
        print(market)
        tmp_df, llen = load_rawsize(market)
        print(llen)
        tmp_df['PERMNO'] = market + "+" + tmp_df['PERMNO'].astype('str')
        tmp_df['Market'] = market
        df_l.append(tmp_df)
        llen_l.append(llen)

    df = pd.concat(df_l)
    df.reset_index(drop=True, inplace=True)
    df_dummy = Get_Dummy(df)

    print(df_dummy)
    print(sum(llen_l))

    local_path = join(your_path, 'features', 'WorldDummy_rawsize.csv')
    df_dummy.to_csv(local_path, index=False)


if __name__ == '__main__':
    # International_Pool()
    International_Pool_Dummy()
    International_Size_Dummy()