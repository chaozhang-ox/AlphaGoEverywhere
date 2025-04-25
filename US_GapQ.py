'''
Compute the USA characteristic gaps and its interaction with the market.
'''

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from functools import partial
import os
from os.path import join
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns

from SetUp import *
from Load_Data import *

# your local path to store the data and results
your_path = '/data01/AI_Finance'
feature_path = join(your_path, 'Data')



# calculate the average value of each decile in a certain month
def calc_dec(month_data, var, num=10, level=0):
    month_name = month_data.index[0].strftime('%Y-%m-%d')
    # diff between decile10 and decile1
    var_h_l = pd.DataFrame(np.percentile(month_data[var], 95) - np.percentile(month_data[var], 5),
                           index=['H-L'], columns=[month_name])
    return var_h_l


def Local_Gap(market):
    final_data, start_year, end_year = load_raw_data(market)
    final_data['DATE'] = pd.to_datetime(final_data['DATE'], format='%Y-%m-%d')
    vars_l = [i for i in final_data if i not in ['PERMNO', 'DATE', 'TARGET']]
    local_factor_df_l = []
    for var in vars_l:
        months_l = [g.set_index('DATE') for _, g in final_data.groupby(pd.Grouper(key='DATE', freq='M'))]  # python3
        var_h_l = [calc_dec(month_data, var) for month_data in months_l if month_data.shape[0] != 0]
        tmp_local_factor_df = pd.concat(var_h_l, axis=1).T
        local_factor_df_l.append(tmp_local_factor_df)

    local_factor_df = pd.concat(local_factor_df_l, axis=1)
    local_factor_df.columns = vars_l
    return local_factor_df


def Rolling_Scale(local_factor_df):
    local_factor_df.index = pd.to_datetime(local_factor_df.index)
    year_df = local_factor_df.groupby(local_factor_df.index.year).mean()
    assert not np.isnan(year_df).any().any()

    new_month_l = []
    for i, g in local_factor_df.groupby(local_factor_df.index.year):
        if i == year_df.index[0]:
            new_month_df = g.pct_change() + 1
        else:
            new_month_df = g / year_df.loc[i-1]

        new_month_l.append(new_month_df)

    scale_local_factor_df = pd.concat(new_month_l)
    return scale_local_factor_df


def Interaction_Market(market, us_factor_df, scale_us_factor_df, scale):
    final_data, start_year, end_year = load_data(market)
    month_name_l = set(final_data['DATE'].tolist())
    month_name_l = list(month_name_l)
    month_name_l.sort()

    sub_df_l = []
    influent_features = [i for i in us_factor_df.columns if i not in ['PERMNO', 'DATE', 'TARGET']]

    name = 'USGapQ'

    if scale:
        us_factor_df = scale_us_factor_df
        name = 'SUSGapQ'

    for month_name in month_name_l:
        sub_df = final_data[final_data['DATE'] == month_name]
        for var in influent_features:
            sub_df[var+'-'+name] = sub_df[var] * us_factor_df.loc[month_name, var]

        sub_df_l.append(sub_df)

    new_data = pd.concat(sub_df_l)
    new_data.reset_index(drop=True, inplace=True)
    print(new_data)
    new_data.to_csv(join(feature_path, "%s_%s.csv" % (market, name)), index=False)


if __name__ == '__main__':
    markets_l = load_markets(your_path)
    scale = False
    for market in markets_l:
        print(' - ' * 20 + market + ' - ' * 20)
        Interaction_Market(market, scale)

