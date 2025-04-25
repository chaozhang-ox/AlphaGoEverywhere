'''
Compute the USA factor and its interaction with the market.
'''

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from functools import partial
import os
from os.path import join

from SetUp import *
from Load_Data import *

# your local path to store the data and results
your_path = '/data01/AI_Finance'
feature_path = join(your_path, 'Data')



# calculate the average value of each decile in a certain month
def calc_dec(month_data, var, way='size', num=10, level=0):
    month_name = month_data.index[0].strftime('%Y-%m-%d')
    try:
        if way == 'size':
            tmp_data = month_data.groupby(pd.qcut(month_data[var], q=num)).apply(size_weight_decile)
        else:
            tmp_data = month_data.groupby(pd.qcut(month_data[var], q=num)).mean()
        target_month = pd.DataFrame(tmp_data['TARGET'].values, index=range(1, num+1), columns=[month_name])
    except:
        target_month = pd.DataFrame(month_data['TARGET'].mean(), index=range(1, num+1), columns=[month_name])

    target_h_l = pd.DataFrame(target_month.loc[num - level].values[0] - target_month.loc[1 + level].values[0],
                              index=['H-L'], columns=[month_name])

    return target_h_l


# size-weighted return
def size_weight_decile(tmp_df):
    actu_rt = (tmp_df['mom_1'] * tmp_df['size']).sum() / tmp_df['size'].sum()
    return pd.DataFrame(data={'TARGET': actu_rt}, index=[None])


def Local_Factor(market):
    final_data, start_year, end_year = load_raw_data(market)
    final_data['DATE'] = pd.to_datetime(final_data['DATE'], format='%Y-%m-%d')
    vars_l = [i for i in final_data.columns if i not in ['PERMNO', 'DATE', 'TARGET', 'size']]
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
            new_month_df = g.diff() + 1
        else:
            new_month_df = g - year_df.loc[i-1] + 1

        new_month_l.append(new_month_df)

    scale_local_factor_df = pd.concat(new_month_l)
    return scale_local_factor_df


def Interaction_Market(market, us_factor_df, scale_us_factor_df, scale):
    final_data, _ = load_data(market)
    month_name_l = set(final_data['DATE'].tolist())
    month_name_l = list(month_name_l)
    month_name_l.sort()

    name = 'USVWF'

    if scale:
        us_factor_df = scale_us_factor_df
        name = 'SUSVWF'

    sub_df_l = []
    influent_features = [i for i in us_factor_df.columns if i not in ['PERMNO', 'DATE', 'TARGET']]

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
