'''
Performance of Machine Learning Portfolios
'''

import pandas as pd
import numpy as np
import os
from sklearn.metrics import mean_squared_error
import time
from os.path import join
from Load_Data import *
from SetUp import *

# your local path to store the data and results
your_path = '/data01/AI_Finance/'
basedir = join(your_path, 'result_analysis')


# the customized performance evaluation in the paper, benchmark is 0, not the mean of testing data
def oos_r2(pred_df, test_y):
    diff = pred_df - test_y
    R2_oos = 1 - np.sum(diff ** 2) / np.sum(test_y ** 2)
    return R2_oos


# basic performance
def mse_r2_vi(forecast, market, model_name):
    df_l = []
    var_names = forecast.columns[~forecast.columns.isin(['TARGET', 'PERMNO', 'DATE'])]
    for i, variable_name in enumerate(var_names):
        mse = mean_squared_error(forecast[variable_name], forecast['TARGET'])
        r_2 = oos_r2(forecast[variable_name], forecast['TARGET']) * 100
        df_l.append(pd.DataFrame(data={'MSE': mse, 'R2': r_2, 'Variable': variable_name}, index=[i]))
    df = pd.concat(df_l)
    true_r2 = df[df['Variable'] == 'pred']['R2'].values[0]
    df['VI'] = true_r2 - df['R2']
    print("Basic Summary: ")
    print(df)
    # save
    local_path = join(basedir, market, '%s_mser2vi.csv' % model_name)
    df.to_csv(local_path)


# Statistics, including Sharpe Ratio, Monotonicity Test, oos-R2 based on Portfolio, Cumulative Return
def other_stats(forecast, market, model_name):
    raw_data = load_rawsize(market)
    newdf = pd.merge(forecast[['PERMNO', 'DATE', 'pred', 'TARGET', 'TARGET']],
                     raw_data[['PERMNO', 'DATE', 'size']],
                     on=['PERMNO', 'DATE'])
    # split data according to a certain month
    months_l = [g.set_index('DATE') for _, g in newdf.groupby(pd.Grouper(key='DATE', freq='M'))]  # python3
    # summary
    for way in ['equal', 'size']:
        pred_target_l = [calc_dec(month_data, way, num=10, level=0) for month_data in months_l if month_data.shape[0] != 0]
        sr_dd(pred_target_l, market, model_name, way)
        oos_r2_port(pred_target_l, market, model_name, way)
        cum_ret(pred_target_l, market, model_name, way)


# calculate the average value of each decile in a certain month
def calc_dec(month_data, way, num, level):
    month_name = month_data.index[0].strftime('%Y-%m-%d')
    month_data_copy = month_data.copy()
    # remove PERMNO column in month_data_copy
    month_data_copy.drop(columns='PERMNO', inplace=True)
    while True:
        try:
            if way == 'size':
                tmp_data = month_data_copy.groupby(pd.qcut(month_data_copy['pred'], q=num)).apply(size_weight_decile)
            else:
                tmp_data = month_data_copy.groupby(pd.qcut(month_data_copy['pred'], q=num)).mean()
            break
        except ValueError as e:
            if 'Bin edges must be unique' in str(e):
                num -= 1
                print(f'Warning: {month_name}: {way}: Adjusting num to {num} due to duplicate bin edges')
            else:
                raise e

    pred_month = pd.DataFrame(tmp_data['pred'].values, index=range(1, num + 1), columns=[month_name])
    target_month = pd.DataFrame(tmp_data['TARGET'].values, index=range(1, num + 1), columns=[month_name])

    # diff between decile10 and decile1
    if level == 0:
        pred_h_l = pd.DataFrame(pred_month.loc[num - level].values[0] - pred_month.loc[1 + level].values[0],
                                index=['H-L'], columns=[month_name])
        target_h_l = pd.DataFrame(target_month.loc[num - level].values[0] - target_month.loc[1 + level].values[0],
                                  index=['H-L'], columns=[month_name])
    else:
        pred_h_l = pd.DataFrame(pred_month.loc[num - level].values[0] - pred_month.loc[1 + level].values[0],
                                index=['%d-%d' % (num - level, 1 + level)], columns=[month_name])
        target_h_l = pd.DataFrame(target_month.loc[num - level].values[0] - target_month.loc[1 + level].values[0],
                                  index=['%d-%d' % (num - level, 1 + level)],
                                  columns=[month_name])
    pred_month = pd.concat([pred_month, pred_h_l])
    target_month = pd.concat([target_month, target_h_l])
    return pred_month, target_month


# size-weighted return
def size_weight_decile(tmp_df):
    pred_rt = (tmp_df['pred'] * tmp_df['size']).sum() / tmp_df['size'].sum()
    actu_rt = (tmp_df['TARGET'] * tmp_df['size']).sum() / tmp_df['size'].sum()
    return pd.DataFrame(data={'pred': pred_rt, 'TARGET': actu_rt}, index=[None])


# calculate average predicted monthly returns for each decile
def calc_pred(pred_target_l):
    pred_all = pd.concat([i[0] for i in pred_target_l], axis=1)
    pred_avg = pred_all.mean(axis=1)
    return pred_avg * 100


# calculate average realized monthly returns, standard deviations, annualized Sharpe Ratio
def calc_real(pred_target_l):
    real_all = pd.concat([i[1] for i in pred_target_l], axis=1)
    real_avg = real_all.mean(axis=1)
    real_std = real_all.std(axis=1)
    real_sr = real_avg * np.sqrt(12) / real_std
    return [real_avg * 100, real_std * 100, real_sr]


# statistics of Sharpe Ratio
def sr_dd(pred_target_l, market, model_name, way):
    # summary
    pred_avg = calc_pred(pred_target_l)
    real_l = calc_real(pred_target_l)
    real_l.insert(0, pred_avg)
    sum_all = pd.concat(real_l, axis=1)
    sum_all.columns = ['Pred', 'Avg', 'Std', 'SR']
    print("%s-Weighted Performance of Portfolio: " % way.upper())
    print(sum_all)
    # save
    local_path = join(basedir, market, '%s_%s-sr.csv' % (model_name, way))
    sum_all.to_csv(local_path)


# monotonicity test
def mono_test(pred_target_l, market, model_name, way, b=1000):
    real_all = pd.concat([i[1] for i in pred_target_l], axis=1)
    real_all.drop(index='H-L', inplace=True)
    mu_df = real_all.mean(axis=1)
    delta_df = mu_df.diff()
    J_orig = delta_df.min()
    # bootstrap
    mysum = 0
    for i in range(b):
        bs_sample = np.random.choice(real_all.columns, size=real_all.shape[1])
        example = real_all[bs_sample]
        mu_b_df = example.mean(axis=1)
        delta_b_df = mu_b_df.diff() - delta_df
        J_b = delta_b_df.min()
        mysum += 1 if J_b > J_orig else 0
    p_value = 1.0 * mysum / b
    print("%s-Weighted Mono Test P-value: %.3f" % (way.upper(), p_value))
    # save
    local_path = join(basedir, market, '%s_%s-monotest.csv' % (model_name, way))
    # local_path = '/scratch/AI_Finance2021/result_analysis/%s/%s/%s_%s-monotest.csv' % (market, model_name, way)
    pd.DataFrame(data={'MonoTest': p_value}, index=[model_name]).to_csv(local_path)


# convert average predicted and actual values of each decile to a form that matches input requirements of func oos_r2
def transform_month_data(df, id):
    if id == 0:
        df_copy = pd.DataFrame(data={'DATE': df[0].columns[0], 'PERMNO': range(1, 11), 'pred': df[0].values[:10, 0]})
    else:
        df_copy = pd.DataFrame(data={'DATE': df[1].columns[0], 'PERMNO': range(1, 11), 'TARGET': df[1].values[:10, 0]})

    return df_copy


# oos R2 based on portfolio returns
def oos_r2_port(pred_target_l, market, model_name, way):
    pred_all = pd.concat([transform_month_data(df, 0) for df in pred_target_l], axis=0)
    real_all = pd.concat([transform_month_data(df, 1) for df in pred_target_l], axis=0)
    merge_df = pd.merge(pred_all, real_all, on=['PERMNO', 'DATE'])
    port_r2 = oos_r2(merge_df['pred'], merge_df['TARGET'])
    print("%s-Weighted OOS R2 of Portfolio: %.3f" % (way.upper(), port_r2))
    # save
    local_path = join(basedir, market, '%s_%s-r2port.csv' % (model_name, way))
    # local_path = '/scratch/AI_Finance2021/result_analysis/%s/%s/%s_%s-r2port.csv' % (market, model_name, way)
    pd.DataFrame(data={'Port R2': port_r2}, index=[model_name]).to_csv(local_path)


# cumulative return of strategy
def cum_ret(pred_target_l, market, model_name, way):
    # return in each month
    ret_l = [(pd.to_datetime(tp[1].columns[0], format='%Y-%m-%d'), tp[1].iloc[-1].values[0],
              tp[1].iloc[-2].values[0], tp[1].iloc[0].values[0]) for tp in pred_target_l]
    ret_df = pd.DataFrame(ret_l, columns=['DATE', 'H-L+return', 'H+return', 'L+return'])
    ret_df.set_index('DATE', inplace=True)
    ret_df[['H-L+return', 'H+return', 'L+return']] += 1.0
    cumret_df = pd.DataFrame(data={'H-L+return': 1.0, 'H+return': 1.0, 'L+return': 1.0},
                             index=['Start'])
    # cumret_df = cumret_df.append(ret_df)
    cumret_df = pd.concat([cumret_df, ret_df])
    cumret_df['H-L+return'] =  cumret_df['H-L+return'].cumprod()
    cumret_df['H+return'] = cumret_df['H+return'].cumprod()
    cumret_df['L+return'] = cumret_df['L+return'].cumprod()

    cumret_df.rename(columns={'H-L+return': 'H-L+cum_return', 'H+return': 'H+cum_return', 'L+return': 'L+cum_return'},
                     inplace=True)
    cumret_df['dropdown'] = 0.0
    for i in cumret_df.index:
        max_current = cumret_df.loc[:i, 'H-L+cum_return'].max()
        tmp_dd = (cumret_df.loc[i, 'H-L+cum_return'] - max_current) / max_current
        cumret_df.loc[i, 'dropdown'] = tmp_dd * 100

    cumret_df['log10_H-L+cum_return'] = np.log10(cumret_df['H-L+cum_return'])
    cumret_df['log10_H+cum_return'] = np.log10(cumret_df['H+cum_return'])
    cumret_df['log10_L+cum_return'] = np.log10(cumret_df['L+cum_return'])

    print("%s-Weighted DropDown of Portfolio: %.1f" % (way.upper(), cumret_df['dropdown'].min()))

    # save
    local_path = join(basedir, market, '%s_%s-cumret.csv' % (model_name, way))
    cumret_df.to_csv(local_path)


def size_weight_mkt(tmp_df):
    return (tmp_df['TARGET'] * tmp_df['size']).sum() / tmp_df['size'].sum()


def weight_l(months_l, way):
    # different weighting ways
    if way == 'size':
        pred_target_l = [size_weight_mkt(tmp_df) for tmp_df in months_l]
    else:
        pred_target_l = [tmp_df['TARGET'].mean() for tmp_df in months_l if tmp_df.shape[0] != 0]
    pred_target_l = [x for x in pred_target_l if ~np.isnan(x)]
    return pred_target_l


# market sharpe ratio
def market_srdd(forecast, market):
    raw_data = load_rawsize(market)
    newdf = pd.merge(forecast[['PERMNO', 'DATE', 'TARGET']], raw_data[['PERMNO', 'DATE', 'size']],
                     on=['PERMNO', 'DATE'])
    months_l = [g for _, g in newdf.groupby(pd.Grouper(key='DATE', freq='M'))]  # python3
    # python2: months_l = [g for _, g in newdf.set_index('DATE').groupby(pd.TimeGrouper('M'))]
    for way in ['equal', 'size']:
        pred_target_l = weight_l(months_l, way)
        market_avg = np.mean(pred_target_l)
        market_std = np.std(pred_target_l)
        market_sr = market_avg * np.sqrt(12) / market_std
        cash_l = []
        cash_l.append(1.0)
        for i in pred_target_l:
            cash_l.append(cash_l[-1] * (1 + i))
        market_dd = 0.0
        for i in range(1, len(cash_l)):
            max_current = np.max(cash_l[:i])
            tmp_dd = (cash_l[i] - max_current) / max_current
            if tmp_dd < market_dd:
                market_dd = tmp_dd
        market_df = pd.DataFrame(
            data={'Market Avg': market_avg * 100, 'Market Std': market_std * 100, 'Market SR': market_sr,
                  'Market DD': market_dd * 100},
            index=[0])
        print("%s-Weighted Performance of Market: " % way.upper())
        print(market_df)
        # save
        local_path = join(basedir, market, '%s_market.csv' % (way))
        market_df.to_csv(local_path)


# analysis
def analysis_result(market, model_name_l):
    for model_name in model_name_l:
        print('* ' * 10 + model_name.upper() + ' *' * 10)
        # need to change the forecast path if you are using a different model, e.g. US-based or enhanced models
        forecast = pd.read_csv(join(your_path, 'forecasts', market, model_name+'_pred.csv'))
        forecast = forecast.drop(columns='Unnamed: 0')
        forecast['DATE'] = pd.to_datetime(forecast['DATE'], format='%Y-%m-%d')

        # remove the duplicate columns after merging and checking if they are the same 
        if 'TARGET_x' in forecast.columns:
            assert (forecast['TARGET_x'] == forecast['TARGET_y']).all()
            del forecast['TARGET_y']
            forecast.rename(columns={'TARGET_x': 'TARGET'}, inplace=True)
        else:
            pass
        
        # as forecasts and targets are in percentage
        # we need to convert them to decimal
        forecast['pred'] /= 100    
        forecast['TARGET'] /= 100
        result_dir = os.path.join(basedir, market)
        os.makedirs(result_dir, exist_ok=True)
        mse_r2_vi(forecast, market, model_name)
        other_stats(forecast, market, model_name)

    market_srdd(forecast, market)


if __name__ == '__main__':
    model_name_l = ['ols-3', 'linear', 'lasso', 'ridge', 'rf', 'gbrt+h'] # ['nn1', 'nn2', 'nn3', 'nn4', 'nn5']
    markets_l = load_markets(your_path)
    for market in markets_l:
        print(' *' * 20)
        print(' * ' * 6 + market + ' * ' * 6)
        print(' *' * 20)
        analysis_result(market)
