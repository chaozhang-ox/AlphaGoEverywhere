'''
Merge all data into onefile for each market, including US factor and GapQ, local factor
'''

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from functools import partial
import os
from os.path import join

from Load_Data import *

your_path = '/data01/AI_Finance'
feature_path = join(your_path, 'Data')


def Combine_VWF_GapQ(market, scale):
    print('# ' * 20 + market + ' #' * 20)
    USnameF = 'USVWF'
    USnameG = 'USGapQ'
    LnameF = 'LVWF'
    if scale:
        USnameF = 'SUSVWF'
        USnameG = 'SUSGapQ'
        LnameF = 'SLVWF'

    US_factor_data = pd.read_csv(join(feature_path, "%s_%s.csv" % (market, USnameF)))
    cors_f_US = [i for i in US_factor_data.columns if i not in ['TARGET', 'PERMNO', 'DATE']]
    cors_f_US = [i.split('-'+USnameF)[0] for i in cors_f_US if '-'+USnameF in i]

    US_gapq_data = pd.read_csv(join(feature_path, "%s_%s.csv" % (market, USnameG)))
    L_factor_data = pd.read_csv(join(feature_path, "%s_%s.csv" % (market, LnameF)))

    for i in cors_f_US:
        a = (US_factor_data[i] == US_gapq_data[i])

        if a.all():
            pass
        else:
            print(i)

    vwf_gapq_data = US_factor_data.copy()
    for i in cors_f_US:
        vwf_gapq_data[i+'-'+USnameG] = US_gapq_data[i+'-'+USnameG]
        vwf_gapq_data[i+'-'+LnameF] = L_factor_data[i+'-'+LnameF]

    vwf_gapq_data.reset_index(drop=True, inplace=True)
    print(vwf_gapq_data)
    vwf_gapq_data.to_csv(join(feature_path, "%s_%s+%s.csv" % (market, 'USLVWF', 'USLGapQ')), index=False)


if __name__ == '__main__':
    markets_l = load_markets(your_path)
    scale = False
    for market in markets_l:
        print(' - ' * 20 + market + ' - ' * 20)
        Combine_VWF_GapQ(market, scale)
