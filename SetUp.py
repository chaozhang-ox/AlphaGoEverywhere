# variables used in the model
# PERMNO and DATE are the stock identifier and date, respectively
# TARGET is the dependent variable, i.e. future returns
# the rest are the independent variables
variables = ['PERMNO', 'DATE', 'TARGET', 'mom_1', 'mvel1', 'mom_6', 'mom_12', 'chmom_6', 'maxret', 'indmom_a_12',
            'retvol', 'dolvol', 'sp', 'turn', 'bm', 'ep', 'cfp', 'bm_ia', 'cfp_ia', 'herf', 'mve_ia', 'lev', 'pctacc',
            'stddolvol', 'stdturn', 'dy', 'salecash', 'ill', 'cashpr', 'depr', 'acc', 'absacc', 'roe', 'egr',
            'agr', 'cashdebt', 'lgr', 'sgr', 'chpmia']

USA_columns_dic = {'permno': 'PERMNO', 'RET': 'TARGET', 'std_turn': 'stdturn', 'std_dolvol': 'stddolvol',
                   'mom1m': 'mom_1', 'mom6m': 'mom_6', 'mom12m':'mom_12', 'chmom': 'chmom_6', 'indmom': 'indmom_a_12',
                   'roaq': 'roa', 'mve': 'mvel1'}

China_columns_dic = {'stkcd': 'PERMNO', 'ret':'mom_1', 'target': 'TARGET', 'ym': 'DATE'}

Others_columns_dic = {'sedol': 'PERMNO', 'target': 'TARGET', 'ym': 'DATE'}

