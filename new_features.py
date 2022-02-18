# get new features from initial dataset
# author: pan.yiming@upf.edu
# date: 2021-11-08

import numpy as np
from tqdm import tqdm
import pandas as pd

MONFEATURES = [ # factors
                'Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'Mom',
                # variables on last date
                'me',  'beta', 'trade_volume', 'trade_ratio', 'bm',
                'cfpr', 'apr', 'oidpr', 'ret5', 'ret21', 'ret63',
                # 21 days statistic variables
                'avg_beta21', 'avg_me21', 'trade_ratio21', 'mret21', 'sk21', 'vol21',
                'sp500_mret21', 'sp500sk21', 'sp500vol21',
                'IXC_mret21', 'EFG_mret21', 'IDU_mret21',
                'IYH_mret21', 'QUAL_mret21', 'IYW_mret21',
                # 63 days statistic variables
                'trade_ratio63', 'mret63', 'sk63', 'vol63',
                'sp500_mret63', 'sp500sk63', 'sp500vol63',
                'IXC_mret63', 'EFG_mret63', 'IDU_mret63',
                'IYH_mret63', 'QUAL_mret63', 'IYW_mret63',
                # 252 days statistic variables
                'trade_ratio252', 'mret252', 'sk252', 'vol252',
                'sp500_mret252', 'sp500sk252',  'sp500vol252']
TARGET = [# regression target:
           'ret_n21']

def trans_date(date_old):
    str_list = list(str(date_old))
    str_list.insert(4, '-')
    date_str = ''.join(str_list)
    date_new = date_str + '-01'
    return date_new

def date_ahead1mon(date):
    date_ls = date.split('-')
    date_ls[1] = str(int(date_ls[1]) + 1).zfill(2)
    date1 = '-'.join(date_ls)
    return date1


def get_date_bounds(month_ls):
    date_bounds = []
    length = len(month_ls)
    for i, date in enumerate(month_ls):
        if i < length - 1:
            date_1 = month_ls[i + 1]
            date_bounds.append(date_1)
        else:
            date_1 = '2021-06-01'
            date_bounds.append(date_1)
    return date_bounds


def get_anchor_dates(date_bound, date_ls):
    month_i_dates = [i for i in date_ls if i < date_bound]
    next_bound = date_ahead1mon(date_bound)
    month_i_dates1 = [i for i in date_ls if i < next_bound]
    last_date = month_i_dates[-1]
    last_inx = date_ls.index(last_date)
    pre_year_date = date_ls[last_inx-252]
    pre_quarter_date = date_ls[last_inx-63]
    pre_month_date = date_ls[last_inx-21]
    pre_week_date = date_ls[last_inx-5]
    nex_last = month_i_dates1[-1]
    return pre_week_date, pre_month_date, pre_quarter_date, pre_year_date, last_date, nex_last


# get previous return rate:
def get_returns(df_stock, w_date, m_date, q_date, l_date, n_date):
    pre5 = df_stock[df_stock['t_day'] == w_date]['adjclose'].values[0]
    pre21 = df_stock[df_stock['t_day'] == m_date]['adjclose'].values[0]
    pre63 = df_stock[df_stock['t_day'] == q_date]['adjclose'].values[0]
    pre0 = df_stock[df_stock['t_day'] == l_date]['adjclose'].values[0]
    n21 = df_stock[df_stock['t_day'] == n_date]['adjclose'].values[0]
    ret5 = pre5/pre0 - 1
    ret21 = pre21/pre0 - 1
    ret63 = pre63/pre0 - 1
    ret_n21 = n21/pre0 - 1
    ret_ls = [ret5, ret21, ret63]
    return ret_ls, ret_n21


def get_stat_var21(df_stock21):
    # 63 days statistic variables:  15 dims
    # 'avg_beta21', 'avg_me21', 'trade_ratio21', 'mret21', 'sk21', 'vol21',
    # 'sp500_mret21', 'sp500sk21', 'sp500vol21',
    # 'IXC_mret21', 'EFG_mret21', 'IDU_mret21',
    # 'IYH_mret21', 'QUAL_mret21', 'IYW_mret21'
    avg_beta21 = df_stock21['beta'].mean()
    avg_me21 = df_stock21['me'].mean()
    trade_ratio21 = df_stock21['trade_ratio'].mean()
    mret21 = df_stock21['ret'].mean()
    sk21 = df_stock21['ret'].skew()
    vol21 = df_stock21['ret'].var()
    sp500_mret21 = df_stock21['ret'].mean()
    sp500sk21 = df_stock21['ret'].skew()
    sp500vol21 = df_stock21['ret'].var()
    IXC_mret21 = df_stock21['IXC_ret'].mean()
    EFG_mret21 = df_stock21['EFG_ret'].mean()
    IDU_mret21 = df_stock21['IDU_ret'].mean()
    IYH_mret21 = df_stock21['IYH_ret'].mean()
    QUAL_mret21 = df_stock21['QUAL_ret'].mean()
    IYW_mret21 = df_stock21['IYW_ret'].mean()
    var21_ls = [avg_beta21, avg_me21, trade_ratio21, mret21, sk21, vol21,
                sp500_mret21, sp500sk21, sp500vol21,
                IXC_mret21, EFG_mret21, IDU_mret21,
                IYH_mret21, QUAL_mret21, IYW_mret21]
    return var21_ls


def get_stat_var63(df_stock63):
    # 63 days statistic variables:  13 dims
    # 'trade_ratio63', 'mret63', 'sk63', 'vol63',
    # 'sp500_mret63', 'sp500sk63', 'sp500vol63',
    # 'IXC_mret63', 'EFG_mret63', 'IDU_mret63',
    # 'IYH_mret63', 'QUAL_mret63', 'IYW_mret63',
    trade_ratio63 = df_stock63['trade_ratio'].mean()
    mret63 = df_stock63['ret'].mean()
    sk63 = df_stock63['ret'].skew()
    vol63 = df_stock63['ret'].var()
    sp500_mret63 = df_stock63['^GSPC_ret'].mean()
    sp500sk63 = df_stock63['^GSPC_ret'].skew()
    sp500vol63 = df_stock63['^GSPC_ret'].var()
    IXC_mret63 = df_stock63['IXC_ret'].mean()
    EFG_mret63 = df_stock63['EFG_ret'].mean()
    IDU_mret63 = df_stock63['IDU_ret'].mean()
    IYH_mret63 = df_stock63['IYH_ret'].mean()
    QUAL_mret63 = df_stock63['QUAL_ret'].mean()
    IYW_mret63 = df_stock63['IYW_ret'].mean()
    var63_ls = [trade_ratio63, mret63, sk63, vol63,
                sp500_mret63, sp500sk63, sp500vol63,
                IXC_mret63, EFG_mret63, IDU_mret63,
                IYH_mret63, QUAL_mret63, IYW_mret63]
    return var63_ls


def get_stat_var252(df_stock252):
    # 252 days statistic variables: 7 dims
    # 'mret252', 'sk252', 'vol252',
    # 'sp500_mret252', 'sp500sk252', 'sp500vol252'
    trade_ratio252= df_stock252['trade_ratio'].mean()
    mret252 = df_stock252['ret'].mean()
    sk252 = df_stock252['ret'].skew()
    vol252 = df_stock252['ret'].var()
    sp500_mret252 = df_stock252['^GSPC_ret'].mean()
    sp500sk252 = df_stock252['^GSPC_ret'].skew()
    sp500vol252 = df_stock252['^GSPC_ret'].var()
    var252_ls = [trade_ratio252, mret252, sk252, vol252,
                 sp500_mret252, sp500sk252, sp500vol252]
    return var252_ls


def stock_vars(df_stock, df_factors, ticker_name):
    # compute variables for every stock
    # we assume every year there are 252 trading days
    # we assume every quarter there are 63 trading days
    # we assume every month there are 21 trading days
    # we assume every week there are 5 trading days
    # get new market equity value:
    latest_num_shares = list(df_stock['cshoq'])[-1]
    df_stock['me'] = latest_num_shares * df_stock['adjclose']
    # get trading volume:
    df_stock['trade_volume'] = df_stock['volume'] * df_stock['adjclose']
    # get trade_ratio:
    df_stock['trade_ratio'] = df_stock['volume']/df_stock['cshoq']
    # get bm:
    df_stock['bm'] = df_stock['seqq']/df_stock['me']
    # get cash flow to price ratio cfpr:
    df_stock['cfpr'] = df_stock['cheq'] / df_stock['me']
    # get asset to price ratio cfpr:
    df_stock['apr'] = df_stock['atq'] / df_stock['me']
    # get operation income after depreciation to price ratio cfpr:
    df_stock['oidpr'] = df_stock['oiadpq'] / df_stock['me']
    month_ls = list(df_factors['date'])
    date_ls = list(df_stock['t_day'])
    date_bounds = get_date_bounds(month_ls)
    df_stock_new = pd.DataFrame(columns=[
                      # dates:
                      'ticker', 'w_date', 'm_date', 'q_date', 'y_date', 'date', 'n_date',
                      # factors
                      'Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'Mom',
                      # variables on last date
                      'me',  'beta', 'trade_volume', 'trade_ratio', 'bm',
                      'cfpr', 'apr', 'oidpr', 'ret5', 'ret21', 'ret63',
                      # 21 days statistic variables
                      'avg_beta21', 'avg_me21', 'trade_ratio21', 'mret21', 'sk21', 'vol21',
                      'sp500_mret21', 'sp500sk21', 'sp500vol21',
                      'IXC_mret21', 'EFG_mret21', 'IDU_mret21',
                      'IYH_mret21', 'QUAL_mret21', 'IYW_mret21',
                      # 63 days statistic variables
                      'trade_ratio63', 'mret63', 'sk63', 'vol63',
                      'sp500_mret63', 'sp500sk63', 'sp500vol63',
                      'IXC_mret63', 'EFG_mret63', 'IDU_mret63',
                      'IYH_mret63', 'QUAL_mret63', 'IYW_mret63',
                      # 252 days statistic variables
                      'trade_ratio252', 'mret252', 'sk252', 'vol252',
                      'sp500_mret252', 'sp500sk252',  'sp500vol252',
                      # regression target:
                      'ret_n21'])
    var_date_ls = ['me',  'beta', 'trade_volume', 'trade_ratio', 'bm',
                 'cfpr', 'apr', 'oidpr']
    line_i = 0
    for bound_i in date_bounds:
        w_date, m_date, q_date, y_date, l_date, n_date = get_anchor_dates(bound_i, date_ls)
        features_i = [ticker_name, w_date, m_date, q_date, y_date, l_date, n_date]
        # get factors: 6
        factor_date = l_date[0:8] + '01'
        features_i += df_factors[df_factors['date'] == factor_date].values.tolist()[0][2:]
        # get variables on last date: 8+3
        feat_date = df_stock[df_stock['t_day'] == l_date][var_date_ls].values.tolist()[0]
        rets, ret_n21 = get_returns(df_stock, w_date, m_date, q_date, l_date, n_date)
        features_i += feat_date + rets
        # get 21 days statistic variables:
        df_stock21 = df_stock[(df_stock['t_day'] > m_date)&(df_stock['t_day'] <= l_date)]
        stat_var21 = get_stat_var21(df_stock21)
        features_i += stat_var21
        # get 63 days statistic variables:
        df_stock63 = df_stock[(df_stock['t_day'] > q_date)&(df_stock['t_day'] <= l_date)]
        stat_var63 = get_stat_var63(df_stock63)
        features_i += stat_var63
        # get 252 days statistic variables: 7
        df_stock252 = df_stock[(df_stock['t_day'] > y_date)&(df_stock['t_day'] <= l_date)]
        stat_var252 = get_stat_var252(df_stock252)
        features_i += stat_var252
        features_i += [ret_n21]
        df_stock_new.loc[line_i, :] = features_i
        # print(w_date, m_date, q_date, y_date, l_date, n_date)
        line_i += 1
    return df_stock_new


def process_all(df_all, df_factors):
    tickers = set(list(df_all['ticker']))
    dfs = []
    for ticker_i in tqdm(tickers):
        df_ticker_i = df_all[df_all['ticker'] == ticker_i]
        df_ticker_i_new = stock_vars(df_ticker_i, df_factors, ticker_i)
        dfs.append(df_ticker_i_new)
    df_all_new = pd.concat(dfs, ignore_index=True)
    return df_all_new

if __name__ == '__main__':
    df_train = pd.read_csv('../data/df_train.csv')
    df_f6 = pd.read_csv('../data/FF6_monthly.csv')
    df_train_monthly = process_all(df_train, df_f6)
    df_train_monthly.to_csv('../data/df_train_monthly.csv')
    #df_stock = df_train[df_train['ticker']=='aapl']
    #df_stock_new = stock_vars(df_stock, df_f6)
    #df_stock_new.to_csv('../data/df_apple_new.csv')