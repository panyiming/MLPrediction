# read and reorganize data
# author: pan.yiming@barcelonagse.eu


import os
import numpy as np
from tqdm import tqdm
import pandas as pd
from train_models import FEATURES


# compute future return separately: in one trading day, one trading month, one trading quarter.
def firm_data(df_firm):
    adjclose_ls = list(df_firm['adjclose'])
    ls_week = adjclose_ls[6:]
    ls_week.extend([-1 for i in range(6)])
    df_firm['pre7a'] = ls_week
    df_firm['mret7a'] = df_firm['pre7a'] / df_firm['adjclose'] - 1
    ls_month = adjclose_ls[20:]
    ls_month.extend([-1 for i in range(20)])
    df_firm['pre21a'] = ls_month
    df_firm['mret21a'] = df_firm['pre21a'] / df_firm['adjclose'] - 1
    ls_quarter = adjclose_ls[62:]
    ls_quarter.extend([-1 for i in range(62)])
    df_firm['pre63a'] = ls_quarter
    df_firm['mret63a'] = df_firm['pre63a'] / df_firm['adjclose'] - 1
    return df_firm

# concat all yearly data files into a whole .sv file.
def contact_dfs(root_dir):
    dfs = []
    for file_i in os.listdir(root_dir):
        if '.csv' in file_i:
            path_i = os.path.join(root_dir, file_i)
            df_i = pd.read_csv(path_i)
            dfs.append(df_i)
    df_all = pd.concat(dfs, ignore_index=True)
    return df_all


def get_rets(df, full_num):
    firm_ls = set(list(df['ticker']))
    firm_ls = list(firm_ls)
    df_ls = []#pd.DataFrame(columns=[])
    firm_num = len(firm_ls)
    j = 0
    for i in tqdm(range(firm_num)):
        firm_sym = firm_ls[i]
        j += 1
        df_i = df[df['ticker']==firm_sym]
        if df_i.shape[0] == full_num:
            df_i_new = firm_data(df_i)
            df_ls.append(df_i_new)
            #df_all = df_all.append(df_i_new, ignore_index=True)
    df_all = pd.concat(df_ls, ignore_index=True)
    return df_all


# get initial train data
def get_df_trian(df_data):
    # col_names:'pre7a', 'mret7a', 'pre21a', 'mret21a', 'pre63a', 'mret63a'
    # delete irrelevant data:
    col_names = FEATURES + ['t_day', 'ticker', 'pre7a',  'pre21a', 'pre63a',
                            'mret7a', 'mret21a', 'mret63a']
    df_data = df_data[col_names]
    # delete invalid values:
    df_data = df_data[df_data.replace([np.inf, -np.inf], np.nan).notnull().all(axis=1)]
    df_data = df_data.dropna(axis=0, how='any', inplace=False)
    full_num = len(set(df_data['t_day']))
    print('deleting  firms without full data')
    df_data = get_full_data(df_data, full_num)
    return df_data


def get_full_data(df, full_num):
    firm_ls = set(list(df['ticker']))
    firm_ls = list(firm_ls)
    pbar = tqdm(total=len(firm_ls))
    for firm_i in firm_ls:
        pbar.update(1)
        df_i = df[df['ticker']==firm_i]
        if df_i.shape[0] != full_num:
            df = df.drop(df[df['ticker']==firm_i].index)
    return df


if __name__ == '__main__':
    data_root = '../data/init_data/'
    df_all = contact_dfs(data_root)
    path_new = '../data/df_train_new.csv'
    full_num = len(set(df_all['t_day']))
    df_all = get_rets(df_all, full_num)
    df_trian = get_df_trian(df_all)
    df_trian.to_csv(path_new)