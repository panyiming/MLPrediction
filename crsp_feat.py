# processing data we get from: https://drive.google.com/drive/folders/1TrYzMUA_xLID5-gXOy_as8sH2ahLwz-l
# there are three parts of these data: macro_vars, individual stock vars, ret vars.
# pan.yiming@upf.edu: 2021/12/16

import pandas as pd
import numpy as np
from tqdm import tqdm


def trans_date(date_old):
    str_list = list(str(date_old))
    str_list.insert(4, '-')
    str_list.insert(7, '-')
    date_str = ''.join(str_list)
    return date_str


def firm_data(feat_array_i, company_name, col_names, dates,
              st_idx, num_th):
    valid_num = np.sum(feat_array_i[st_idx:,  0] > -99)
    if valid_num >= num_th:
        df_firm = pd.DataFrame(data=feat_array_i, columns=col_names)
        df_firm['name'] = company_name
        df_firm['date'] = dates
        return df_firm
    else:
        return 0


def concat_df(test_np, col_names, dates, st_idx=0, num_th=300):
    df_all = test_np['data']
    num = df_all.shape[1]
    df_ls = []
    for i in tqdm(range(num)):
        feat_array_i = df_all[:, i, :]
        company_name = 'company_{}'.format(i)
        df_firm_i = firm_data(feat_array_i, company_name,
                    col_names, dates, st_idx, num_th)
        if df_firm_i is not 0:
            df_ls.append(df_firm_i)
    df_all_new = pd.concat(df_ls, ignore_index=True)
    return df_all_new


if __name__ == '__main__':
    test_np = np.load('../../../../datasets/char/Char_test.npz')
    col_names = test_np['variable'].tolist()
    dates = test_np['date'].tolist()
    dates = [trans_date(i) for i in dates]
    df_all_new = concat_df(test_np, col_names, dates, st_idx=0, num_th=300)
    df_all_new.to_csv('../data/test_data_crsp.csv')
