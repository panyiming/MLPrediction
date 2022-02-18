# Lasso regression , randomforest, and  neural network.
# author: pan.yiming@barcelonagse.eu


import os
import warnings
import datetime
import mltnet
import pandas as pd
import numpy as np
from mylog import logger
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler


warnings.filterwarnings('ignore')
filename = os.path.basename(__file__)
logger = logger(filename)

'''
FEATURES = ['open', 'high', 'low', 'close', 'adjclose',
            'volume', 'ret', 'atq', 'cheq', 'cshoq',
            'oiadpq', 'seqq', 'IYW_ret', 'IXC_ret', 'IYH_ret',
            'IDU_ret', 'IAU_ret', 'QUAL_ret', 'EFG_ret', '^GSPC_ret',
            'SPY_ret', 'beta', 'me', 'avg_size', 'bm', 'avg_ret',
            'prof', 'cash', 'avg_beta', 'prc-7', 'prc-21', 'prc-180',
            'mret7', 'mret21', 'mret180']
'''

FEATURES = ['prof','bm','IXC_ret','atq','mret180','cash','cshoq','oiadpq','EFG_ret','seqq',
            'avg_beta','cheq','IDU_ret','IYH_ret','beta','IAU_ret','QUAL_ret','prc-180','IYW_ret',
            'avg_size','SPY_ret','avg_ret','^GSPC_ret','me','prc-21','volume']

# sort dates
def sort_dates(date):
    return datetime.datetime.strptime(date,"%Y-%m-%d").timestamp()


# get initial train data
def get_df(data_path, adj_var, ret_var):
    logger.info('loading data')
    df_data = pd.read_csv(data_path)
    # select train data:
    # col_names:'pre7a', 'mret7a', 'pre21a', 'mret21a', 'pre63a', 'mret63a'
    df_data = df_data[df_data[adj_var]>-1]
    # delete irrelevant data:
    col_names = FEATURES + ['t_day', 'ticker', ret_var]
    df_data = df_data[col_names]
    return df_data


# delete extreme values:
def del_extreme(df_data, pred_var):
    cols = FEATURES + [pred_var]
    for col_name in cols:
        p025 = df_data[col_name].quantile(q=0.005)
        p975 = df_data[col_name].quantile(q=0.995)
        df_data = df_data[(df_data[col_name] > p025) & (df_data[col_name] <= p975)]
    return df_data


def get_df_train(df_data, dates, target_date, pred_var, roll_length=180):
    logger.info('processing data')
    start_idx = dates.index(target_date) - roll_length
    start_date = dates[start_idx]
    # get train and test data dataframe:
    df_test = df_data[df_data['t_day'] == target_date]
    df_train = df_data[(df_data['t_day'] >= start_date) & (df_data['t_day'] < target_date)].drop(['t_day', 'ticker'], axis=1)

    # delete train data extreme values:
    df_train = del_extreme(df_train, pred_var)

    # normalize train data and test data
    scaler = StandardScaler()
    df_train[FEATURES] = scaler.fit_transform(df_train[FEATURES])
    df_test[FEATURES] = scaler.transform(df_test[FEATURES])
    ticker_date_test_y = df_test[['t_day', 'ticker', pred_var]]
    df_test = df_test.drop(['t_day', 'ticker'], axis=1)

    y_train = pd.DataFrame(columns=[pred_var])
    y_test = pd.DataFrame(columns=[pred_var])
    X_train = df_train.drop([pred_var], axis=1)
    y_train[pred_var] = df_train[pred_var]
    X_test = df_test.drop([pred_var], axis=1)
    y_test[pred_var] = df_test[pred_var]
    return X_train, y_train, X_test, y_test, ticker_date_test_y


# lasso regression model documents:
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html
def lasso_model(X_train, y_train, X_test, y_test):
    logger.info('traning lasso regeression model')
    reg = Lasso(alpha=0.01,
                fit_intercept=True,
                normalize=False,
                warm_start=True,
                max_iter=4000,
                tol=0.0001,
                random_state=42,
               selection='random')
    reg.fit(X_train, y_train)
    logger.info('Lasso Regression: R^2 score on training set')
    logger.info('Lasso Regression: R^2 score on test set', reg.score(X_test, y_test) * 100)
    return reg


# the documents of randomforest:
# https://scikit-learn.org/0.15/modules/generated/sklearn.ensemble.RandomForestRegressor.html
def rf_model(X_train, y_train, X_test, y_test, params_dict):
    logger.info('traning randomforest regeression model')
    rf_reg = RandomForestRegressor(n_estimators=params_dict['n_estimators'],
                                   criterion=params_dict['loss'],
                                   max_depth=params_dict['max_depth'],
                                   min_samples_split=params_dict['min_samples_split'],
                                   min_samples_leaf=params_dict['min_samples_leaf'],
                                   max_features='auto',
                                   max_leaf_nodes=None,
                                   bootstrap=True,
                                   oob_score=False,
                                   n_jobs=params_dict['n_jobs'],
                                   random_state=2,
                                   verbose=True)
    rf_reg.fit(X_train, y_train)
    R2_train = rf_reg.score(X_train, y_train) * 100
    R2_test = rf_reg.score(X_test, y_test) * 100
    pred_y_test = rf_reg.predict(X_test)
    pred_y_train = rf_reg.predict(X_train)
    mse_test = mean_squared_error(y_test, pred_y_test)
    mse_train = mean_squared_error(y_train, pred_y_train)
    feats_importance = {}  # a dict to hold feature_name: feature_importance
    for feature, importance in zip(X_train.columns, rf_reg.feature_importances_):
        feats_importance[feature] = importance  # add the name/value pair
    logger.info(feats_importance)
    return rf_reg, mse_train, R2_train, mse_test, R2_test, feats_importance


# the documents link:
# https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html
def neural_model(X_train, y_train, X_test, y_test, params_dict):
    logger.info('traning neural regeression model')
    neueral_reg = MLPRegressor(hidden_layer_sizes=params_dict['layers_unit'],
                               activation='relu',
                               solver='sgd',
                               alpha=params_dict['alpha'],
                               batch_size='auto',
                               learning_rate='adaptive',
                               learning_rate_init=params_dict['init_lr'],
                               power_t=0.5,
                               max_iter=params_dict['max_iter'],
                               shuffle=True,
                               random_state=2,
                               tol=params_dict['tol'],
                               verbose=True,
                               warm_start=True,
                               momentum=0.9,
                               nesterovs_momentum=True,
                               early_stopping=True,
                               validation_fraction=0.1,
                               n_iter_no_change=params_dict['n_iter_no_change'])
    neueral_reg.fit(X_train, y_train)
    logger.info('neueral Regression: R^2 score on training set', neueral_reg.score(X_train, y_train) * 100)
    logger.info('neueral Regression: R^2 score on test set', neueral_reg.score(X_test, y_test) * 100)
    return neueral_reg


def rolling_fit(df_train, pred_var, roll_length, cfg_rf, cfg_net):
    dates = list(set(df_train['t_day']))
    dates = sorted(dates, key=lambda date:sort_dates(date))
    date_num =len(dates)+1
    df_preds = []
    R2_ls_net_train, MSE_ls_net_train = [], []
    R2_ls_rf_train, MSE_ls_rf_train = [], []
    R2_ls_net_test, MSE_ls_net_test = [], []
    R2_ls_rf_test, MSE_ls_rf_test = [], []
    feat_importance_all = {}
    for feat_i in FEATURES:
        feat_importance_all[feat_i] = 0

    rooling_i = 0
    for i in range(roll_length+1, date_num, 10):#roll_length+2):
        rooling_i += 1
        target_date = dates[i]
        logger.info('*------------Training {}th/{} day prediction model------------*'.format(i-roll_length, date_num-roll_length+1))
        X_train, y_train, X_test, y_test, ticker_date_test_y = get_df_train(df_train, dates,
                                                               target_date, pred_var, roll_length=roll_length)
        rf_reg, mse_train_rf, R2_train_rf, mse_test_rf, R2_test_rf, feat_importance = rf_model(X_train, y_train[pred_var],
                                                                                      X_test, y_test[pred_var], cfg_rf)

        for feat_i in FEATURES:
            feat_importance_all[feat_i] = feat_importance_all[feat_i] + feat_importance[feat_i]

        pred_rf = rf_reg.predict(X_test)
        logger.info('random forest train MSE: {:.6f}, random forest train R2: {:.6f}'.format(mse_train_rf, R2_train_rf))
        logger.info('random forest test MSE: {:.6f}, random forest test R2: {:.6f}'.format(mse_test_rf, R2_test_rf))
        MSE_ls_rf_train.append(mse_train_rf)
        MSE_ls_rf_test.append(mse_test_rf)
        R2_ls_rf_train.append(R2_train_rf)
        R2_ls_rf_test.append(R2_test_rf)

        # neural_reg = neural_model(X_train, y_train, X_test, y_test, cfg_net)
        # pred_net = neural_reg.predict(X_test)
        net_trianed = mltnet.train(X_train, y_train, cfg_net)
        train_pred_net, mse_train_net, R2_train_net = mltnet.predict(X_train, y_train, net_trianed, cfg_net.batch, cfg_net.GPU)
        test_pred_net,  mse_test_net, R2_test_net = mltnet.predict(X_test, y_test, net_trianed, cfg_net.batch, cfg_net.GPU)
        logger.info('network train MSE: {:.6f}, network train R2: {:.6f}'.format(mse_train_net, R2_train_net))
        logger.info('network test MSE: {:.6f}, network test R2: {:.6f}'.format(mse_test_net, R2_test_net))
        MSE_ls_net_train.append(mse_train_net)
        MSE_ls_net_test.append(mse_test_net)
        R2_ls_net_train.append(R2_train_net)
        R2_ls_net_test.append(R2_test_net)

        df_pred_i = ticker_date_test_y
        df_pred_i['pred_rf'] = pred_rf
        df_pred_i['pred_nnet'] = test_pred_net
        df_preds.append(df_pred_i)
    df_pred = pd.concat(df_preds, ignore_index=True)
    logger.info('RF: average train MSE: {:.6f}, average train R2: {:.6f}'.format(np.mean(MSE_ls_rf_train), np.mean(R2_ls_rf_train)))#
    logger.info('RF: average test MSE: {:.6f},  average test R2 : {:.6f}'.format(np.mean(MSE_ls_rf_test), np.mean(R2_ls_rf_test)))
    logger.info('DL: average train MSE of neteork: {:.6f}, the average train R2 of neteork: {:.6f}'.format(np.mean(MSE_ls_net_train), np.mean(R2_ls_net_train)))
    logger.info('DL: average test MSE of neteork: {:.6f}, the average test R2 of neteork: {:.6f}'.format(np.mean(MSE_ls_net_test), np.mean(R2_ls_net_test)))
    for feat_i in FEATURES:
        feat_importance_all[feat_i] = feat_importance_all[feat_i]/rooling_i
    logger.info(feat_importance_all)
    return df_pred


if __name__ == '__main__':
    data_path = 'E:/data_pym/MLAP/df_train.csv'
    path_pred = 'E:/data_pym/MLAP/df_pred.csv'

    ret_var = 'mret63a'
    adj_var = 'pre63a'
    df_train = get_df(data_path, adj_var, ret_var)
    roll_length = 120

    # random forest parameters setting:
    cfg_rf = {'n_estimators':8,
                 'loss':'mse',
                 'max_depth':10,
                 'min_samples_split':100,
                 'min_samples_leaf':50,
                 'n_jobs': 8}

    # neural network parameters setting:
    params_ne = {'layers_unit':[70, 35, 18, 9],
                 'alpha': 0.05,
                 'init_lr': 0.01,
                 'max_iter': 1000,
                 'tol': 0.00001,
                 'n_iter_no_change': 40}

    batch = 2000
    max_epoch = 15
    lr = 0.01
    gamma = 0.1
    weight_decay = 1e-4
    momentum = 0.9
    GPU = True
    net_loss = 'huber'
    delta = 0.3
    cfg_net = mltnet.train_cfg(lr, batch, max_epoch, momentum,
              gamma, weight_decay, net_loss, GPU, delta)
    df_pred = rolling_fit(df_train, ret_var, roll_length, cfg_rf, cfg_net)
    df_pred.to_csv(path_pred)
