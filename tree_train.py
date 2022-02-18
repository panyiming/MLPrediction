# Lasso regression , randomforest, and  neural network.
# author: pan.yiming@upf.edu


import os
import datetime
import warnings
import mltnet
import pandas as pd
import numpy as np
import xgboost as xg
from mylog import logger
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from new_features import MONFEATURES
from new_features import TARGET


ANCHORS = ['ticker', 'w_date', 'm_date', 'q_date', 'y_date', 'date', 'n_date']
MONFEATURES1 = ['Mkt-RF', 'RMW', 'me', 'beta', 'trade_volume', 'bm', 'apr', 'oidpr',
            'ret5', 'ret21', 'ret63', 'avg_beta21', 'avg_me21', 'vol21', 'sp500vol21',
            'IYH_mret21', 'mret63', 'sk63', 'vol63', 'sp500_mret63', 'sp500vol63',
            'IYH_mret63', 'QUAL_mret63', 'IYW_mret63', 'mret252', 'vol252', 'sp500_mret252',
             'sp500sk252']


warnings.filterwarnings('ignore')
filename = os.path.basename(__file__)
logger = logger(filename)


# sort dates
def sort_dates(date):
    return datetime.datetime.strptime(date,"%Y-%m-%d").timestamp()


# delete extreme values:
def del_extreme(df_data):
    cols = MONFEATURES + TARGET
    for col_name in cols:
        p025 = df_data[col_name].quantile(q=0.001)
        p975 = df_data[col_name].quantile(q=0.95)
        df_data = df_data[df_data[col_name] <= p975]
        #df_data = df_data[(df_data[col_name] > p025) & (df_data[col_name] < p975)]
    return df_data


def get_df_train(df_data, target_date):
    logger.info('processing data')
    # get train and test dataframe:
    df_test = df_data[df_data['n_date'] == target_date]
    start_date, end_date = df_data[df_data['n_date'] == target_date][['y_date', 'date']].values[0]
    df_train = df_data[(df_data['n_date'] >= start_date) & (df_data['n_date'] <= end_date)]
    # df_train = df_data[df_data['n_date'] <= end_date]

    # delete train data extreme values:
    pred_df = df_test[['n_date', 'ticker', TARGET[0], 'mret252']]
    df_train = del_extreme(df_train)
    # normalize train data and test data
    scaler = StandardScaler()
    df_train[MONFEATURES] = scaler.fit_transform(df_train[MONFEATURES])
    df_test[MONFEATURES] = scaler.transform(df_test[MONFEATURES])

    df_train = df_train.drop(ANCHORS, axis=1)
    df_test = df_test.drop(ANCHORS, axis=1)
    y_train = pd.DataFrame(columns=TARGET)
    y_test = pd.DataFrame(columns=TARGET)
    X_train = df_train.drop(TARGET, axis=1)
    y_train[TARGET] = df_train[TARGET]
    X_test = df_test.drop(TARGET, axis=1)
    y_test[TARGET] = df_test[TARGET]
    return X_train, y_train, X_test, y_test, pred_df


# the documents of randomforest:
# https://scikit-learn.org/0.15/modules/generated/sklearn.ensemble.RandomForestRegressor.html
def rf_model(X_train, y_train, X_test, y_test, params_dict):
    logger.info('traning randomforest regeression model')
    rf_reg1 = RandomForestRegressor(n_estimators=params_dict['n_estimators'],
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

    rf_reg = xg.XGBRegressor(objective='reg:squarederror',
                             learning_rate=0.30,
                             eval_metric=["error", "logloss"],
                             n_estimators=8,
                             max_depth=8,
                             min_child_weight=40,
                             subsample=1.0,
                             colsample_bytree=1,
                             seed=0)

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
    dates = list(set(df_train['n_date']))
    dates = sorted(dates, key=lambda date:sort_dates(date))
    date_num = len(dates)
    df_preds = []
    # R2_ls_net_train, MSE_ls_net_train = [], []
    R2_ls_rf_train, MSE_ls_rf_train = [], []
    # R2_ls_net_test, MSE_ls_net_test = [], []
    R2_ls_rf_test, MSE_ls_rf_test = [], []

    feat_importance_all = {}
    for feat_i in MONFEATURES:
        feat_importance_all[feat_i] = 0

    rooling_i = 0
    for i in range(roll_length+1, date_num):
        rooling_i += 1
        target_date = dates[i]
        logger.info('*------------Training {}/{} day prediction model------------*'.format(rooling_i, date_num-roll_length+1))

        # get train data :
        X_train, y_train, X_test, y_test, pred_df = get_df_train(df_train, target_date)

        # train random forest model:
        rf_reg, mse_train_rf, R2_train_rf, mse_test_rf, R2_test_rf, feat_importance = rf_model(X_train, y_train[pred_var],
                                                                                      X_test, y_test[pred_var], cfg_rf)
        # record feature importance:
        for feat_i in MONFEATURES:
            feat_importance_all[feat_i] = feat_importance_all[feat_i] + feat_importance[feat_i]
        # predict return on target date:
        pred_rf = rf_reg.predict(X_test)
        logger.info('random forest train MSE: {:.6f}, random forest train R2: {:.6f}'.format(mse_train_rf, R2_train_rf))
        logger.info('random forest test MSE: {:.6f}, random forest test R2: {:.6f}'.format(mse_test_rf, R2_test_rf))
        MSE_ls_rf_train.append(mse_train_rf)
        MSE_ls_rf_test.append(mse_test_rf)
        R2_ls_rf_train.append(R2_train_rf)
        R2_ls_rf_test.append(R2_test_rf)

        # neural_reg = neural_model(X_train, y_train, X_test, y_test, cfg_net)
        # pred_net = neural_reg.predict(X_test)
        # net_trianed = mltnet.train(X_train, y_train, cfg_net)
        # train_pred_net, mse_train_net, R2_train_net = mltnet.predict(X_train, y_train, net_trianed, cfg_net.batch, cfg_net.GPU)
        # test_pred_net,  mse_test_net, R2_test_net = mltnet.predict(X_test, y_test, net_trianed, cfg_net.batch, cfg_net.GPU)
        # logger.info('network train MSE: {:.6f}, network train R2: {:.6f}'.format(mse_train_net, R2_train_net))
        # logger.info('network test MSE: {:.6f}, network test R2: {:.6f}'.format(mse_test_net, R2_test_net))
        # MSE_ls_net_train.append(mse_train_net)
        # MSE_ls_net_test.append(mse_test_net)
        # R2_ls_net_train.append(R2_train_net)
        # R2_ls_net_test.append(R2_test_net)

        df_pred_i = pred_df
        df_pred_i['pred_rf'] = pred_rf
        # df_pred_i['pred_nnet'] = test_pred_net
        df_preds.append(df_pred_i)
    df_pred = pd.concat(df_preds, ignore_index=True)
    logger.info('RF: average train MSE: {:.6f}, average train R2: {:.6f}'.format(np.mean(MSE_ls_rf_train), np.mean(R2_ls_rf_train)))#
    logger.info('RF: average test MSE: {:.6f},  average test R2 : {:.6f}'.format(np.mean(MSE_ls_rf_test), np.mean(R2_ls_rf_test)))
    # logger.info('DL: average train MSE of neteork: {:.6f}, the average train R2 of neteork: {:.6f}'.format(np.mean(MSE_ls_net_train), np.mean(R2_ls_net_train)))
    # logger.info('DL: average test MSE of neteork: {:.6f}, the average test R2 of neteork: {:.6f}'.format(np.mean(MSE_ls_net_test), np.mean(R2_ls_net_test)))
    for feat_i in MONFEATURES:
        feat_importance_all[feat_i] = feat_importance_all[feat_i]/rooling_i
    logger.info(feat_importance_all)
    return df_pred


if __name__ == '__main__':
    data_path = '../data/df_train_monthly.csv'
    path_pred = '../data/df_pred_monthly.csv'
    roll_length = 12
    pred_var = 'ret_n21'
    df_train = pd.read_csv(data_path)
    df_train = df_train.drop(['Unnamed: 0'], axis=1)
    # random forest parameters setting:
    cfg_rf = {'n_estimators':10,
                 'loss':'mse',
                 'max_depth':8,
                 'min_samples_split':200,
                 'min_samples_leaf':100,
                 'n_jobs': 4}

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
    GPU = False
    net_loss = 'huber'
    delta = 0.3
    cfg_net = mltnet.train_cfg(lr, batch, max_epoch, momentum,
              gamma, weight_decay, net_loss, GPU, delta)
    df_pred = rolling_fit(df_train, pred_var, roll_length, cfg_rf, cfg_net)
    df_pred.to_csv(path_pred)
