# train and prediction neural network models:

import os
import time
import torch
from mylog import logger
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import torch.optim as optim
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score


filename = os.path.basename(__file__)
logger = logger(filename)


class NetAP(nn.Module):

    def __init__(self):
        super(NetAP, self).__init__()
        # constructing neural network:
        layers_unit = [26, 32, 16, 8,  1]
        self.fc1 = nn.Linear(layers_unit[0], layers_unit[1])
        self.bn1 = nn.BatchNorm1d(layers_unit[1])
        self.fc2 = nn.Linear(layers_unit[1], layers_unit[2])
        self.bn2 = nn.BatchNorm1d(layers_unit[2])
        self.fc3 = nn.Linear(layers_unit[2], layers_unit[3])
        self.bn3 = nn.BatchNorm1d(layers_unit[3])
        self.fc4 = nn.Linear(layers_unit[3], layers_unit[4])

        # self.fc5 = nn.Linear(layers_unit[4], layers_unit[5])

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        #x = F.relu(self.fc3(x))
        # x = F.relu(self.fc4(x))
        # x = F.relu(self.fc5(x))
        y = self.fc4(x)
        return y


class Dataset(torch.utils.data.Dataset):

    def __init__(self, df_x, df_y):
        self.dfx = df_x
        self.dfy = df_y

    def __len__(self):
        return self.dfx.shape[0]

    def __getitem__(self, index):
        x = self.dfx.iloc[index].values.astype('float32')
        y = self.dfy.iloc[index].values.astype('float32')
        return(x, y)


def adjust_lr(lr, optimizer, gamma, max_epoch, epoch):
    step_epoches = (int(max_epoch*0.3), int(max_epoch*0.6), int(max_epoch*0.8))
    if epoch in step_epoches:
        logger.info('*---------------------lr decay------------------*')
        lr = lr * gamma
        logger.info('lr decay to {:.6f}'.format(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def train(dfx_train, dfy_train, cfg):
    net = NetAP()
    # setting loss type:
    if cfg.loss == 'huber':# robust to outliers
        loss_func = torch.nn.HuberLoss(reduction='mean', delta=cfg.delta)
    elif cfg.loss == 'mse':
        loss_func = nn.MSELoss()
    if cfg.GPU:
        net.cuda()
    optimizer = optim.SGD(net.parameters(), lr=cfg.lr, momentum=cfg.momentum,
                weight_decay=cfg.weight_decay)
    trainset = Dataset(dfx_train, dfy_train)
    train_dl = DataLoader(trainset, batch_size=cfg.batch,
                  shuffle=True, pin_memory=False, num_workers=4)
    lr = cfg.lr
    iter_num = int(len(trainset) / cfg.batch)
    for epoch in range(cfg.max_epoch):
        lr = adjust_lr(lr, optimizer, cfg.gamma, cfg.max_epoch, epoch)
        for iteration, (x, y) in enumerate(train_dl):
            batch_x = x
            batch_y = y
            if cfg.GPU:
                batch_x = batch_x.cuda()
                batch_y = batch_y.cuda()
            # Forward pass
            output = net(batch_x)
            loss = loss_func(output, batch_y)
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (iteration + 1) % 50 == 0:
                logger.info('Epoch: [{}/{}],step:[{}/{}], lr: {:.6f}, Loss: {:.4f}'.format(epoch+1,
                      cfg.max_epoch, iteration+1, iter_num, lr, loss.item()))
            #if (epoch+1) % 5 ==0:
            #   torch.save(net.state_dict(), cfg.save_folder + '/Net_' + repr(epoch+1) + '.pth')
    return net


def predict(dfx_test, dfy_test, net, batch, GPU):
    testset = Dataset(dfx_test, dfy_test)
    test_dl = DataLoader(testset, batch_size=batch, shuffle=False)
    predictions, actuals = list(), list()
    for i, (x, y) in enumerate(test_dl):
        # evaluate the model on the test set
        if GPU:
            x = x.cuda()
        yhat = net(x)
        # retrieve numpy array
        yhat = yhat.cpu().detach().numpy()
        yreal = y.numpy()
        yreal = yreal.reshape((len(yreal), 1))
        # store
        predictions.append(yhat)
        actuals.append(yreal)
    pred_y, actuals = np.vstack(predictions), np.vstack(actuals)
    # calculate mse
    mse = mean_squared_error(actuals, pred_y)
    R2 = r2_score(actuals, pred_y)*100
    return pred_y, mse, R2


class train_cfg:

    def __init__(self, lr, batch, max_epoch, momentum,
                 gamma, weight_decay, loss, GPU, delta):
        self.GPU = GPU
        self.lr = lr
        self.batch = batch
        self.loss = loss
        self.max_epoch = max_epoch
        self.momentum = momentum
        self.gamma = gamma
        self.weight_decay = weight_decay
        self.delta = delta


if __name__ =='__main__':
    X_train = pd.read_csv('./test/X_train1.csv')
    y_train = pd.read_csv('./test/y_train1.csv')
    X_test = pd.read_csv('./test/X_test.csv')
    y_test = pd.read_csv('./test/y_test.csv')
    batch = 512
    max_epoch = 10
    lr = 0.001
    gamma = 0.1
    weight_decay = 4e-5
    momentum = 0.9
    cfg = train_cfg(lr, batch, max_epoch, momentum,
          gamma, weight_decay)
    net_trianed = train(X_train, y_train, cfg)
    pred_y = predict(X_test, y_test, net_trianed, batch, cfg.GPU)