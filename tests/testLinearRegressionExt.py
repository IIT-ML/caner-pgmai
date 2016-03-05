from sklearn.linear_model import Ridge

from utils.node import Neighborhood
from utils.readdata import convert_time_window_df_randomvar
from models.LinearRegressionExt import LinearRegressionExt

import numpy as np
import pickle

def computeTransitionParametersLinearChain(train_mat):
    sensorCount = train_mat.shape[0]
    beta = np.zeros(shape=(sensorCount, 2), dtype=np.float_)
    sigmasq = np.zeros(shape=sensorCount, dtype=np.float_)
    for sensor in range(sensorCount):
        X = train_mat[sensor, :-1].reshape(-1, 1)
        y = train_mat[sensor, 1:].reshape(-1, 1)

        degree = 1
        reg = LinearRegressionExt(degree=degree)
        reg.fit(X,y)
        beta[sensor] = [reg.intercept_,reg.coef_]
        y_pred = reg.predict(X)
        err = y - y_pred
        # print err

        # mu = np.mean(err.reshape(-1))
        var = np.cov(err.reshape(-1))
        # stddev = np.std(err.reshape(-1))
        sigmasq[sensor] = var
    return beta, sigmasq


def computeTransitionParametersCompleteNetwork(train_mat):
    sensorCount = train_mat.shape[0]
    beta = np.zeros(shape=(sensorCount,sensorCount + 1), dtype=np.float_)
    sigmasq = np.zeros(shape=sensorCount, dtype=np.float_)
    X = train_mat[:, :-1].T
    for sensor in range(sensorCount):
        y = train_mat[sensor, 1:].reshape(-1,1)
        degree = 1
        reg = LinearRegressionExt(degree=degree)
        reg.fit(X, y)
        beta[sensor, 0] = reg.intercept_
        beta[sensor, 1:] = reg.coef_
        y_pred = reg.predict(X)
        err = y - y_pred
        # print err

        # mu = np.mean(err.reshape(-1))
        var = np.cov(err.reshape(-1))
        # stddev = np.std(err.reshape(-1))
        sigmasq[sensor] = var
    return beta, sigmasq


def computeBetasSigmasqs():
    neighborhood_def = Neighborhood.itself_previous_others_current
    train_set, test_set = convert_time_window_df_randomvar(True, neighborhood_def)

    train_mat = np.vectorize(lambda x: x.true_label)(train_set)

    # beta, sigmasq = computeTransitionParametersLinearChain(train_mat=train_mat)
    beta, sigmasq = computeTransitionParametersCompleteNetwork(train_mat=train_mat)
    # print beta
    # print sigmasq
    pickle.dump((beta, sigmasq), open(
            'C:\\Users\\ckomurlu\\Documents\\workbench\\experiments\\20160301\\completeNetworkParameters.pkl', 'wb'
    ))


def computeMeanCov():
    neighborhood_def = Neighborhood.itself_previous_others_current
    train_set, test_set = convert_time_window_df_randomvar(True, neighborhood_def)

    train_mat = np.vectorize(lambda x: x.true_label)(train_set)
    mu = np.mean(train_mat, axis=1)
    sigmasq = np.var(train_mat, axis=1)
    pickle.dump((mu, sigmasq), open(
            'C:\\Users\\ckomurlu\\Documents\\workbench\\experiments\\20160301\\initialSliceParameters.pkl', 'wb'
    ))
    return mu, sigmasq


def computeRidgeCoefficients():
    neighborhood_def = Neighborhood.itself_previous_others_current
    train_set, test_set = convert_time_window_df_randomvar(True, neighborhood_def)

    train_mat = np.vectorize(lambda x: x.true_label)(train_set)
    sensorCount = train_mat.shape[0]
    beta = np.zeros(shape=(sensorCount,2), dtype=np.float_)
    sigmasq = np.zeros(shape=sensorCount, dtype=np.float_)
    for sensor in range(sensorCount):
        X = train_mat[sensor, :-1].reshape(-1, 1)
        y = train_mat[sensor, 1:].reshape(-1, 1)
        reg = Ridge()
        reg.fit(X,y)
        beta[sensor, 0] = reg.intercept_
        beta[sensor, 1:] = reg.coef_
        y_pred = reg.predict(X)
        err = y - y_pred
        # print err

        # mu = np.mean(err.reshape(-1))
        var = np.cov(err.reshape(-1))
        # stddev = np.std(err.reshape(-1))
        sigmasq[sensor] = var
    pickle.dump((beta, sigmasq), open(
            'C:\\Users\\ckomurlu\\Documents\\workbench\\experiments\\20160301\\ridgeLinearChainParameters.pkl', 'wb'
    ))

computeRidgeCoefficients()

# Checking how much data represents a normal distribution
# print 'Symmetry of data around mean'
# print np.count_nonzero(err > mu)
# print 'Count instances within [mu-sigma, mu+sigma]'
# flag_mat = np.all(np.hstack((err > mu - stddev, err < mu + stddev)), axis=1)
# print np.count_nonzero(flag_mat)/143.0

