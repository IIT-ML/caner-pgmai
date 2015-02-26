'''
Created on Feb 4, 2015

@author: ckomurlu
'''


import numpy as np
from time import time
from sklearn.linear_model import Lasso,LinearRegression
import copy

from joint.iterative_classifier import ICAModel
from utils.node import RandomVarNode
from utils.node import Neighborhood
from utils.readdata import convert_time_window_df_randomvar

from sklearn.preprocessing import PolynomialFeatures
from independent.local_mean_regressor import LocalMeanRegressor


def main():
    rs = np.random.RandomState()
    rs.seed(seed = 0)
    Y_test = rs.randint(0,2,12).reshape(4,-1)
    print Y_test
#     Y_pred = rs.randint(0,2,12).reshape(4,-1)
    Y_pred = Y_test.copy()
    Y_pred[0] = 1 - Y_pred[0]
    Y_pred[1,:] = 1 - Y_pred[1,:]
    print Y_pred
    test_set = np.vectorize(lambda x: RandomVarNode(true_label=x))(Y_test)
    evidence_mat = np.zeros(Y_test.shape, dtype=bool)
#     evidence_mat[0] = 1 
    print np.sum(evidence_mat)
    
#     ica_model = ICAModel()
#     acc = ica_model.compute_accuracy(test_set, Y_pred, 1, evidence_mat)
#     print acc

def regression_main():
    begin = time()
    neighborhood_def = Neighborhood.all_others_current_time
    train_set,test_set = convert_time_window_df_randomvar(True,
                                                          neighborhood_def)
    row_count,col_count = train_set.shape
    xtrain = np.empty(shape=(row_count,col_count,4),dtype=np.float_)
    for row in range(row_count):
        for col in range(col_count):
            xtrain[row,col] = train_set[row,col].local_feature_vector
#     xtrain = np.vectorize(lambda x: x.local_feature_vector)(train_set)
    ytrain = np.vectorize(lambda x: x.true_label)(train_set)
    xtrain = xtrain.reshape(-1,4)
    ytrain = ytrain.reshape(-1,1)
    
    
    regr = LinearRegression()
#     regr = SVR(kernel='linear')
    
    poly = PolynomialFeatures(degree=3)
    xtrain2 = poly.fit_transform(xtrain)

    regr.fit(xtrain2, ytrain)
    
    row_count,col_count = test_set.shape
    xtest = np.empty(shape=(row_count,col_count,4),dtype=np.bool_)
    for row in range(row_count):
        for col in range(col_count):
            xtest[row,col] = test_set[row,col].local_feature_vector
#     xtrain = np.vectorize(lambda x: x.local_feature_vector)(train_set)
    ytest = np.vectorize(lambda x: x.true_label)(test_set)
    xtest = xtrain.reshape(-1,4)
    ytest = ytrain.reshape(-1,1)
    
    xtest2 = poly.fit_transform(xtest)
    
    print np.mean((regr.predict(xtest2)-ytest)**2)
    print regr.coef_,regr.intercept_
    
    print 'end of regression - duration: ', time() - begin
    
def test_local_mean_fit():
    neighborhood_def = Neighborhood.itself_previous_others_current
    train_set,test_set = convert_time_window_df_randomvar(True,
                                                          neighborhood_def)
    lmreg = LocalMeanRegressor()
    lmreg.fit(train_set)
    Y_pred = lmreg.predict(train_set)
    print lmreg.compute_accuracy(train_set, Y_pred)
    Y_pred = lmreg.predict(test_set)
    print lmreg.compute_accuracy(test_set, Y_pred)
    
test_local_mean_fit()