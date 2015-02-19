'''
Created on Feb 4, 2015

@author: ckomurlu
'''


import numpy as np
from time import time
from sklearn.linear_model import LassoLars

from joint.iterative_classifier import ICAModel
from utils.node import RandomVarNode
from main import all_others_current_time
from utils.readdata import convert_time_window_df_randomvar


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
    neighborhood_def = all_others_current_time
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
    
#     regr = LinearRegression()
#     regr = SVR(kernel='linear')
    regr = LassoLars(alpha=.1)

    regr.fit(xtrain, ytrain)
    
    
    row_count,col_count = test_set.shape
    xtest = np.empty(shape=(row_count,col_count,4),dtype=np.bool_)
    for row in range(row_count):
        for col in range(col_count):
            xtest[row,col] = test_set[row,col].local_feature_vector
#     xtrain = np.vectorize(lambda x: x.local_feature_vector)(train_set)
    ytest = np.vectorize(lambda x: x.true_label)(test_set)
    xtest = xtrain.reshape(-1,4)
    ytest = ytrain.reshape(-1,1)
    
    print np.mean((regr.predict(xtest)-ytest)**2)
    print regr.coef_,regr.intercept_
    
    print 'end of regression'
    
main()