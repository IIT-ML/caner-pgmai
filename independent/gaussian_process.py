'''
Created on Mar 9, 2015

@author: ckomurlu
'''

from utils.readdata import convert_time_window_df_randomvar_hour
from utils.node import Neighborhood
from models.ml_reg_model import MLRegModel

import numpy as np
from sklearn.gaussian_process import GaussianProcess
from sklearn.metrics import mean_absolute_error,mean_squared_error
from time import time

class GaussianProcessLocal(MLRegModel):
    def __init__(self):
        self.gpmat = object()
    
    def fit(self,train_mat):
        Xtrain = np.vectorize(lambda x: x.local_feature_vector)(train_mat)
        ytrain = np.vectorize(lambda x: x.true_label)(train_mat)
        
        ytrain_split = np.split(ytrain,3,axis=1)
        ytrain_split_array = np.array(ytrain_split)
        ytrain_mean = np.mean(ytrain_split_array,axis=0)
        
        row_count = Xtrain.shape[0]
        self.gpmat = np.empty(shape=(row_count,),dtype=np.object_)
        for row in range(row_count):
            self.gpmat[row] = GaussianProcess(corr='cubic', theta0=1e-2, thetaL=1e-4, thetaU=1e-1,
                             random_start=100)
            self.gpmat[row].fit(Xtrain[row,:48].reshape(-1,1), ytrain_mean[row])
    
    def predict(self,test_mat):
        Xtest = np.vectorize(lambda x: x.local_feature_vector)(test_mat)
        ytest = np.vectorize(lambda x: x.true_label)(test_mat)
        
        ypred = np.empty(shape=ytest.shape,dtype=ytest.dtype)
        
        row_count = Xtest.shape[0]

        # gp.predict(Xtest)
        for row in range(row_count):
            ypred[row] = self.gpmat[row].predict(Xtest[row].reshape(-1,1))
        return ypred
    
#     def compute_mean_absolute_error(self):
#         raise NotImplementedError
#         
#     def copute_mean_squared_error(self):
#         raise NotImplementedError
     
    def compute_accuracy(self, Y_test, Y_pred):
        raise NotImplementedError
     
    def compute_confusion_matrix(self, Y_test, Y_pred):
        raise NotImplementedError
    
    @staticmethod
    def run():
        start = time()
        neighborhood_def = Neighborhood.all_others_current_time
        trainset,testset = convert_time_window_df_randomvar_hour(True,
                                                                neighborhood_def)
        
        gp = GaussianProcessLocal()
        gp.fit(trainset)
        
        ypred = gp.predict(trainset)
        
        print gp.compute_mean_squared_error(trainset, ypred)
        print gp.compute_mean_absolute_error(trainset, ypred)
        
#         print mean_absolute_error(ytest, ypred)
#         print mean_squared_error(ytest, ypred)
        
        end = time()
        print 'Process ended, duration:', end - start
        
GaussianProcessLocal.run()