'''
Created on Mar 9, 2015

@author: ckomurlu
'''

from utils.readdata import convert_time_window_df_randomvar_hour, DATA_DIR_PATH
from utils.node import Neighborhood
from models.ml_reg_model import MLRegModel
import utils.properties
from utils.toolkit import standard_error

import numpy as np
from sklearn.gaussian_process import GaussianProcess
from sklearn.metrics import mean_absolute_error,mean_squared_error
from time import time
import cPickle as cpk
import os

class GaussianProcessLocal(MLRegModel):
    def __init__(self):
        self.gpmat = object()
    
    def fit(self,train_mat, load=False):
        if load:
            tempgp = cpk.load(open(DATA_DIR_PATH+'gaussianProcessLocal.pkl','rb'))
            self.gpmat = tempgp.gpmat
            self.rvCount = self.gpmat.shape[0]
        else:
            Xtrain = np.vectorize(lambda x: x.local_feature_vector)(train_mat)
            ytrain = np.vectorize(lambda x: x.true_label)(train_mat)
            
            ytrain_split = np.split(ytrain,3,axis=1)
            ytrain_split_array = np.array(ytrain_split)
            ytrain_mean = np.mean(ytrain_split_array,axis=0)
            
            self.rvCount = Xtrain.shape[0]
            self.gpmat = np.empty(shape=(self.rvCount,),dtype=np.object_)
            for row in range(self.rvCount):
                self.gpmat[row] = GaussianProcess(corr='cubic', theta0=1e-2, thetaL=1e-4, thetaU=1e-1,
                                 random_start=100)
                self.gpmat[row].fit(Xtrain[row,:48].reshape(-1,1), ytrain_mean[row])
    
    def predict(self,test_mat, evid_mat = None):
        if evid_mat is None:
            evid_mat = np.zeros(shape=test_mat.shape,dtype=np.bool_)
        Xtest = np.vectorize(lambda x: x.local_feature_vector)(test_mat)
        ytest = np.vectorize(lambda x: x.true_label)(test_mat)
        
        ypred = np.empty(shape=ytest.shape,dtype=ytest.dtype)
        
        row_count = Xtest.shape[0]

        # gp.predict(Xtest)
        for row in range(row_count):
            if evid_mat[row]:
                ypred[row] = ytest[row]
            else:  
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
#         neighborhood_def = Neighborhood.all_others_current_time
#         trainset,testset = convert_time_window_df_randomvar_hour(True,
#                                                                 neighborhood_def)
        trainset,testset = convert_time_window_df_randomvar_hour(True,
                            Neighborhood.itself_previous_others_current)
        
        
        testset = testset[:,35:47]
        gp = GaussianProcessLocal()
        gp.fit(trainset,load=True)
        
        evid_mat = np.zeros(shape=testset.shape,dtype=np.bool_)
        
        ypred = gp.predict(testset,evid_mat)
        
        print gp.compute_mean_squared_error(testset, ypred)
        print gp.compute_mean_absolute_error(testset, ypred)
        
#         print mean_absolute_error(ytest, ypred)
#         print mean_squared_error(ytest, ypred)
        
        end = time()
        print 'Process ended, duration:', end - start
    
    @staticmethod    
    def runActiveInference():
        start = time()
        randomState = np.random.RandomState(seed=0)
        numTrials = utils.properties.numTrials
        T = utils.properties.timeSpan
        trainset,testset = convert_time_window_df_randomvar_hour(True,
                            Neighborhood.itself_previous_others_current)
        gp = GaussianProcessLocal()
        gp.fit(trainset,load=True)
        for obsrate in utils.properties.obsrateList:
            obsCount = obsrate * gp.rvCount
            errResults = np.empty(shape=(numTrials,T,6))
            predResults = np.empty(shape=(numTrials, gp.rvCount, T))
            selectMat = np.empty(shape=(T,obsCount), dtype=np.int16)
            evidencepath = utils.properties.outputDirPath + str(obsrate) + '/evidences/'
            if not os.path.exists(evidencepath):
                os.makedirs(evidencepath)
            predictionpath = utils.properties.outputDirPath + str(obsrate) + '/predictions/'
            if not os.path.exists(predictionpath):
                os.makedirs(predictionpath)
            errorpath = utils.properties.outputDirPath + str(obsrate) + '/errors/'
            if not os.path.exists(errorpath):
                os.makedirs(errorpath)
            print 'obsrate: {}'.format(obsrate)
            print 'trial:'
            for trial in range(numTrials):
                print trial
                evidMat = np.zeros(shape=(gp.rvCount,T),dtype=np.bool_)
                print '\ttime:'
                for t in range(T):
                    print '\t',t
                    select = np.arange(gp.rvCount)
                    randomState.shuffle(select)
                    selectMat[t] = select[:obsCount]
                    evidMat[select[:obsCount],t] = True
                    ypred = gp.predict(testset[:,t],evidMat[:,t])
                    predResults[trial,:,t] = ypred
                    errResults[trial,t,0] = gp.compute_mean_absolute_error(testset[:,t], ypred,
                                        type_=0, evidence_mat=evidMat[:,t])
                    errResults[trial,t,1] = gp.compute_mean_squared_error(testset[:,t], ypred,
                                        type_=0,evidence_mat=evidMat[:,t])
                    errResults[trial,t,2] = gp.compute_mean_absolute_error(testset[:,t], ypred,
                                        type_=1, evidence_mat=evidMat[:,t])
                    errResults[trial,t,3] = gp.compute_mean_squared_error(testset[:,t], ypred,
                                        type_=1,evidence_mat=evidMat[:,t])
                    errResults[trial,t,4] = gp.compute_mean_absolute_error(testset[:,t], ypred,
                                        type_=2, evidence_mat=evidMat[:,t])
                    errResults[trial,t,5] = gp.compute_mean_squared_error(testset[:,t], ypred,
                                        type_=2,evidence_mat=evidMat[:,t])
                np.savetxt(evidencepath +
                           'evidMat_trial={}_obsrate={}.csv'.format(trial,obsrate),
                           evidMat, delimiter=',')
                np.savetxt(predictionpath +
                    'predResults_activeInf_gaussianProcess_T={}_obsRate={}_{}_trial={}.csv'.
                    format(T,obsrate, utils.properties.timeStamp,trial),
                    predResults[trial], delimiter=',')
                np.savetxt(errorpath +
                    'result_activeInf_gaussianProcess_T={}_obsRate={}_{}_trial={}.csv'.
                    format(T,obsrate, utils.properties.timeStamp,trial),
                    errResults[trial], delimiter=',')
            np.savetxt(errorpath +
                   'meanMAE_activeInf_gaussianProcess_T={}_obsRate={}_trial={}.csv'.
                   format(T,obsrate, 'mean'),
                   np.mean(errResults,axis=0), delimiter=',')
            np.savetxt(errorpath +
                   'stderrMAE_activeInf_gaussianProcess_T={}_obsRate={}_trial={}.csv'.
                   format(T,obsrate, 'mean'),
                   standard_error(errResults,axis=0), delimiter=',')
        print 'End of process, duration: {} secs'.format(time() - start)