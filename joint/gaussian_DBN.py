'''
Created on May 12, 2015

@author: ckomurlu
'''
from models.ml_reg_model import MLRegModel
from utils.readdata import convert_time_window_df_randomvar_hour
from utils.node import Neighborhood
from utils.metropolis_hastings import MetropolisHastings

import numpy as np
from scipy.stats import norm
from toposort import toposort, toposort_flatten
from time import time
import cPickle
from utils import readdata

class GaussianDBN(MLRegModel):
    '''
    classdocs
    '''


    def __init__(self):
        '''
        Constructor
        '''
        self.sortedids = list()
        self.cpdParams = object()
        self.parentDict = dict()
    
    def fit(self,trainset):
        X = np.vectorize(lambda x: x.true_label)(trainset)
        mea = np.mean(X,axis=1)
        mea100 = np.append(mea,mea)
        cova = np.cov(X[:,:])
        Y = np.append(X[:,1:],X[:,:-1],axis=0)
        cova100 = np.cov(Y)
        infomat = np.linalg.inv(cova)
        absround = np.absolute(np.around(infomat,6))
        (a,b) = np.nonzero(absround > 24.3)
        condict = dict()
        for i in range(a.shape[0]):
            if a[i] not in condict:
                condict[a[i]] = list()
            condict[a[i]].append(b[i])
        concount = np.empty((len(condict),),dtype=np.int_)
        for key in condict:
            if key in condict[key]:
                condict[key].remove(key)
            concount[key] = len(condict[key])
        self.sortedids = np.argsort(concount)[::-1].astype(np.int8).tolist()
        arclist = list()
        for id_ in self.sortedids:
            for nei in condict[id_]:
                if self.sortedids.index(id_) < self.sortedids.index(nei):
                    arclist.append((id_,nei))
        self.cpdParams = np.empty(shape=mea.shape + (2,), dtype=tuple)
        self.parentDict = self.__convertArcsToGraphInv(arclist)
        self.checkParentChildOrder()
        for currentid in self.sortedids:
            try:
                self.parentDict[currentid].append(currentid + 50)
            except KeyError:
                self.parentDict[currentid] = list()
                self.parentDict[currentid].append(currentid + 50)
        for i in self.sortedids:
            self.cpdParams[i,0] = self.__computeCondGauss(i,self.parentDict,mea100,
                                                        cova100,initial=True)
            self.cpdParams[i,1] = self.__computeCondGauss(i,self.parentDict,
                                                        mea100,cova100)
    
        
    def __convertArcsToGraphInv(self,arclist):
        graph = dict()
        for edge in arclist:
            if edge[1] not in graph:
                graph[edge[1]] = list()
            graph[edge[1]].append(edge[0])
        return graph
    
    def __computeCondGauss_bkp(self, sensid, parentDict, mea, cova, initial=False):
#     try:
#         parents = parentDict[sensid]
#     except KeyError:
#         return mea[sensid],0,cova[sensid,sensid]
        parents = parentDict[sensid]
        if initial == True:
            parents = parents[:-1]
            if parents == []:
                return mea[sensid],0,cova[sensid,sensid]
        firstInd = np.tile(tuple(parents),len(parents))
        secondInd = np.repeat(tuple(parents),len(parents))
        YY = cova[sensid,sensid]
        YX = cova[sensid,tuple(parents)]
        XY = cova[tuple(parents),sensid]
        XXinv = np.linalg.inv(cova[firstInd,secondInd].reshape(len(parents),
                                                               len(parents)))
        b0 = mea[sensid] - np.dot(np.dot(YX, XXinv), mea[list(parents)])
        b = np.dot(XXinv,YX)
        sigsq = YY - np.dot(np.dot(YX,XXinv),XY.reshape(-1,1))
        return b0,b,sigsq[0]
    
    def __computeCondGauss(self, sensid, parentDict, mea, cova, initial=False):
#     try:
#         parents = parentDict[sensid]
#     except KeyError:
#         return mea[sensid],0,cova[sensid,sensid]
        parents = parentDict[sensid]
        if initial == True:
            parents = parents[:-1]
            if parents == []:
                return mea[sensid],np.array([]),cova[sensid,sensid]
        firstInd = np.tile(tuple(parents),len(parents))
        secondInd = np.repeat(tuple(parents),len(parents))
        YY = cova[sensid,sensid].reshape(1,1)
        YX = cova[sensid,tuple(parents)].reshape(1,-1)
        XY = cova[tuple(parents),sensid].reshape(-1,1)
        XXinv = np.linalg.inv(cova[firstInd,secondInd].reshape(len(parents),
                                                               len(parents)))
        b0 = mea[sensid].reshape(1,1) - np.dot(np.dot(YX, XXinv),
                                            mea[list(parents)].reshape(-1,1))
        b = np.dot(XXinv,XY)
        sigsq = YY - np.dot(np.dot(YX,XXinv),XY.reshape(-1,1))
        return b0,b,sigsq[0]
    
    def likelihoodWeighting(self, evidMap, testset, sampleSize=100):
        if not evidMap.shape[0] == len(self.sortedids):
            raise ValueError('Evidence map doesn\'t have appropriate dimension for sensors')
        numSlice = evidMap.shape[1]
        sampleStorage = np.empty(shape = (sampleSize,len(self.sortedids),numSlice),
                                 dtype=np.float64)
        weightList = list()
        logWeightList = list()
        for currentSample in xrange(sampleSize):
#             print currentSample,
            sampleWeight = 1
            logSampleWeight = 0
            for currentid in self.sortedids:
                (b0,b,var) = self.cpdParams[currentid,0]
                parents = self.parentDict[currentid][:-1]
                if parents == []:
                    currentmean = b0
                else:
                    parentValues = sampleStorage[currentSample,parents,0]
                    currentmean = b0 + np.dot(b.T,parentValues.reshape(-1,1))
                if evidMap[currentid,0]:
                    sampleStorage[currentSample,currentid,0] = \
                        testset[currentid,0].true_label
                    rv = norm(loc=currentmean,scale=var**.5)
                    if parents != []:
                        currentCoef = rv.pdf(sampleStorage[currentSample,currentid,0])
#                         print currentCoef
                        sampleWeight *= currentCoef.reshape(-1)
                        logSampleWeight += np.log(currentCoef.reshape(-1))
                else:            
                    sampleStorage[currentSample,currentid,0] = \
                        np.random.normal(currentmean,var**.5)
            for t in xrange(1,numSlice):
        #         print '\t',t
                for currentid in self.sortedids:
        #             print '\t\t',currentid
                    parents = self.parentDict[currentid]
                    (b0,b,var) = self.cpdParams[currentid,1]
                    parentValuesT = sampleStorage[currentSample,parents[:-1],t]
                    itselfPrevious = sampleStorage[currentSample,currentid,t-1]
                    parentValues = np.append(parentValuesT, itselfPrevious)
                    currentmean = b0 + np.dot(b.T,parentValues.reshape(-1,1))
                    if evidMap[currentid,t]:
                        sampleStorage[currentSample,currentid,t] = \
                            testset[currentid,t].true_label
                        rv = norm(loc=currentmean,scale=var**.5)
                        currentCoef = rv.pdf(sampleStorage[currentSample,currentid,t])
#                         print currentCoef
                        sampleWeight *= currentCoef.reshape(-1)
                        logSampleWeight += np.log(currentCoef.reshape(-1))
                    else:
                        sampleStorage[currentSample,currentid,t] = \
                            np.random.normal(currentmean,var**.5)
#             print np.exp(sampleWeight)
            weightList.append(sampleWeight)
            logWeightList.append(np.exp(logSampleWeight))
#             print sampleWeight
#         print weightList
        return (sampleStorage, weightList, logWeightList)
    
    def predict(self, testset, evidence_mat=None):
        Y_true = np.vectorize(lambda x: x.true_label)(testset)
        if evidence_mat is None:
            evidence_mat = np.zeros(testset.shape,dtype=np.bool8)
        else:
            assert(testset.shape == evidence_mat.shape)
        Y_pred = np.empty(shape=testset.shape,dtype=np.float64)
        for currentid in self.sortedids:
            (b0,b,var) = self.cpdParams[currentid,0]
            parents = self.parentDict[currentid][:-1]
            if parents == []:
                currentmean = b0
            else:
                parentValues = Y_pred[:,0]
                evidCurTime = evidence_mat[:,0]
                parentValues[parents][evidCurTime[parents]] = Y_true[parents,0][evidCurTime[parents]]
                parentValues = parentValues[parents]
                currentmean = b0 + np.dot(b,parentValues)
            Y_pred[currentid,0] = currentmean
        for t in xrange(1,testset.shape[1]):
            for currentid in self.sortedids:
                (b0,b,var) = self.cpdParams[currentid,1]
                parents = self.parentDict[currentid][:-1]
                parentValuesT = Y_pred[:,t]
                evidCurTime = evidence_mat[:,t]
                parentValuesT[parents][evidCurTime[parents]] = Y_true[parents,t][evidCurTime[parents]]
                if evidence_mat[currentid,t-1]:
                    itselfPrevious = Y_true[currentid,t-1]
                else:
                    itselfPrevious = Y_pred[currentid,t-1]
                parentValuesT = parentValuesT[parents]
                parentValues = np.append(parentValuesT, itselfPrevious)
                currentmean = b0 + np.dot(b,parentValues)
                Y_pred[currentid,t] = currentmean
        return Y_pred
    
    def predictTrial(self,testset, evidence_mat=None):
        Y_true = np.vectorize(lambda x: x.true_label)(testset)
        if evidence_mat is None:
            evidence_mat = np.zeros(testset.shape,dtype=np.bool8)
        else:
            assert(testset.shape == evidence_mat.shape)
        Y_pred = np.empty(shape=testset.shape,dtype=np.float64)
        (sampleStorage,weightList, logWeightList) = \
                self.likelihoodWeighting(evidence_mat, testset, sampleSize = 10)
        for i in xrange(Y_pred.shape[0]):
            for j in xrange(Y_pred.shape[1]):
                try:
                    Y_pred[i,j] = np.dot(sampleStorage[:,i,j],logWeightList)/np.sum(logWeightList)
                except ValueError as valueError:
                    print valueError
        return Y_pred
        
    def compute_accuracy(self, Y_test, Y_pred):
        raise NotImplementedError('Method compute_accuracy() is not yet implemented.')
    
    def compute_confusion_matrix(self, Y_test, Y_pred):
        raise NotImplementedError('Method compute_confusion_matrix() is not yet implemented.')

    def checkParentChildOrder(self):
        for i in xrange(len(self.sortedids)):
            curid = self.sortedids[i]
            try:
                parentset = set(self.parentDict[curid])
            except KeyError:
                continue
            if not parentset.issubset(self.sortedids[:i]):
                raise ValueError('parents are not in former positions for '+
                                 str(curid))
    



def test():
    np.random.seed(42)
    trainset,testset = convert_time_window_df_randomvar_hour(True,
                            Neighborhood.all_others_current_time)
    gdbn = GaussianDBN()
    gdbn.fit(trainset)
    numSlice = 6
    result = np.empty(shape=(100,12),dtype=np.float64)
    for i in range(6):
        evidMap = np.zeros((len(gdbn.sortedids),numSlice),dtype=np.bool_)
        evidMap[1,i] = True
    #     evidMap[15,3] = True
        (sampleStorage,weightList, logWeightList) = gdbn.likelihoodWeighting(evidMap, trainset, sampleSize = 100)
        result[:,2*i] = weightList
        result[:,2*i+1] = logWeightList
    for row in result:
        for col in row:
            print col,
        print
    return

def testPredictTrial():
    trainset,testset = convert_time_window_df_randomvar_hour(True,
                            Neighborhood.all_others_current_time)
    gdbn = GaussianDBN()
    gdbn.fit(trainset)
    Y_pred = gdbn.predict_trial(testset)
    return

def testPredict():
    np.random.seed(17)
    trainset,testset = convert_time_window_df_randomvar_hour(True,
                            Neighborhood.all_others_current_time)
    gdbn = GaussianDBN()
    gdbn.fit(trainset)
    (sensorCount,testTimeCount) = testset.shape
    numTrial = 10
#     rateList = np.arange(0.0,1.1,0.1)
    rateList = [0.001, 0.01]
    resultMat = np.empty((numTrial,len(rateList),6))
    for currentTrial in xrange(numTrial):
        rateInd = 0
        for evidRate in rateList:
            evidMat = np.random.rand(sensorCount,testTimeCount) < evidRate
#             Y_pred = gdbn.predict(testset,evidMat)
            Y_pred = gdbn.predictTrial(testset,evidMat)
#             Y_true = np.vectorize(lambda x: x.true_label)(testset)
            resultMat[currentTrial,rateInd,0] = gdbn.compute_mean_squared_error(test_set=testset,
                                                                                 Y_pred=Y_pred, type_=0, evidence_mat=evidMat)
            resultMat[currentTrial,rateInd,1] = gdbn.compute_mean_squared_error(test_set=testset,
                                                                                 Y_pred=Y_pred, type_=1, evidence_mat=evidMat)
            resultMat[currentTrial,rateInd,2] = gdbn.compute_mean_squared_error(test_set=testset,
                                                                                 Y_pred=Y_pred, type_=2, evidence_mat=evidMat)
            resultMat[currentTrial,rateInd,3] = gdbn.compute_mean_absolute_error(test_set=testset,
                                                                                 Y_pred=Y_pred, type_=0, evidence_mat=evidMat)
            resultMat[currentTrial,rateInd,4] = gdbn.compute_mean_absolute_error(test_set=testset,
                                                                                 Y_pred=Y_pred, type_=1, evidence_mat=evidMat)
            resultMat[currentTrial,rateInd,5] = gdbn.compute_mean_absolute_error(test_set=testset,
                                                                                 Y_pred=Y_pred, type_=2, evidence_mat=evidMat)
            rateInd += 1
    np.savetxt('C:/Users/ckomurlu/Documents/workbench/experiments/20150528/GDBN_RandomSamplingWrtRates.txt',
               np.mean(resultMat,axis=0),delimiter=',',fmt='%.4f')

def stupidTest():
    np.random.seed(17)
    trainset,testset = convert_time_window_df_randomvar_hour(True,
                            Neighborhood.all_others_current_time)
    gdbn = GaussianDBN()
    gdbn.fit(trainset)
    evidMat = np.zeros((50,10),dtype=np.bool_)
    (sampleStorage,weightList, logWeightList) = \
                gdbn.likelihoodWeighting(evidMat, testset, sampleSize = 10)
    numSlice = 10
    muMat = np.empty((50,10),dtype=np.float64)
    varMat = np.empty((50,10),dtype=np.float64)
    probDenMat = np.empty((50,10),dtype=np.float64)
    for id_ in gdbn.sortedids:
        parents = gdbn.parentDict[id_][:-1]
        cpdParams = gdbn.cpdParams[id_,0]
        try:
            mu = cpdParams[0] + np.dot(cpdParams[1].T,sampleStorage[0,parents,0])
        except AttributeError:
            mu = cpdParams[0]
        var = cpdParams[2]
        rv = norm(loc=mu,scale=var**.5)
        muMat[id_,0] = mu
        varMat[id_,0] = var
        probDenMat[id_,0] = rv.pdf(testset[id_,0].true_label)
    for t in xrange(1,numSlice):
        for id_ in gdbn.sortedids:
            parentsT = gdbn.parentDict[id_][:-1]
            parentsVals = np.append(sampleStorage[0,parentsT,t],
                                    sampleStorage[0,id_,t-1])
            cpdParams = gdbn.cpdParams[id_,1]
            mu = cpdParams[0] + np.dot(cpdParams[1].T,parentsVals)
            var = cpdParams[2]
            rv = norm(loc=mu,scale=var**.5)
            muMat[id_,t] = mu
            varMat[id_,t] = var
            probDenMat[id_,t] = rv.pdf(testset[id_,0].true_label)

def testMH():
    start = time()
    trainset,testset = convert_time_window_df_randomvar_hour(True,
                            Neighborhood.itself_previous_others_current)
#                             Neighborhood.all_others_current_time)
    gdbn = GaussianDBN()
    gdbn.fit(trainset)
    parentDict = gdbn.parentDict
    rvids = np.array(parentDict.keys())
    rvCount = int(rvids.shape[0])
    T = 10
    evidMat = np.zeros((rvCount,T),dtype=np.bool_)
    evidMat[2,0] = True
    testMat = np.zeros((rvCount,T),dtype=np.float_)
    testMat[2,0] = 25.0
    
    
    sampleSize = 20
    burnInCount = 10
    samplingPeriod = 2
    width = 0.3
    
    startupVals = np.ones((rvCount,T),dtype=np.float_)*20
    
    metropolisHastings = MetropolisHastings()
    (data,accList,propVals,accCount) = metropolisHastings.sampleTemporal(gdbn.sortedids,
                                            parentDict, gdbn.cpdParams, startupVals, evidMat,
                                            testMat, sampleSize=sampleSize,
                                            burnInCount=burnInCount, samplingPeriod=samplingPeriod,
                                            proposalDist='uniform', width=width)
    cPickle.dump((data,accList,propVals,accCount), open(readdata.DATA_DIR_PATH + 'mhResultsEvidDnm8.pkl','wb'))
    
    print 'Process Ended in ', time() - start
    
testMH()

# stupidTest()
# testPredict()


########################
####### Garbage ########
########################

#     def checkTopoSort(self):
#         parentSet = dict()
#         for i in self.parentDict:
#             parentSet[i] = set(self.parentDict[i])
#         toposorted = list(toposort(parentSet))
#         followIndex = 0
#         for item in self.sortedids:
#             for i in xrange(followIndex,len(toposorted)):
#                 if item in toposorted[i]:
#                     followIndex = i
#                     break
#             else:
#                 raise ValueError('sortedids list is not topologically sorted.'+str(item)+
#                                  ' could not be found in the remaining topological sets' +
#                                  ' following index:', followIndex)
# 
#     @staticmethod
#     def checkTopoSortA(sortedids,parentDict):
#         toposorted = list(toposort(parentDict))
#         followIndex = 0
#         for item in sortedids:
#             for i in xrange(followIndex,len(toposorted)):
#                 if item in toposorted[i]:
#                     followIndex = i
#                     break
#             else:
#                 raise ValueError('sortedids list is not topologically sorted.'+str(item)+
#                                  ' could not be found in the remaining topological sets' +
#                                  ' following index:', followIndex)
# 
# def testCheckTopoSort():
# #     ts = list(toposort({2: {11},
# #                 9: {11, 8, 10},
# #                 10: {11, 3},
# #                 11: {7, 5},
# #                 8: {7, 3},
# #                }))
#     parentDict = {2: {11},
#                 9: {11, 8, 10},
#                 10: {11, 3},
#                 11: {7, 5},
#                 8: {7, 3},
#                }
#     tsf = toposort_flatten({2: {11},
#                 9: {11, 8, 10},
#                 10: {11, 3},
#                 11: {7, 5},
#                 8: {7, 3},
#                })
#     
#     GaussianDBN.checkTopoSortA(tsf[::-1], parentDict)