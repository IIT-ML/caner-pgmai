'''
Created on May 12, 2015

@author: ckomurlu
'''
from models.ml_reg_model import MLRegModel
from utils.readdata import convert_time_window_df_randomvar_hour
from utils.node import Neighborhood

import numpy as np
from scipy.stats import norm

class GaussianDBN(MLRegModel):
    '''
    classdocs
    '''


    def __init__(self):
        '''
        Constructor
        '''
        self.sortedids = list()
    
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
    
    def __computeCondGauss(self, sensid, parentDict, mea, cova, initial=False):
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
                    currentmean = b0 + np.dot(b,parentValues)
                if evidMap[currentid,0]:
                    sampleStorage[currentSample,currentid,0] = \
                        testset[currentid,0].true_label
                    rv = norm(loc=currentmean,scale=var**.5)
                    if parents != []:
                        currentCoef = rv.pdf(sampleStorage[currentSample,currentid,0])
#                         print currentCoef
                        sampleWeight *= currentCoef
                        logSampleWeight += np.log(currentCoef)
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
                    currentmean = b0 + np.dot(b,parentValues)
                    sampleStorage[currentSample,currentid,t] = \
                        np.random.normal(currentmean,var**.5)
                    if evidMap[currentid,t]:
                        sampleStorage[currentSample,currentid,t] = \
                            testset[currentid,t].true_label
                        rv = norm(loc=currentmean,scale=var**.5)
                        currentCoef = rv.pdf(sampleStorage[currentSample,currentid,t])
#                         print currentCoef
                        sampleWeight *= currentCoef
                        logSampleWeight += np.log(currentCoef)
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
        raise NotImplementedError
        if evidence_mat is None:
            evidence_mat = np.zeros(testset.shape,dtype=np.bool8)
        Y_pred = np.empty(shape=testset.shape,dtype=np.float64)
        for currentid in self.sortedids:
            (b0,b,var) = self.cpdParams[currentid,0]
            parents = self.parentDict[currentid][:-1]
            if parents == []:
                currentmean = b0
            else:
                parentValues = Y_pred[parents,0]
                currentmean = b0 + np.dot(b,parentValues)
            Y_pred[currentid,0] = currentmean
        for t in xrange(1,testset.shape[1]):
            for currentid in self.sortedids:
                (b0,b,var) = self.cpdParams[currentid,1]
                parents = self.parentDict[currentid]
                parentValuesT = Y_pred[parents[:-1],t] 
                itselfPrevious = Y_pred[currentid,t-1]
                parentValues = np.append(parentValuesT, itselfPrevious)
                currentmean = b0 + np.dot(b,parentValues)
                Y_pred[currentid,t] = currentmean
        return Y_pred
                
    def compute_accuracy(self, Y_test, Y_pred):
        raise NotImplementedError('Method compute_accuracy() is not yet implemented.')
    
    def compute_confusion_matrix(self, Y_test, Y_pred):
        raise NotImplementedError('Method compute_confusion_matrix() is not yet implemented.')
    
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

test()
