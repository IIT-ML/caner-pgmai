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
from utils.readdata import convert_time_window_df_randomvar, \
                            convert_time_window_df_randomvar_hour

from sklearn.preprocessing import PolynomialFeatures
from independent.local_mean_regressor import LocalMeanRegressor
from scipy.stats import norm


def computeCondGauss(sensid, parentDict, mea, cova, initial=False):
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

def likelihood_weighting():
    trainset,testset = convert_time_window_df_randomvar_hour(True,
                            Neighborhood.all_others_current_time)
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
    sortedids = np.argsort(concount)[::-1].astype(np.int8).tolist()
    arrowlist = list()
    for id_ in sortedids:
        for nei in condict[id_]:
            if sortedids.index(id_) < sortedids.index(nei):
                arrowlist.append((id_,nei))
    cpdParams = np.empty(shape=mea.shape + (2,), dtype=tuple)
    parentDict = convertArrowsToGraphInv(arrowlist)
    for currentid in sortedids:
        try:
            parentDict[currentid].append(currentid + 50)
        except KeyError:
            parentDict[currentid] = list()
            parentDict[currentid].append(currentid + 50)
    for i in sortedids:
        cpdParams[i,0] = computeCondGauss(i,parentDict,mea100,cova100,initial=True)
        cpdParams[i,1] = computeCondGauss(i,parentDict,mea100,cova100)
    
    
    sampleSize = 30
    numSlice = 15
    evidMap = np.zeros((len(sortedids),numSlice),dtype=np.bool_)
#     evidMap[24,0] = 21
#     evidMap[24,2] = 21
#     evidMap[12,4] = 21
#     evidMap[12,6] = 21
    evidMap[15,1] = 21
    evidMap[15,3] = 21
    sampleStorage = np.empty(shape = (sampleSize,len(sortedids),numSlice),
                                 dtype=np.float64)
    weightList = list()
    for currentSample in xrange(sampleSize):
    #     print currentSample
        sampleWeight = 1
        for currentid in sortedids:
            (b0,b,var) = cpdParams[currentid,0]
            parents = parentDict[currentid][:-1]
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
                    print currentCoef
                    sampleWeight *= currentCoef
            else:            
                sampleStorage[currentSample,currentid,0] = \
                    np.random.normal(currentmean,var**.5)
        for t in xrange(1,numSlice):
    #         print '\t',t
            for currentid in sortedids:
    #             print '\t\t',currentid
                parents = parentDict[currentid]
                (b0,b,var) = cpdParams[currentid,1]
                parentValuesT = sampleStorage[currentSample,parents[:-1],t]
                itselfPrevious = sampleStorage[currentSample,parents[-1]-50,t-1]
                parentValues = np.append(parentValuesT, itselfPrevious)
                currentmean = b0 + np.dot(b,parentValues)
                sampleStorage[currentSample,currentid,t] = \
                    np.random.normal(currentmean,var**.5)
                if evidMap[currentid,t]:
                    sampleStorage[currentSample,currentid,t] = \
                        testset[currentid,t].true_label
                    rv = norm(loc=currentmean,scale=var**.5)
                    currentCoef = rv.pdf(sampleStorage[currentSample,currentid,t])
                    print currentCoef
                    sampleWeight *= currentCoef
                else:            
                    sampleStorage[currentSample,currentid,t] = \
                        np.random.normal(currentmean,var**.5)
        weightList.append(sampleWeight)
    print weightList

def convertArrowsToGraphInv(arrowlist):
    graph = dict()
    for edge in arrowlist:
        if edge[1] not in graph:
            graph[edge[1]] = list()
        graph[edge[1]].append(edge[0])
    return graph

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
    
# test_local_mean_fit()
start = time()
likelihood_weighting()
print time() - start