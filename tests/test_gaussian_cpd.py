'''
Created on May 13, 2015

@author: ckomurlu
'''
PATH_TO_PGMAI = 'C:/Users/ckomurlu/git/pgmai'
import os
import sys
sys.path.append(os.path.abspath(PATH_TO_PGMAI))
from utils import readdata
from utils.node import Neighborhood
import numpy as np


def forwardSampling(ids, cpdParams, parentDict, sampleSize = 100):    
    sampleStorage = dict()
    for curid in ids:
        sampleStorage[curid] = np.empty(shape = (sampleSize,), dtype=np.float64)
    for currentSample in xrange(sampleSize):
    #     print currentSample
        for currentid in ids:
            try:
                (b0,b,var) = cpdParams[currentid]
                parents = parentDict[currentid]
            except KeyError:
                parents = []
            if parents == []:
                currentmean = b0
            else:
                parentValues = list()
                for parent in parents:
                    parentValues.append(sampleStorage[parent][currentSample])
                parentValues = np.array(parentValues)
                currentmean = b0 + np.dot(b,parentValues)
            sampleStorage[currentid][currentSample] = \
                np.random.normal(currentmean,var**.5)
    return sampleStorage

def convertArcsToGraphInv(arrowlist):
    graph = dict()
    for edge in arrowlist:
        if edge[1] not in graph:
            graph[edge[1]] = list()
        graph[edge[1]].append(edge[0])
    return graph

def computeCondGauss(sensid, parentDict, mea, cova, initial=False):
    try:
        parents = parentDict[sensid]
    except KeyError:
        return mea[sensid],0,cova[sensid,sensid]
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

np.random.seed(10)

trainset,testset = readdata.convert_time_window_df_randomvar_hour(True,
                        Neighborhood.all_others_current_time)
X = np.vectorize(lambda x: x.true_label)(trainset)
mea = np.mean(X,axis=1)
cova = np.cov(X)
Y = np.append(X[:,1:],X[:,:-1],axis=0)
cova100 = np.cov(Y)
mea100 = np.append(mea,mea)
ids = [17, 21, 18, 20]
arclist = list()
for i in xrange(3):
    arclist.append((ids[i],ids[i+1]))
arclist.append((17,20))
arclist.append((21,20))
parentDict = convertArcsToGraphInv(arclist)
cpdParams = dict()
for curid in ids:
    cpdParams[curid] = computeCondGauss(curid,parentDict,mea100,cova100,initial=False)
sampleSto = forwardSampling(ids, cpdParams, parentDict)
for curid in ids:
    print sampleSto[curid].mean(), mea[curid], sampleSto[curid].var(), cova[curid,curid]