'''
Created on Jul 1, 2015

@author: ckomurlu
'''
import numpy as np
from scipy.stats import norm
from sklearn.metrics.metrics import mean_squared_error

rs=np.random.RandomState(0)

def getChildDict(parentDict):
    rvids = np.array(parentDict.keys())
    rvCount = rvids.shape[0]
    childDict = dict()
    for key in rvids:
        childDict[key] = list()
    for key in rvids:
        for val in parentDict[key]:
            if val < rvCount:
                childDict[val].append(key)
            else:
                childDict[val - rvCount].append(key + rvCount)
    return childDict

def proposeFromUniform(center, width):
    return rs.uniform(center-width, center+width)

def proposeFromNormal(loc=0,scale=1):
    return rs.normal(loc,scale)

def acceptInitial(x_new, rvid, parentDict, childDict, currentVals, cpdParams):
    rvCount = int(currentVals.shape[0])
    parents = [parentid for parentid in parentDict[rvid] if parentid<rvCount]
    parentValues = currentVals[parents,0]
    loc = cpdParams[rvid,0][0] + np.dot(cpdParams[rvid,0][1].T,parentValues)
    distVal = norm.pdf(x_new,loc=loc,scale=cpdParams[rvid,0][2]**.5)
    x_curr = currentVals[rvid,0]
#     loc = cpdParams[rvid,0][0] + np.dot(cpdParams[rvid,0][1].T,parentValues)
    distVal2 = norm.pdf(x_curr,loc=loc,scale=cpdParams[rvid,0][2]**.5)
    children = childDict[rvid]
    for child in children:
        if child < rvCount:
            parents = [parentid for parentid in parentDict[child] if parentid<rvCount]
            parentValues = currentVals[parents,0]
            relativeInd = parents.index(rvid)
            parentValues[relativeInd] = x_new
            loc = cpdParams[child,0][0] + np.dot(cpdParams[child,0][1].T,parentValues)
            distVal *= norm.pdf(currentVals[child,0],loc=loc,scale=cpdParams[child,0][2]**.5)
            parentValues[relativeInd] = x_curr
            loc = cpdParams[child,0][0] + np.dot(cpdParams[child,0][1].T,parentValues)
            distVal2 *= norm.pdf(currentVals[child,0],loc=loc,scale=cpdParams[child,0][2]**.5)
        elif len(currentVals.shape) > 1 and currentVals.shape[1] > 1:
            child = child - rvCount
            parentValues = getParentValsTemporal(rvid,rvCount,parentDict,currentVals,1)
            parents = parentDict[child]
            relativeInd = parents.index(rvid + rvCount)
            parentValues[relativeInd] = x_new
            loc = cpdParams[child,1][0] + np.dot(cpdParams[child,1][1].T,parentValues)
            distVal *= norm.pdf(currentVals[child,1],loc=loc,scale=cpdParams[child,1][2]**.5)
            parentValues[relativeInd] = x_curr
            loc = cpdParams[child,1][0] + np.dot(cpdParams[child,1][1].T,parentValues)
            distVal2 *= norm.pdf(currentVals[child,1],loc=loc,scale=cpdParams[child,1][2]**.5)
    return distVal/distVal2

def getParentValsTemporal(rvid,rvCount,parentDict,currentVals,t):
    parents = parentDict[rvid]
    parentsNow = [parentid for parentid in parents if parentid<rvCount]
    parentsPrev = [parentid - rvCount for parentid in parents if parentid>=rvCount]
    parentNowValues = currentVals[parentsNow,t]
    parentPrevValues = currentVals[parentsPrev,t-1]
    parentValues = np.append(parentNowValues, parentPrevValues)
    return parentValues

def acceptTemporal(x_new, rvid, parentDict, childDict, currentVals, t,
                   cpdParams):
    rvCount = int(currentVals.shape[0])
    T = currentVals.shape[1]
    parentValues = getParentValsTemporal(rvid,rvCount,parentDict,currentVals,t)
    loc = cpdParams[rvid,1][0] + np.dot(cpdParams[rvid,1][1].T,parentValues)
    distVal = norm.pdf(x_new,loc=loc,scale=cpdParams[rvid,1][2]**.5)
    x_curr = currentVals[rvid,t]
#     loc = cpdParams[rvid,1][0] + np.dot(cpdParams[rvid,1][1].T,parentValues)
    distVal2 = norm.pdf(x_curr,loc=loc,scale=cpdParams[rvid,1][2]**.5)
    children = childDict[rvid]
    for child in children:
        if child < rvCount:
            parents = parentDict[child]
            parentValues = getParentValsTemporal(child,rvCount,parentDict,currentVals,t)
            relativeInd = parents.index(rvid)
            parentValues[relativeInd] = x_new
            loc = cpdParams[child,1][0] + np.dot(cpdParams[child,1][1].T,parentValues)
            distVal *= norm.pdf(currentVals[child,t],loc=loc,scale=cpdParams[child,1][2]**.5)
            parentValues[relativeInd] = x_curr
            loc = cpdParams[child,1][0] + np.dot(cpdParams[child,1][1].T,parentValues)
            distVal2 *= norm.pdf(currentVals[child,t],loc=loc,scale=cpdParams[child,1][2]**.5)
        elif t < T - 1:
            child = child - rvCount
            parents = parentDict[child]
            parentValues = getParentValsTemporal(child,rvCount,parentDict,currentVals,t+1)
            relativeInd = parents.index(rvid + rvCount)
            parentValues[relativeInd] = x_new
            loc = cpdParams[child,1][0] + np.dot(cpdParams[child,1][1].T,parentValues)
            distVal *= norm.pdf(currentVals[child,t+1],loc=loc,scale=cpdParams[child,1][2]**.5)
            parentValues[relativeInd] = x_curr
            loc = cpdParams[child,1][0] + np.dot(cpdParams[child,1][1].T,parentValues)
            distVal2 *= norm.pdf(currentVals[child,t+1],loc=loc,scale=cpdParams[child,1][2]**.5)
    return distVal/distVal2

def metropolisHastingsSamplingTemporal(rvids, parentDict, cpdParams, startupVals, evidMat=None,
                                       testMat=None, sampleSize=100000, burnInCount=1000,
                                       samplingPeriod=2, proposalDist='uniform', width = 5):
    
    if proposalDist=='uniform':
        proposalDist = proposeFromUniform
    elif proposalDist=='normal':
        proposalDist = proposeFromNormal
        
    if type(rvids) is list:
        rvids = np.array(rvids)
    
    if evidMat is not None:
        if testMat is None:
            raise NameError('Missing parameter: testvec; test vector is not provided.')
        assert(evidMat.dtype==bool)
        assert(evidMat.shape <= testMat.shape)
    else:
        evidMat = np.zeros(shape=rvids.shape,dtype=bool)
    if testMat is None:
        testMat = np.zeros(shape=rvids.shape,dtype=bool)
        
    childDict = getChildDict(parentDict)
    currentVals = startupVals.copy()
    currentVals[evidMat] = testMat[evidMat]
    T = evidMat.shape[1]
    data = list()
    accList = np.zeros((sampleSize,rvids.shape[0],T))
    propVals = np.zeros((sampleSize,rvids.shape[0],T))
    accCount = np.zeros((rvids.shape[0],T))
    for i in xrange(sampleSize):
        print i
#         print '\t0'
        for rvid in rvids[~evidMat[:,0]]:
#             newVal = proposeFromUniform(currentVals[rvid,0],width)
#             newVal = proposeFromNormal(currentVals[rvid,0],width)
            newVal = proposalDist(currentVals[rvid,0],width)
            propVals[i,rvid,0] = newVal
            acceptProb = acceptInitial(x_new = newVal, rvid=rvid,
                                        parentDict=parentDict, childDict=childDict, 
                                        currentVals=currentVals, cpdParams=cpdParams)
            accList[i,rvid,0] = acceptProb
            if acceptProb > rs.rand():
                currentVals[rvid,0] = newVal
                accCount[rvid,0] += 1
        for t in range(1,T):
#             print '\t',t
            for rvid in rvids[~evidMat[:,t]]:
#                 newVal = proposeFromUniform(currentVals[rvid,t],width)
#                 newVal = proposeFromNormal(currentVals[rvid,t],width)
                newVal = proposalDist(currentVals[rvid,t],width)
                propVals[i,rvid,t] = newVal
                acceptProb = acceptTemporal(x_new = newVal, rvid=rvid,
                                            parentDict=parentDict, childDict=childDict, 
                                            currentVals=currentVals, t=t, cpdParams=cpdParams)
                accList[i,rvid,t] = acceptProb
                if acceptProb > rs.rand():
                    currentVals[rvid,t] = newVal
                    accCount[rvid,t] += 1
        if i>burnInCount: 
            if i % samplingPeriod:
                data.append(currentVals.copy())
    return data, accList, propVals, accCount