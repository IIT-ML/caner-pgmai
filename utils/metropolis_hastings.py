'''
Created on Jul 1, 2015

@author: ckomurlu
'''
import numpy as np
from scipy.stats import norm
from sklearn.metrics.metrics import mean_squared_error

class MetropolisHastings:
    def __init__(self):
        self.rs=np.random.RandomState(0)
        
    def getChildDict(self, parentDict):
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
    
    def proposeFromUniform(self, center, width):
        return self.rs.uniform(center-width, center+width)
    
    def proposeFromNormal(self,loc=0,scale=1):
        return self.rs.normal(loc,scale)
    
    def acceptInitial_backup(self, x_new, rvid, parentDict, childDict, currentVals, cpdParams):
        parents = [parentid for parentid in parentDict[rvid] if parentid<self.rvCount]
        parentValues = currentVals[parents,0]
        loc = cpdParams[rvid,0][0] + np.dot(cpdParams[rvid,0][1].T,parentValues)
        distVal = norm.pdf(x_new,loc=loc,scale=cpdParams[rvid,0][2]**.5)
        x_curr = currentVals[rvid,0]
#         loc = cpdParams[rvid,0][0] + np.dot(cpdParams[rvid,0][1].T,parentValues)
        distVal2 = norm.pdf(x_curr,loc=loc,scale=cpdParams[rvid,0][2]**.5)
        for child in childDict[rvid]:
            if child < self.rvCount:
                parents = [parentid for parentid in parentDict[child] if parentid<self.rvCount]
                parentValues = currentVals[parents,0]
                relativeInd = parents.index(rvid)
                parentValues[relativeInd] = x_new
                loc = cpdParams[child,0][0] + np.dot(cpdParams[child,0][1].T,parentValues)
                distVal *= norm.pdf(currentVals[child,0],loc=loc,scale=cpdParams[child,0][2]**.5)
                parentValues[relativeInd] = x_curr
                loc = cpdParams[child,0][0] + np.dot(cpdParams[child,0][1].T,parentValues)
                distVal2 *= norm.pdf(currentVals[child,0],loc=loc,scale=cpdParams[child,0][2]**.5)
            elif len(currentVals.shape) > 1 and currentVals.shape[1] > 1:
                child = child - self.rvCount
                parentValues = self.getParentValsTemporal(rvid,self.rvCount,parentDict,currentVals,1)
                parents = parentDict[child]
                relativeInd = parents.index(rvid + self.rvCount)
                parentValues[relativeInd] = x_new
                loc = cpdParams[child,1][0] + np.dot(cpdParams[child,1][1].T,parentValues)
                distVal *= norm.pdf(currentVals[child,1],loc=loc,scale=cpdParams[child,1][2]**.5)
                parentValues[relativeInd] = x_curr
                loc = cpdParams[child,1][0] + np.dot(cpdParams[child,1][1].T,parentValues)
                distVal2 *= norm.pdf(currentVals[child,1],loc=loc,scale=cpdParams[child,1][2]**.5)
        return distVal/distVal2
    
    def acceptInitial(self, x_new, rvid, parentDict, childDict, currentVals, cpdParams):
        parents = [parentid for parentid in parentDict[rvid] if parentid<self.rvCount]
        parentValues = currentVals[parents,0]
        loc = cpdParams[rvid,0][0] + np.dot(cpdParams[rvid,0][1].T,parentValues)
        currentProb = norm.pdf(x_new,loc=loc,scale=cpdParams[rvid,0][2]**.5)
        self.localPrimeProbs[rvid,0] = currentProb
        distVal = currentProb
#         x_curr = currentVals[rvid,0]
#         loc = cpdParams[rvid,0][0] + np.dot(cpdParams[rvid,0][1].T,parentValues)
#         distVal2 = norm.pdf(x_curr,loc=loc,scale=cpdParams[rvid,0][2]**.5)
        distVal2 = self.localProbs[rvid,0]
        for child in childDict[rvid]:
            if child < self.rvCount:
                parents = [parentid for parentid in parentDict[child] if parentid<self.rvCount]
                parentValues = currentVals[parents,0]
                relativeInd = parents.index(rvid)
                parentValues[relativeInd] = x_new
                loc = cpdParams[child,0][0] + np.dot(cpdParams[child,0][1].T,parentValues)
                currentProb = norm.pdf(currentVals[child,0],loc=loc,scale=cpdParams[child,0][2]**.5)
                self.localPrimeProbs[child,0] = currentProb
                distVal *= currentProb
#                 parentValues[relativeInd] = x_curr
#                 loc = cpdParams[child,0][0] + np.dot(cpdParams[child,0][1].T,parentValues)
#                 distVal2 *= norm.pdf(currentVals[child,0],loc=loc,scale=cpdParams[child,0][2]**.5)
                distVal2 *= self.localProbs[child,0]
            elif len(currentVals.shape) > 1 and currentVals.shape[1] > 1:
                child = child - self.rvCount
                parentValues = self.getParentValsTemporal(rvid,self.rvCount,parentDict,currentVals,1)
                parents = parentDict[child]
                relativeInd = parents.index(rvid + self.rvCount)
                parentValues[relativeInd] = x_new
                loc = cpdParams[child,1][0] + np.dot(cpdParams[child,1][1].T,parentValues)
                currentProb = norm.pdf(currentVals[child,1],loc=loc,scale=cpdParams[child,1][2]**.5)
                self.localPrimeProbs[child,1] = currentProb
                distVal *= currentProb
#                 parentValues[relativeInd] = x_curr
#                 loc = cpdParams[child,1][0] + np.dot(cpdParams[child,1][1].T,parentValues)
#                 distVal2 *= norm.pdf(currentVals[child,1],loc=loc,scale=cpdParams[child,1][2]**.5)
                distVal2 *= self.localProbs[child,1]
        return distVal/distVal2
    
    def getParentValsTemporal(self, rvid, rvCount, parentDict, currentVals, t):
        parents = parentDict[rvid]
        parentsNow = [parentid for parentid in parents if parentid<rvCount]
        parentsPrev = [parentid - rvCount for parentid in parents if parentid>=rvCount]
        parentNowValues = currentVals[parentsNow,t]
        parentPrevValues = currentVals[parentsPrev,t-1]
        parentValues = np.append(parentNowValues, parentPrevValues)
        return parentValues
    
    def acceptTemporal_backup(self, x_new, rvid, parentDict, childDict, currentVals, t,
                       cpdParams):
        
        parentValues = self.getParentValsTemporal(rvid,self.rvCount,parentDict,currentVals,t)
        loc = cpdParams[rvid,1][0] + np.dot(cpdParams[rvid,1][1].T,parentValues)
        distVal = norm.pdf(x_new,loc=loc,scale=cpdParams[rvid,1][2]**.5)
        x_curr = currentVals[rvid,t]
#         loc = cpdParams[rvid,1][0] + np.dot(cpdParams[rvid,1][1].T,parentValues)
        distVal2 = norm.pdf(x_curr,loc=loc,scale=cpdParams[rvid,1][2]**.5)
        for child in childDict[rvid]:
            if child < self.rvCount:
                parents = parentDict[child]
                parentValues = self.getParentValsTemporal(child,self.rvCount,parentDict,currentVals,t)
                relativeInd = parents.index(rvid)
                parentValues[relativeInd] = x_new
                loc = cpdParams[child,1][0] + np.dot(cpdParams[child,1][1].T,parentValues)
                distVal *= norm.pdf(currentVals[child,t],loc=loc,scale=cpdParams[child,1][2]**.5)
                parentValues[relativeInd] = x_curr
                loc = cpdParams[child,1][0] + np.dot(cpdParams[child,1][1].T,parentValues)
                distVal2 *= norm.pdf(currentVals[child,t],loc=loc,scale=cpdParams[child,1][2]**.5)
            elif t < self.T - 1:
                child = child - self.rvCount
                parents = parentDict[child]
                parentValues = self.getParentValsTemporal(child,self.rvCount,parentDict,currentVals,t+1)
                relativeInd = parents.index(rvid + self.rvCount)
                parentValues[relativeInd] = x_new
                loc = cpdParams[child,1][0] + np.dot(cpdParams[child,1][1].T,parentValues)
                distVal *= norm.pdf(currentVals[child,t+1],loc=loc,scale=cpdParams[child,1][2]**.5)
                parentValues[relativeInd] = x_curr
                loc = cpdParams[child,1][0] + np.dot(cpdParams[child,1][1].T,parentValues)
                distVal2 *= norm.pdf(currentVals[child,t+1],loc=loc,scale=cpdParams[child,1][2]**.5)
        return distVal/distVal2
    
    def acceptTemporal(self, x_new, rvid, parentDict, childDict, currentVals, t,
                       cpdParams):
        parentValues = self.getParentValsTemporal(rvid,self.rvCount,parentDict,currentVals,t)
        loc = cpdParams[rvid,1][0] + np.dot(cpdParams[rvid,1][1].T,parentValues)
        currentProb = norm.pdf(x_new,loc=loc,scale=cpdParams[rvid,1][2]**.5)
        self.localPrimeProbs[rvid,t] = currentProb
        distVal = currentProb
#         x_curr = currentVals[rvid,t]
#         loc = cpdParams[rvid,1][0] + np.dot(cpdParams[rvid,1][1].T,parentValues)
#         distVal2 = norm.pdf(x_curr,loc=loc,scale=cpdParams[rvid,1][2]**.5)
        distVal2 = self.localProbs[rvid,t]
        for child in childDict[rvid]:
            if child < self.rvCount:
                parents = parentDict[child]
                parentValues = self.getParentValsTemporal(child,self.rvCount,parentDict,currentVals,t)
                relativeInd = parents.index(rvid)
                parentValues[relativeInd] = x_new
                loc = cpdParams[child,1][0] + np.dot(cpdParams[child,1][1].T,parentValues)
                currentProb = norm.pdf(currentVals[child,t],loc=loc,scale=cpdParams[child,1][2]**.5)
                self.localPrimeProbs[child,t] = currentProb
                distVal *= currentProb
#                 parentValues[relativeInd] = x_curr
#                 loc = cpdParams[child,1][0] + np.dot(cpdParams[child,1][1].T,parentValues)
#                 distVal2 *= norm.pdf(currentVals[child,t],loc=loc,scale=cpdParams[child,1][2]**.5)
                distVal2 *= self.localProbs[child,t]
            elif t < self.T - 1:
                child = child - self.rvCount
                parents = parentDict[child]
                parentValues = self.getParentValsTemporal(child,self.rvCount,parentDict,currentVals,t+1)
                relativeInd = parents.index(rvid + self.rvCount)
                parentValues[relativeInd] = x_new
                loc = cpdParams[child,1][0] + np.dot(cpdParams[child,1][1].T,parentValues)
                currentProb = norm.pdf(currentVals[child,t+1],loc=loc,scale=cpdParams[child,1][2]**.5)
                self.localPrimeProbs[child,t+1] = currentProb
                distVal *= currentProb
#                 parentValues[relativeInd] = x_curr
#                 loc = cpdParams[child,1][0] + np.dot(cpdParams[child,1][1].T,parentValues)
#                 distVal2 *= norm.pdf(currentVals[child,t+1],loc=loc,scale=cpdParams[child,1][2]**.5)
                distVal2 *= self.localProbs[child,t+1]
        return distVal/distVal2
    
    def sampleTemporal(self, rvids, parentDict, cpdParams, startupVals,
                                           evidMat=None, testMat=None, sampleSize=100000,
                                           burnInCount=1000, samplingPeriod=2, proposalDist='uniform',
                                           width = None):

        tuneWindow = sampleSize/50
        rejRateStar = 0.5
        epsilon = 0.05

        continueWidthSearch = True

        self.rvids = rvids
        if proposalDist=='uniform':
            proposalDist = self.proposeFromUniform
        elif proposalDist=='normal':
            proposalDist = self.proposeFromNormal

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

        self.rvCount = int(startupVals.shape[0])
        self.T = startupVals.shape[1]
        if width is None:
            width = np.ones(shape=(rvids.shape[0],self.T)) * 5.
        elif type(width) is int:
            width = np.ones(shape=(rvids.shape[0],self.T)) * width
        self.localProbs = np.zeros((self.rvCount,self.T),dtype=np.float64)
        self.localPrimeProbs = np.zeros((self.rvCount,self.T),dtype=np.float64)
        childDict = self.getChildDict(parentDict)
        currentVals = startupVals.copy()
        currentVals[evidMat] = testMat[evidMat]
        self.initializeLocalProb(parentDict, cpdParams, currentVals)
        T = evidMat.shape[1]
        data = list()
        accList = np.zeros((sampleSize,rvids.shape[0],T))
        propVals = np.zeros((sampleSize,rvids.shape[0],T))
        stopFlag = evidMat.copy()
        count = 0
        widthList = list()
        widthList.append(width)
        rejRate = np.zeros(shape=(burnInCount/tuneWindow,rvids.shape[0],T))
        accTrack = np.zeros((sampleSize,rvids.shape[0],T), dtype=np.bool_)
        for i in xrange(sampleSize):
            #             print i
            #         print '\t0'
            for rvid in rvids[~evidMat[rvids,0]]:
                newVal = proposalDist(currentVals[rvid,0],width[rvid,0])
                propVals[i,rvid,0] = newVal
                acceptProb = self.acceptInitial(x_new = newVal, rvid=rvid,
                                                parentDict=parentDict, childDict=childDict,
                                                currentVals=currentVals, cpdParams=cpdParams)
                accList[i,rvid,0] = acceptProb
                if acceptProb > self.rs.rand():
                    currentVals[rvid,0] = newVal
                    self.localProbs[rvid,0] = self.localPrimeProbs[rvid,0]
                    accTrack[i,rvid,0] = True
                    for child in childDict[rvid]:
                        if child < self.rvCount:
                            self.localProbs[child,0] = self.localPrimeProbs[child,0]
                        elif T > 1:
                            self.localProbs[child-self.rvCount,1] = self.localPrimeProbs[child-self.rvCount,1]
            for t in range(1,T):
                #             print '\t',t
                for rvid in rvids[~evidMat[rvids,t]]:
                    newVal = proposalDist(currentVals[rvid,t],width[rvid,0])
                    propVals[i,rvid,t] = newVal
                    acceptProb = self.acceptTemporal(x_new = newVal, rvid=rvid,
                                                     parentDict=parentDict, childDict=childDict,
                                                     currentVals=currentVals, t=t, cpdParams=cpdParams)
                    accList[i,rvid,t] = acceptProb
                    if acceptProb > self.rs.rand():
                        currentVals[rvid,t] = newVal
                        self.localProbs[rvid,t] = self.localPrimeProbs[rvid,t]
                        accTrack[i,rvid,t] = True
                        for child in childDict[rvid]:
                            if child < self.rvCount:
                                self.localProbs[child,t] = self.localPrimeProbs[child,t]
                            elif t < T-1:
                                self.localProbs[child-self.rvCount,t+1] = self.localPrimeProbs[child-self.rvCount,t+1]
            if i < burnInCount and continueWidthSearch:
                if count == tuneWindow:
                    rejRate[i/tuneWindow] = 1 - np.sum(accTrack[i-count:i], axis=0)/float(tuneWindow)
                    curWidth = widthList[-1].copy()
                    for t in range(T):
                        for rvid in rvids:
                            if not stopFlag[rvid,t] and \
                                    (rejRate[i/tuneWindow,rvid,t] < rejRateStar - epsilon or
                                             rejRate[i/tuneWindow,rvid,t] > rejRateStar + epsilon):
                                curWidth[rvid,t] = curWidth[rvid,t] / rejRate[i/tuneWindow,rvid,t] * \
                                                   rejRateStar
                            else:
                                stopFlag[rvid,t] = True
                    widthList.append(curWidth)
                    count = 1
                    if np.all(stopFlag):
                        continueWidthSearch = False
                    else:
                        width = curWidth
                else:
                    count += 1
            else:
                if i % samplingPeriod:
                    data.append(currentVals.copy())
        return data, accList, propVals, accTrack
    
    def initializeLocalProb(self,parentDict,cpdParams,startupVals):
        for rvid in self.rvids:
            parents = [parentid for parentid in parentDict[rvid] if parentid<self.rvCount]
            parentValues = startupVals[parents,0]
            loc = cpdParams[rvid,0][0] + np.dot(cpdParams[rvid,0][1].T,parentValues)
            self.localProbs[rvid,0] = norm.pdf(startupVals[rvid,0],loc=loc,scale=cpdParams[rvid,0][2]**.5)
        for t in xrange(1,self.T):
            for rvid in self.rvids:
                parentValues = self.getParentValsTemporal(rvid,self.rvCount,parentDict,startupVals,t)
                loc = cpdParams[rvid,1][0] + np.dot(cpdParams[rvid,1][1].T,parentValues)
                self.localProbs[rvid,t] = norm.pdf(startupVals[rvid,t],loc=loc,scale=cpdParams[rvid,1][2]**.5)
                