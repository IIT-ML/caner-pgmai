'''
Created on Jan 8, 2015

@author: ckomurlu
'''

import numpy as np
from collections import deque


class RandomStrategy(object):
    def __init__(self, seed=1):
        self.rgen = np.random.RandomState(seed)
    
    def choices(self, pool, k):
        permuted_pool = self.rgen.permutation(pool)
        permuted_pool = map(tuple, permuted_pool)
        return permuted_pool[:k],permuted_pool[k:]


class RandomStrategy2(object):
    def __init__(self, pool, seed=1):
        self.rgen = np.random.RandomState(seed)
        self.pool = np.array(pool)
    
    def choices(self, k):
        self.rgen.shuffle(self.pool)
        return self.pool[:k]


class SlidingWindow(object):
    def __init__(self, pool, seed=1):
        self.rgen = np.random.RandomState(seed)
        self.pool = np.array(pool)
        self.rgen.shuffle(self.pool)
        self.rotationDeque = deque(self.pool)
    
    def choices(self, k):
        selectees = list(self.rotationDeque)[:k]
        self.rotationDeque.rotate(-k)
        return selectees


class ImpactBased(object):
    def __init__(self, pool, parentDict, childDict, cpdParams, rvCount):
        self.pool = pool
        self.parentDict = parentDict
        self.childDict = childDict
        self.cpdParams = cpdParams
        self.rvCount = rvCount

    def choices(self, countSelectees, t, evidMat):
        selectees = list() #np.empty(shape=(countSelectees,),dtype=np.int_)
        for i in range(countSelectees):
            maxImpact = 0
            maxImapctSensor = -1
            for sensor in self.pool:
                if sensor not in selectees:
                    currentImpact = self.__computeSensorImpact(sensor, t, evidMat)
                    if currentImpact > maxImpact:
                        maxImpact = currentImpact
                        maxImapctSensor = sensor
            selectees.append(maxImapctSensor)
        return selectees

    def __computeSensorImpact(self, sensor, t, evidMat):
        if 0 == t:
            betas = np.abs(self.cpdParams[sensor][0][1])
        else:
            betas = np.abs(self.cpdParams[sensor][1][1])
        impactFactor = 0
        parents = np.array(self.parentDict[sensor])
        for parent in parents:
            if parent < self.rvCount:
                if not evidMat[parent,t]:
                    indexInBetaVec = np.where(parents == parent)
                    impactFactor += betas[indexInBetaVec]
            elif t > 0:
                if not evidMat[parent - self.rvCount,t-1]:
                    indexInBetaVec = np.where(parents == parent)
                    impactFactor += betas[indexInBetaVec]
        children = self.childDict[sensor]
        for child in children:
            parents = np.array(self.parentDict[child])
            if child < self.rvCount:
                if not evidMat[child,t]:
                    if 0 == t:
                        betas = np.abs(self.cpdParams[child][0][1])
                    else:
                        betas = np.abs(self.cpdParams[child][1][1])
                    indexInBetaVec = np.where(parents == sensor)
                    impactFactor += betas[indexInBetaVec]
            #  the next time slice is irrelevant of the current/prediction time slice
            # elif t < T:
            #     if not evidMat[sensorid,t+1]:
            #         betas = cpdParams[sensor][1]
            #         indexInBetaVec = np.where(parents == sensor+rvCount)
            #         impactFactor += betas[indexInBetaVec]
        return impactFactor


class UNCSampling(object):
    '''
    This class performs uncertainty sampling based on the model.
    '''
    
    def choices_old(self, model, X, pool, k):
        y_decision = model.decision_function(X[pool])
        uncerts = np.argsort(np.min(np.absolute(y_decision),axis=1))[:k]
        return pool[uncerts]
    
    def choices(self, model, X, pool, k):
        reordered = list()
        reordered.append(tuple([i[0] for i in pool]))
        reordered.append(tuple([i[1] for i in pool]))
        predicted_probs = model.predict_proba(X)
        uncerts = np.argsort(np.max(predicted_probs[reordered],axis=1))[:k]
        selectees = [pool[x] for x in uncerts]
        reducedpool = list()
        for i in range(len(pool)):
            if i not in uncerts:
                reducedpool.append(pool[i])
        return selectees,reducedpool
    
    def choices_model_per_sensor(self, model_dict, X_dict, pool_dict, k):
#       for implementation purpose only,
        model_dict = dict()
        X_dict = dict()
        pool_dict = dict()
        
        current_sensor_list = model_dict.keys()
        for sensor in current_sensor_list:
            pass
        

def testRandomSampling2():
    pool = np.arange(50)
    sstr = RandomStrategy2(pool,seed=1)
    print sstr.choices(5)
    print sstr.choices(5)
    print sstr.choices(5)
    
# testRandomSampling2()