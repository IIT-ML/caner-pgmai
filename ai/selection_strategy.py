'''
Created on Jan 8, 2015

@author: ckomurlu
'''
from abc import ABCMeta, abstractmethod
import numpy as np
from collections import deque
import sys

import utils.properties
from utils.decorations import deprecated


class StrategyFactory(object):
    __metaclass__ = ABCMeta

    @staticmethod
    def generate_selection_strategy(strategy_name, **kwargs):
        selection_strategy = ''
        if 'randomStrategy2' == strategy_name:
            selection_strategy = RandomStrategy2(pool=kwargs['pool'], seed=kwargs['seed'])
        elif 'slidingWindow' == strategy_name:
            selection_strategy = SlidingWindow(pool=kwargs['pool'], seed=kwargs['seed'])
        elif 'impactBased' == strategy_name:
            selection_strategy = ImpactBased(pool=kwargs['pool'], parentDict=kwargs['parentDict'],
                                             childDict=kwargs['childDict'], cpdParams=kwargs['cpdParams'],
                                             rvCount=kwargs['rvCount'])
        elif 'minimumImpactBased' == strategy_name:
            selection_strategy = MinimumImpactBased(pool=kwargs['pool'], parentDict=kwargs['parentDict'],
                                             childDict=kwargs['childDict'], cpdParams=kwargs['cpdParams'],
                                             rvCount=kwargs['rvCount'])
        elif 'netImpactBased' == strategy_name:
            selection_strategy = NetImpactBased(pool=kwargs['pool'], parentDict=kwargs['parentDict'],
                                             childDict=kwargs['childDict'], cpdParams=kwargs['cpdParams'],
                                             rvCount=kwargs['rvCount'])
        elif 'varianceBased' == strategy_name:
            selection_strategy = VarianceBased()
        elif 'varianceBased2' == strategy_name:
            selection_strategy = VarianceBased2()
        elif 'incrementalVariance' == strategy_name:
            selection_strategy = IncrementalVarianceBased()
        elif 'firstOrderChildren' == strategy_name:
            selection_strategy = FirstorderVarianceReductionOnChildren()
        elif 'firstOrderChildren_err' == strategy_name:
            selection_strategy = FirstorderVarianceReductionOnChildren_err()
        elif 'batchTotalVarianceReduction' == strategy_name:
            selection_strategy = BatchTotalVarianceReduction(pool=kwargs['pool'])
        elif 'iterativeTotalVarianceReduction' == strategy_name:
            selection_strategy = IterativeTotalVarianceReduction(pool=kwargs['pool'])
        elif 'constantSelection' == utils.properties.selectionStrategy:
            selection_strategy = ConstantSelection()
        else:
            raise ValueError('Unknown strategy choice. Please double check selection strategy name in ' +
                             'utils.properties.')
        return selection_strategy


class AbstractSelectionStrategy(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def choices(self, **kwargs):
        pass

    @abstractmethod
    def __str__(self):
        pass

@deprecated
class RandomStrategy(object):
    def __init__(self, seed=1):
        self.rgen = np.random.RandomState(seed)
    
    def choices(self, pool, k):
        permuted_pool = self.rgen.permutation(pool)
        permuted_pool = map(tuple, permuted_pool)
        return permuted_pool[:k],permuted_pool[k:]


class RandomStrategy2(AbstractSelectionStrategy):
    def __init__(self, pool, seed=1):
        self.seed = seed
        self.rgen = np.random.RandomState(seed)
        self.pool = np.array(pool)
    
    def choices(self, count_selectees, **kwargs):
        self.rgen.shuffle(self.pool)
        return self.pool[:count_selectees].tolist()

    def __str__(self):
        return '[rgen seed: ' + str(self.seed) + ', pool: ' + str(self.pool) + ']'


class ConstantSelection(AbstractSelectionStrategy):
    def __init__(self):
        self.preselections = utils.properties.preselections

    def choices(self, count_selectees, **kwargs):
        if count_selectees != len(self.preselections):
            raise ValueError('count_selectees and the length of self.preselections do not match.')
        return self.preselections

    def __str__(self):
        return 'constant selections: ' + str(self.preselections)


class SlidingWindow(AbstractSelectionStrategy):
    def __init__(self, pool, seed=1):
        self.seed = seed
        self.rgen = np.random.RandomState(seed)
        self.pool = np.array(pool)
        self.rgen.shuffle(self.pool)
        self.rotationDeque = deque(self.pool)
    
    def choices(self, count_selectees, **kwargs):
        selectees = list(self.rotationDeque)[:count_selectees]
        self.rotationDeque.rotate(-count_selectees)
        return selectees

    def __str__(self):
        return '[rgen seed: ' + str(self.seed) + ', pool: ' + str(self.pool) + \
               ', deque: ' + str(list(self.rotationDeque)) + ']'


class ImpactBased(AbstractSelectionStrategy):
    def __init__(self, pool, parentDict, childDict, cpdParams, rvCount):
        self.pool = pool
        self.parentDict = parentDict
        self.childDict = childDict
        self.cpdParams = cpdParams
        self.rvCount = rvCount

    def choices(self, count_selectees, t, evidMat, **kwargs):
        selectees = list()  # np.empty(shape=(countSelectees,),dtype=np.int_)
        for i in range(count_selectees):
            max_impact = 0
            max_imapct_sensor = -1
            for sensor in self.pool:
                if sensor not in selectees:
                    current_impact = self._computeSensorImpact(sensor, t, evidMat)
                    if current_impact > max_impact:
                        max_impact = current_impact
                        max_imapct_sensor = sensor
            selectees.append(max_imapct_sensor)
        return selectees

    def _computeSensorImpact(self, sensor, t, evidMat):
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
            if child < self.rvCount:
                parents = np.array(self.parentDict[child])
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

    def __str__(self):
        return '[pool: ' + str(self.pool) + ', rv count: ' + str(self.rvCount) + ']'


class MinimumImpactBased(ImpactBased):
    def choices(self, count_selectees, t, evidMat, **kwargs):
        selectees = list()  # np.empty(shape=(countSelectees,),dtype=np.int_)
        for i in range(count_selectees):
            min_impact = sys.maxint
            min_imapct_sensor = -1
            for sensor in self.pool:
                if sensor not in selectees:
                    current_impact = self._computeSensorImpact(sensor, t, evidMat)
                    if current_impact < min_impact:
                        min_impact = current_impact
                        min_imapct_sensor = sensor
            selectees.append(min_imapct_sensor)
        return selectees


class NetImpactBased(ImpactBased):
    def _computeSensorImpact(self, sensor, t, evid_mat):
        if 0 == t:
            betas = np.abs(self.cpdParams[sensor][0][1])
        else:
            betas = np.abs(self.cpdParams[sensor][1][1])
        impact_factor = 0
        parents = np.array(self.parentDict[sensor])
        for parent in parents:
            if parent < self.rvCount:
                index_in_beta_vec = np.where(parents == parent)
                if evid_mat[parent, t]:
                    impact_factor -= betas[index_in_beta_vec]
                else:
                    impact_factor += betas[index_in_beta_vec]
            elif t > 0:
                index_in_beta_vec = np.where(parents == parent)
                if evid_mat[parent - self.rvCount, t-1]:
                    impact_factor -= betas[index_in_beta_vec]
                else:
                    impact_factor += betas[index_in_beta_vec]
        children = self.childDict[sensor]
        for child in children:
            if child < self.rvCount:
                parents = np.array(self.parentDict[child])
                if 0 == t:
                    betas = np.abs(self.cpdParams[child][0][1])
                else:
                    betas = np.abs(self.cpdParams[child][1][1])
                index_in_beta_vec = np.where(parents == sensor)
                if evid_mat[child, t]:
                    impact_factor -= betas[index_in_beta_vec]
                else:
                    impact_factor += betas[index_in_beta_vec]
                    #  the next time slice is irrelevant of the current/prediction time slice
                    # elif t < T:
                    #     if not evidMat[sensorid,t+1]:
                    #         betas = cpdParams[sensor][1]
                    #         indexInBetaVec = np.where(parents == sensor+rvCount)
                    #         impact_factor += betas[indexInBetaVec]
        return impact_factor


class VarianceBased(AbstractSelectionStrategy):
    def __init__(self, **kwargs):
        self.mostRecentSelectees = None

    def choices(self, count_selectees, t, evidMat, predictionModel, testMat, sampleSize, burnInCount,
                startupVals=None, tWin=None, **kwargs):
        curEvidMat = evidMat[:, :t + 1]
        Ymeanpred, Yvarpred = predictionModel.predict(testMat, curEvidMat, sampleSize=sampleSize, tWin=tWin,
                                                      burnInCount=burnInCount, startupVals=startupVals, t=t)
        varianceList = Yvarpred[:, -1]
        sortedIds = np.argsort(varianceList)
        sortedIds = sortedIds[::-1]
        self.mostRecentSelectees = sortedIds[:count_selectees]
        return self.mostRecentSelectees

    def __str__(self):
        return 'most recent selectees: ' + str(self.mostRecentSelectees)


class FirstorderVarianceReductionOnChildren(VarianceBased):
    def choices(self, count_selectees, t, evidMat, predictionModel, testMat, sampleSize, burnInCount,
                startupVals=None, tWin=None, **kwargs):
        curEvidMat = evidMat[:, :t+1]
        n_var = len(predictionModel.sortedids)
        betasqsum_list = [None] * n_var
        if t == 0:
            for var in predictionModel.sortedids:
                betasqsum = 0
                for child in predictionModel.childDict[var]:
                    betasqsum += predictionModel.cpdParams[child, 0][1][
                                     predictionModel.parentDict[child].index(var), 0] ** 2
                betasqsum += predictionModel.cpdParams[var, 1][1][
                    predictionModel.parentDict[var].index(var + n_var), 0] ** 2
                betasqsum_list[var] = betasqsum
        else:
            for var in predictionModel.sortedids:
                betasqsum = 0
                for child in predictionModel.childDict[var]:
                    betasqsum += predictionModel.cpdParams[child, 1][1][
                                     predictionModel.parentDict[child].index(var), 0] ** 2
                betasqsum += predictionModel.cpdParams[var, 1][1][
                    predictionModel.parentDict[var].index(var + n_var), 0] ** 2
                betasqsum_list[var] = betasqsum
        self.mostRecentSelectees = list()
        for i in range(count_selectees):
            Ymeanpred, Yvarpred = predictionModel.predict(testMat, curEvidMat, sampleSize=sampleSize, tWin=tWin,
                                                          burnInCount=burnInCount, startupVals=startupVals, t=t)
            Yvarpred = Yvarpred[:, -1].squeeze()
            varReductionList = np.multiply(Yvarpred, betasqsum_list)
            selectee = np.argmax(varReductionList)
            self.mostRecentSelectees.append(selectee)
            curEvidMat[selectee, t] = True
        return self.mostRecentSelectees


@deprecated
class FirstorderVarianceReductionOnChildren_err(VarianceBased):
    def choices(self, count_selectees, t, evidMat, predictionModel, testMat, sampleSize, burnInCount,
                startupVals=None, tWin=None, **kwargs):
        curEvidMat = evidMat[:, :t+1]
        betasqs = map(lambda x: float(np.dot(x[1].T, x[1])), predictionModel.cpdParams[:, 0]) if t == 0 else \
            map(lambda x: float(np.dot(x[1].T, x[1])), predictionModel.cpdParams[:, 1])
        self.mostRecentSelectees = list()
        for i in range(count_selectees):
            Ymeanpred, Yvarpred = predictionModel.predict(testMat, curEvidMat, sampleSize=sampleSize, tWin=tWin,
                                                          burnInCount=burnInCount, startupVals=startupVals, t=t)
            Yvarpred = Yvarpred[:, -1].squeeze()
            varReductionList = np.multiply(Yvarpred, betasqs)
            selectee = np.argmax(varReductionList)
            self.mostRecentSelectees.append(selectee)
            curEvidMat[selectee, t] = True
        return self.mostRecentSelectees


class BatchTotalVarianceReduction(AbstractSelectionStrategy):
    def __init__(self, pool):
        super(BatchTotalVarianceReduction, self).__init__()
        self.pool = pool

    def choices(self, count_selectees, t, evidMat, predictionModel, testMat, **kwargs):
        curEvidMat = evidMat[:, :t + 1].copy()
        totalVariances = list()
        for var in self.pool:
            if curEvidMat[var, -1]:
                continue
            curEvidMat[var, -1] = True
            Ymeanpred, Yvarpred = predictionModel.predict(testMat, curEvidMat, t=t, prediction_slices=-1,
                                                          last_slice_only=True, **kwargs)
            totalVariances.append((var, sum(Yvarpred)))
            curEvidMat[var, -1] = False
        selectees = [x[0] for x in sorted(totalVariances, key=lambda x: x[1])[:count_selectees]]
        return selectees

    def __str__(self):
        '[pool: ' + str(self.pool) + ']'


class IterativeTotalVarianceReduction(BatchTotalVarianceReduction):
    def __init__(self, pool):
        super(IterativeTotalVarianceReduction, self).__init__(pool)
        self.pool = pool[:]
        self.original_pool = self.pool[:]

    def choices(self, count_selectees, t, evidMat, predictionModel, testMat, **kwargs):
        local_evidence_mat = evidMat[:, :t + 1].copy()
        selectees = list()
        for i in range(count_selectees):
            nxt = super(IterativeTotalVarianceReduction, self).choices(1, t, local_evidence_mat, predictionModel,
                                                                       testMat, **kwargs)[0]
            selectees.append(nxt)
            local_evidence_mat[nxt] = True
            self.pool.remove(nxt)
        self.pool = self.original_pool[:]
        return selectees

class VarianceBased2(AbstractSelectionStrategy):
    def __init__(self, **kwargs):
        self.mostRecentSelectees = None

    def choices(self, count_selectees, evidMat, t, predictionModel, **kwargs):
        varianceList = predictionModel.computeVar(evidMat[:, :t + 1])
        sortedIds = np.argsort(varianceList)
        sortedIds = sortedIds[::-1]
        self.mostRecentSelectees = sortedIds[:count_selectees]
        return self.mostRecentSelectees

    def __str__(self):
        return 'most recent selectees: ' + str(self.mostRecentSelectees)


class IncrementalVarianceBased(AbstractSelectionStrategy):
    def __init__(self, **kwargs):
        self.mostRecentSelectees = None

    def choices(self, count_selectees, evidMat, t, predictionModel, **kwargs):
        tempEvidMat = evidMat[:, :t + 1]
        self.mostRecentSelectees = list()
        for i in range(count_selectees):
            varianceList = predictionModel.computeVar(tempEvidMat)
            rvMaxVar = np.argmax(varianceList)
            self.mostRecentSelectees.append(rvMaxVar)
            tempEvidMat[rvMaxVar, t] = 1
        return self.mostRecentSelectees

    def __str__(self):
        return 'most recent selectees: ' + str(self.mostRecentSelectees)

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