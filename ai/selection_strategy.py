'''
Created on Jan 8, 2015

@author: ckomurlu
'''

import numpy as np

class RandomStrategy(object):
    def __init__(self, seed=0):
        self.rgen = np.random.RandomState(seed)
    
    def choices(self, pool, k): 
        return self.rgen.permutation(pool)[:k]

class UNCSampling(object):
    '''
    This class performs uncertainty sampling based on the model.
    '''
    
    def choices(self, model, X, pool, k):        
        y_decision = model.decision_function(X[pool])
        uncerts = np.argsort(np.min(np.absolute(y_decision),axis=1))[:k]
        return pool[uncerts]

    def choices_model_per_sensor(self, model_dict, X_dict, pool_dict, k):
#       for implementation purpose only,
        model_dict = dict()
        X_dict = dict()
        pool_dict = dict()
        
        current_sensor_list = model_dict.keys()
        for sensor in current_sensor_list:
            pass