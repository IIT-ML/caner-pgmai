'''
Created on Jan 13, 2015

@author: ckomurlu
'''

import numpy as np

class RandomVarNode(object):
    def __init__(self, true_label=None, local_feature_vector=None,
                 is_observed=False, neighbors=None):
        self.true_label = true_label
        self.predicted_label = None
        self.local_feature_vector = local_feature_vector
        self.is_observed = is_observed
        self.neighbors = neighbors
    
class SensorRVNode(RandomVarNode):
    def __init__(self, sensor_id, dig_time, day, true_label=None,
                 local_feature_vector=None, is_observed= False,neighbors=None):
        super(SensorRVNode,self).__init__(true_label, local_feature_vector,
                is_observed, neighbors)
        self.sensor_id = sensor_id
        self.dig_time = dig_time
        self.day = day
        

#neighborhood function list
class Neighborhood:
    @staticmethod
    def independent_back(self_id, sensor_IDs):
        neighbors = [(self_id,-1)]
        return neighbors
    
    @staticmethod
    def all_nodes_current_time(self_id, sensor_IDs):
        neighbors = []
        neighbors += zip(sensor_IDs,[0]*len(sensor_IDs))
        return neighbors
    
    @staticmethod
    def all_others_current_time(self_id, sensor_IDs):
        neighbors = []
        neighbors += zip(np.setdiff1d(sensor_IDs, [self_id]),[0]*
                        (len(sensor_IDs)-1))
        return neighbors
    
    @staticmethod
    def itself_previous_others_current(self_id, sensor_IDs):
        neighbors = [(self_id,-1)]
        neighbors += zip(np.setdiff1d(sensor_IDs, [self_id]),[0]*
                        (len(sensor_IDs)-1))
        return neighbors
    
    @staticmethod
    def itself_current_only(self_id, sensor_IDs):
        neighbors = [(self_id,0)]
        return neighbors

    @staticmethod
    def itself_previous_following_others_current(self_id, sensor_IDs):
        neighbors = [(self_id,-1)]
        neighbors.append((self_id,1))
        neighbors += zip(np.setdiff1d(sensor_IDs, [self_id]),[0]*
                        (len(sensor_IDs)-1))
        return neighbors