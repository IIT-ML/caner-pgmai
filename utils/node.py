'''
Created on Jan 13, 2015

@author: ckomurlu
'''
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
        
