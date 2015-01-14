'''
Created on Jan 13, 2015

@author: ckomurlu
'''
class RandomVarNode(object):
    def __init__(self, true_label=None, local_feature_vector=None,
                 is_observed= False):
        self.true_label = true_label
        self.predicted_label = None
        self.local_feature_vector = local_feature_vector
        self.is_observed = is_observed

class SensorRVNode(RandomVarNode):
    def __init__(self, sensor_id, time, true_label=None,
                 local_feature_vector=None, is_observed= False):
        super(SensorRVNode,self).__init__(true_label, local_feature_vector,
                is_observed)
        self.sensor_id = sensor_id
        self.time = time
        
myob = SensorRVNode(sensor_id=5,time=3)