'''
Created on Jul 28, 2015

@author: ckomurlu
'''
#This module is dedicated to properties addressed during experiments.

import time
import datetime
import numpy as np
ts = time.time()

# temperature data
temperature_k2_bin10_topology_ParentChildDictPath = 'C:/Users/ckomurlu/Documents/workbench/experiments/20150911/' + \
                                      'parentChildDicts_k2_bin10.pkl'
temperature_k2_bin5_topology_ParentChildDictPath = 'C:/Users/ckomurlu/Documents/workbench/experiments/20151023/' + \
                                     'parentChildDicts_k2_bin5.pkl'

# humidity
humidity_k2_bin5_topology_ParentChildDictPath = 'C:/Users/ckomurlu/Documents/workbench/experiments/20151102/' + \
                                     'humidity_parentChildDicts_k2_bin5.pkl'

# temperature + humidity
temperature_humidity_k2_bin5_topology_ParentChildDictPath = 'C:/Users/ckomurlu/Documents/workbench/' + \
                                     'experiments/20151103/temp_humid_parentChildDicts_k2_bin5.pkl'

# wunderground-IL
wground_IL_k2_bin5_topology_ParentChildDictPath = 'C:/Users/ckomurlu/Documents/workbench/data/wunderground/IL/' \
                                               'wground_parentChildDicts_k2_bin5.pkl'

# wunderground-countrywide
wground_cdwide_k2_bin5_topology_ParentChildDictPath = 'C:/Users/ckomurlu/Documents/workbench/data/wunderground/' \
                                               'countrywide/wground_parentChildDicts_k2_bin5.pkl'

# outputDirPath = 'C:/Users/ckomurlu/Documents/workbench/experiments/20151103/5_GP/'
# outputDirPath = 'C:/Users/ckomurlu/Documents/workbench/experiments/20151029/5_GP/'
# outputDirPath = 'C:/Users/ckomurlu/Documents/workbench/experiments/20151121/temperature/GP/SW/'
# outputDirPath = 'C:/Users/ckomurlu/Documents/workbench/experiments/20161227/temperature/DGBN/SW/'
outputDirPath = 'C:/Users/ckomurlu/Documents/workbench/experiments/20170120/countrywide/DGBN/RND/'
timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%Y%m%d_%H%M')
numParallelThreads = 10

mh_sampleSize = 2000
mh_burnInCount = 1000
mh_tuneWindow = 50
mh_startupWidth = 5.0

tWin = 12
timeSpan = 12
obsrateList = np.arange(0.0, 0.7, 0.1)
numTrials = 5

selectionStrategy = 'randomStrategy2'
# selectionStrategy = 'slidingWindow'
# selectionStrategy = 'impactBased'
# selectionStrategy = 'netImpactBased'
# selectionStrategy = 'varianceBased'
# selectionStrategy = 'varianceBased2'

dbn_topology = 'k2_bin5' #, 'k2_bin10', 'mst', 'mst_enriched', 'imt'

# data = 'temperature'
# data = 'humidity'
# data = 'temperature+humidity'
# data = 'wunderground-IL'
data = 'wunderground-cwide'

# prediction_model = 'gp'
# prediction_model = 'kf'
# prediction_model = 'lc-linear'
# prediction_model = 'lc-ridge'
# prediction_model = 'lc-lasso'
prediction_model = 'dgbn'

# linearChainRegressionMethod = 'linear'
# linearChainRegressionMethod = 'ridge'
# linearChainRegressionMethod = 'lasso'


#Kalman Filter related parameters
aggregation_period = 4  # This parameter is the length of window on which we will aggregate observation. E.g obs[0] will
                        # be the mean of true_label[0], true_label[aggregation_period], true_label[2*aggregation_period]