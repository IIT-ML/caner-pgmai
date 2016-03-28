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

# outputDirPath = 'C:/Users/ckomurlu/Documents/workbench/experiments/20151103/5_GP/'
# outputDirPath = 'C:/Users/ckomurlu/Documents/workbench/experiments/20151029/5_GP/'
# outputDirPath = 'C:/Users/ckomurlu/Documents/workbench/experiments/20151121/temperature/GP/SW/'
outputDirPath = 'C:/Users/ckomurlu/Documents/workbench/experiments/20160301/temperature/dGBn/VARdebug/'
timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%Y%m%d_%H%M')
numParallelThreads = 1

mh_sampleSize = 2000
mh_burnInCount = 1000
mh_tuneWindow = 50
mh_startupWidth = 5.0

tWin = 12
timeSpan = 12
obsrateList = [0.2]  # np.arange(0.0, 0.7, 0.1)
numTrials = 1

# selectionStrategy = 'randomStrategy2'
# selectionStrategy = 'slidingWindow'
# selectionStrategy = 'impactBased'
# selectionStrategy = 'netImpactBased'
selectionStrategy = 'varianceBased'

dbn_topology = 'k2_bin5' #, 'k2_bin10', 'mst', 'mst_enriched', 'imt'

data = 'temperature'
# data = 'humidity'
# data = 'temperature+humidity'

# prediction_model = 'gp'
# prediction_model = 'kf'
# prediction_model = 'lc-linear'
# prediction_model = 'lc-ridge'
# prediction_model = 'lc-lasso'
prediction_model = 'dgbn'

# linearChainRegressionMethod = 'linear'
# linearChainRegressionMethod = 'ridge'
# linearChainRegressionMethod = 'lasso'
