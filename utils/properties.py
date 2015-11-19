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
# k2bin10StructureParentChildDictPath = 'C:/Users/ckomurlu/Documents/workbench/experiments/20150911/' + \
#                                       'parentChildDicts_k2_bin10.pkl'
k2bin5StructureParentChildDictPath = 'C:/Users/ckomurlu/Documents/workbench/experiments/20151023/' + \
                                     'parentChildDicts_k2_bin5.pkl'

# humidity
# k2bin5StructureParentChildDictPath = 'C:/Users/ckomurlu/Documents/workbench/experiments/20151102/' + \
#                                      'humidity_parentChildDicts_k2_bin5.pkl'

# temperature + humidity
# k2bin5StructureParentChildDictPath = 'C:/Users/ckomurlu/Documents/workbench/experiments/20151103/' + \
#                                      'temp_humid_parentChildDicts_k2_bin5.pkl'

# outputDirPath = 'C:/Users/ckomurlu/Documents/workbench/experiments/20151103/5_GP/'
# outputDirPath = 'C:/Users/ckomurlu/Documents/workbench/experiments/20151029/5_GP/'
outputDirPath = 'C:/Users/ckomurlu/Documents/workbench/experiments/20151102/debug/'
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
selectionStrategy = 'netImpactBased'

dbn_topology = 'k2_bin5' #, 'k2_bin10', 'mst', 'mst_enriched', 'imt'


