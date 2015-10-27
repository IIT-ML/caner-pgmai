'''
Created on Jul 28, 2015

@author: ckomurlu
'''
#This module is dedicated to properties addressed during experiments.

import time
import datetime
import numpy as np
ts = time.time()

k2bin10StructureParentChildDictPath = 'C:/Users/ckomurlu/Documents/workbench/experiments/20150911/' + \
                                      'parentChildDicts_k2_bin10.pkl'
k2bin5StructureParentChildDictPath = 'C:/Users/ckomurlu/Documents/workbench/experiments/20151023/' + \
                                     'parentChildDicts_k2_bin5.pkl'

outputDirPath = 'C:/Users/ckomurlu/Documents/workbench/experiments/20151027/4_NBS_K2/'
timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%Y%m%d_%H%M')
numParallelThreads = 18

mh_sampleSize = 2000
mh_burnInCount = 1000
mh_tuneWindow = 50
mh_startupWidth = 5.0

tWin = 12
timeSpan = 12
obsrateList = np.arange(0.0, 0.7, 0.1)
numTrials = 5

# selectionStrategy = 'randomStrategy2'
# selectionStrategy = 'slidingWindow'
# selectionStrategy = 'impactBased'
selectionStrategy = 'netImpactBased'

dbn_topology = 'k2_bin5' #, 'k2_bin10', 'mst', 'mst_enriched', 'imt'


