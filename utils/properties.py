'''
Created on Jul 28, 2015

@author: ckomurlu
'''
#This module is dedicated to properties addressed during experiments.

import time
import datetime
import numpy as np
ts = time.time()

k2StructureParentChildDictPath = r'C:\Users\ckomurlu\Documents\workbench\experiments\20150911\parentChildDicts.pkl'

outputDirPath = 'C:/Users/ckomurlu/Documents/workbench/experiments/20151001/5_GP_K2_tWin12/'
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
selectionStrategy = 'impactBased'

dbn_topology = 'k2' #, 'mst', 'mst_enriched', 'imt'


