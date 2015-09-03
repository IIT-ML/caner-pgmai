'''
Created on Jul 28, 2015

@author: ckomurlu
'''
#This module is dedicated to properties addressed during experiments.

import time
import datetime
import numpy as np
ts = time.time()

outputDirPath = 'C:/Users/ckomurlu/Documents/workbench/experiments/20150825/13-IMT-debugging/'
timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%Y%m%d_%H%M')
numParallelThreads = 18

mh_sampleSize = 1000
mh_burnInCount = 500
mh_tuneWindow = 50
mh_startupWidth = 5.0

timeSpan = 12
obsrateList = [.6] #np.arange(0.0,0.7,0.1)
numTrials = 1

dbn_topology = 'imt' #, 'mst', 'mst_enriched'
