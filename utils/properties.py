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
temperature_k2_bin10_topology_ParentChildDictPath = 'C:/Users/CnrKmrl/Documents/workbench/experiments/20150911/' + \
                                      'parentChildDicts_k2_bin10.pkl'
temperature_k2_bin5_topology_ParentChildDictPath = 'C:/Users/CnrKmrl/Documents/workbench/data/intelResearch/' + \
                                     'parentChildDicts_k2_bin5.pkl'

# humidity
humidity_k2_bin5_topology_ParentChildDictPath = 'C:/Users/CnrKmrl/Documents/workbench/data/intelResearch/humidity/' + \
                                     'humidity_parentChildDicts_k2_bin5.pkl'

# temperature + humidity
temperature_humidity_k2_bin5_topology_ParentChildDictPath = 'C:/Users/CnrKmrl/Documents/workbench/' + \
                                     'experiments/20151103/temp_humid_parentChildDicts_k2_bin5.pkl'

# wunderground-IL
wground_IL_k2_bin5_topology_ParentChildDictPath = 'C:/Users/CnrKmrl/Documents/workbench/data/wunderground/IL/' \
                                               'wground_parentChildDicts_k2_bin5.pkl'

# wunderground-countrywide
wground_cdwide_k2_bin5_topology_ParentChildDictPath = 'C:/Users/CnrKmrl/Documents/workbench/data/wunderground/' \
                                               'countrywide/wground_parentChildDicts_k2_bin5.pkl'

# outputDirPath = 'C:/Users/ckomurlu/Documents/workbench/experiments/20151103/5_GP/'
# outputDirPath = 'C:/Users/ckomurlu/Documents/workbench/experiments/20151029/5_GP/'
# outputDirPath = 'C:/Users/ckomurlu/Documents/workbench/experiments/20151121/temperature/GP/SW/'
# outputDirPath = 'C:/Users/ckomurlu/Documents/workbench/experiments/20161227/temperature/DGBN/SW/'
outputDirPath = 'C:/Users/CnrKmrl/Documents/workbench/experiments/20170923/lasso_shift_corrected/temperature/DGBN/RND/'
timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%Y%m%d_%H%M')
numParallelThreads = 2

tWin = 12
timeSpan = 12
obsrateList = np.arange(0.0, 0.7, 0.1)
numTrials = 5

selectionStrategy = 'randomStrategy2'
# selectionStrategy = 'slidingWindow'
# selectionStrategy = 'impactBased'
# selectionStrategy = 'minimumImpactBased'
# selectionStrategy = 'netImpactBased'
# selectionStrategy = 'varianceBased'
# selectionStrategy = 'firstOrderChildren'
# selectionStrategy = 'varianceByBetaSquare'
# selectionStrategy = 'batchTotalVarianceReduction'
# selectionStrategy = 'iterativeTotalVarianceReduction'
# selectionStrategy = 'varianceBased2'
# selectionStrategy = 'incrementalVariance'
# selectionStrategy = 'constantSelection'
# selectionStrategy = 'greedyCheating'
# preselections = [7, 33]
# preselections = [11, 35]
preselections = [11, 35, 12, 40, 5, 19, 29, 32, 44, 48]
# preselections = [1, 7, 11, 24, 40]
# preselections = [1, 9, 21, 30, 40]
# preselections = [1, 4, 6, 9, 14, 18, 20, 22, 25, 27, 30, 38, 40, 44, 47]  # count is 15
# preselections = [1, 2, 3, 4, 5, 6, 8, 9, 14, 16, 18, 20, 22, 25, 27, 30, 38, 40, 44, 47]  # count is 20
# preselections = [1, 2, 3, 4, 5, 6, 8, 9, 14, 15, 16, 18, 19, 20, 22, 25, 27, 30, 36, 38, 40, 41, 44, 45, 47]
# count is 25 in the above list
# preselections = [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 14, 15, 16, 18, 19, 20, 21, 22, 25, 27, 28, 29, 30, 36, 38, 40, 41,
#  44, 45, 47]
# count is 30 in the above list

# dbn_topology = 'k2_bin5'
# dbn_topology = 'fully_connected'
# dbn_topology = 'mst'
# dbn_topology = 'k2_bin10'
# dbn_topology = 'imt'
dbn_topology = 'lasso'
# 'mst_enriched'

data = 'temperature'
# data = 'humidity'
# data = 'temperature+humidity'
# data = 'wunderground-IL'
# data = 'wunderground-cwide'

aligned_data = True

# prediction_model = 'gp'
# prediction_model = 'kf'
# prediction_model = 'lc-linear'
# prediction_model = 'lc-ridge'
# prediction_model = 'lc-lasso'
prediction_model = 'dgbn'

# learningDBN = 'multivariate_Guassian'
learningDBN = 'lasso_regression'

# linearChainRegressionMethod = 'linear'
# linearChainRegressionMethod = 'ridge'
# linearChainRegressionMethod = 'lasso'


#Kalman Filter related parameters
aggregation_period = 4  # This parameter is the length of window on which we will aggregate observation. E.g obs[0] will
                        # be the mean of true_label[0], true_label[aggregation_period], true_label[2*aggregation_period]

mh_sampleSize = 2000
mh_burnInCount = 1000
mh_tuneWindow = 50
mh_startupWidth = 5.0