__author__ = 'ckomurlu'

from utils.metropolis_hastings import MetropolisHastings

import numpy as np
from sklearn.metrics import mean_absolute_error,mean_squared_error

def test_sample_temporal():
    parentDict = {0:[3],1:[0,4],2:[1,5]}

    cpdParams = np.empty(shape=(3,2),dtype=tuple)
    cpdParams[0,0] = (21.7211,np.array([]),5.79045)
    cpdParams[1,0] = (-1.43977,np.array([1.04446]),0.638121)
    cpdParams[2,0] = (-1.58985,np.array([1.0515]),0.871026)
    cpdParams[0,1] = (0.90683765,np.array([0.95825092]),0.41825906)
    cpdParams[1,1] = (-1.17329,np.array([0.49933,0.544751]),0.278563)
    cpdParams[2,1] = (-0.902438,np.array([0.410928,0.622744]),0.390682)
    rvids = np.array(parentDict.keys())
    rvCount = int(rvids.shape[0])
    T = 4
    evidMat = np.zeros((rvCount,T),dtype=bool)
    evidMat[1,1] = True
    evidMat[-1,-1] = True
    testMat = np.zeros((rvCount,T))
    testMat[1,1] = 25.0
    testMat[-1,-1] = 19.0

    sampleSize = 2000
    burnInCount = 1000
    samplingPeriod = 2
    width = 4

    startupVals = np.array([[20., 20., 20., 20.],
                            [20., 20., 20., 20.],
                            [20., 20., 20., 20.]])

    mh = MetropolisHastings()

    (data,accList,propVals,accCount) = mh.sampleTemporal(rvids,
                    parentDict, cpdParams, startupVals, evidMat,
                    testMat, sampleSize=sampleSize,
                    burnInCount=burnInCount, samplingPeriod=samplingPeriod,
                    proposalDist='uniform', width=width)

    refMean = np.array([[24.06, 23.80, 22.67, 22.14],
                        [24.16, 25.00, 23.00, 21.75],
                        [22.61, 22.58, 21.21, 19.00]])

    refVar = np.array([[0.45, 0.46, 0.73, 1.02],
                       [0.49, 0.00, 0.39, 0.69],
                       [1.31, 0.81, 0.47, 0.00]])

    # dataArray = np.array(data[burnInCount::samplingPeriod])
    dataArray = np.array(data)
    print 'Total sample size: ', sampleSize,'burnin: ', burnInCount
    print 'freq: 1/',samplingPeriod, 'width: ', width
    print 'mean'
    myMean = np.mean(dataArray, axis=0)
    print myMean
    print 'mse:', mean_squared_error(refMean,myMean), 'mse on nonevidence:', mean_squared_error(refMean[~evidMat], myMean[~evidMat])
    print 'var'
    myVar = np.var(dataArray, axis=0)
    print myVar
    print 'mse:', mean_squared_error(refVar, myVar), 'mse on nonevidence:', mean_squared_error(refVar[~evidMat], myVar[~evidMat])
    print 'rejection ratio:'
    print 1 - accCount/sampleSize

test_sample_temporal()