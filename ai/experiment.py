from abc import ABCMeta
import numpy as np
from time import time
import datetime
import os
import multiprocessing as mp

import utils.properties
from joint.gaussian_DBN import GaussianDBN
from data.data_provider import DataProvider
from ai.selection_strategy import StrategyFactory
from utils.toolkit import standard_error
from independent.gaussian_process import GaussianProcessLocal


def testActiveInferenceGaussianDBNParallel():
    start = time()
    print 'Process started at:', datetime.datetime.fromtimestamp(start).strftime('%H:%M, %m/%d/%Y')
    tWin = utils.properties.tWin
    topology = utils.properties.dbn_topology
    T = utils.properties.timeSpan
    numTrials = utils.properties.numTrials

    trainset, testset = DataProvider.provide_data()

    if 'gp' == utils.properties.prediction_model:
        prediction_model = GaussianProcessLocal()
    elif 'dgbn' == utils.properties.prediction_model:
        prediction_model = GaussianDBN()
    else:
        raise ValueError('Unrecognized prediction model name')
    print 'Prediction model selected: ', prediction_model.__class__
    prediction_model.fit(trainset, topology=topology)
    print 'Prediction model was trained.'
    Y_test_allT = np.vectorize(lambda x: x.true_label)(testset)
    parameterList = list()
    sampleSize = utils.properties.mh_sampleSize
    burnInCount = utils.properties.mh_burnInCount
    for obsrate in utils.properties.obsrateList:
        obsCount = int(obsrate * prediction_model.rvCount)
        evidencepath = utils.properties.outputDirPath + str(obsrate) + '/evidences/'
        if not os.path.exists(evidencepath):
            os.makedirs(evidencepath)
        predictionpath = utils.properties.outputDirPath + str(obsrate) + '/predictions/'
        if not os.path.exists(predictionpath):
            os.makedirs(predictionpath)
        errorpath = utils.properties.outputDirPath + str(obsrate) + '/errors/'
        if not os.path.exists(errorpath):
            os.makedirs(errorpath)
        selection_strategy_name = utils.properties.selectionStrategy
        if 0.0 == obsrate:
            trial = 0
            parameterList.append({'trial': trial, 'prediction_model': prediction_model,
                                  'selection_strategy_name': selection_strategy_name, 'T': T, 'tWin': tWin,
                                  'testset': testset, 'Y_test_allT': Y_test_allT, 'sampleSize': sampleSize,
                                  'burnInCount': burnInCount, 'topology': topology, 'obsrate': obsrate,
                                  'obsCount': obsCount, 'evidencepath': evidencepath,
                                  'predictionpath': predictionpath, 'errorpath': errorpath})
        else:
            for trial in range(numTrials):
                parameterList.append({'trial': trial, 'prediction_model': prediction_model,
                                      'selection_strategy_name': selection_strategy_name, 'T': T, 'tWin': tWin,
                                      'testset': testset, 'Y_test_allT': Y_test_allT, 'sampleSize': sampleSize,
                                      'burnInCount': burnInCount, 'topology': topology, 'obsrate': obsrate,
                                      'obsCount': obsCount, 'evidencepath': evidencepath,
                                      'predictionpath': predictionpath, 'errorpath': errorpath})

    print 'Tasks for parallel computation were created.'
    pool = mp.Pool(processes=utils.properties.numParallelThreads)
    print 'Tasks in parallel are being started.'
    pool.map(trialFuncStar, parameterList)

    for obsrate in utils.properties.obsrateList:
        errorpath = utils.properties.outputDirPath + str(obsrate) + '/errors/'
        if 0.0 == obsrate:
            trial = 0
            errResults = np.loadtxt(errorpath +
                                    'mae_activeInfo_model={}_topology={}_window={}_T={}_obsRate={}_trial={}.csv'.
                                    format(utils.properties.prediction_model, topology, tWin, T, obsrate,
                                                          trial), delimiter=',')
            np.savetxt(errorpath +
                       'meanMAE_activeInf_model={}_topology={}_window={}_T={}_obsRate={}_trial={}.csv'.
                       format(utils.properties.prediction_model, topology, tWin, T, obsrate, 'mean'),
                       errResults, delimiter=',')
            np.savetxt(errorpath +
                       'stderrMAE_activeInf_model={}_topology={}_window={}_T={}_obsRate={}_trial={}.csv'.
                       format(utils.properties.prediction_model, topology, tWin, T, obsrate, 'mean'),
                       np.zeros(shape=errResults.shape), delimiter=',')
        else:
            errResults = np.empty(shape=(numTrials, T, 6))
            for trial in range(numTrials):
                errResults[trial] = np.loadtxt(errorpath + ('mae_activeInfo_model={}_topology={}_window={}_' +
                                               'T={}_obsRate={}_trial={}.csv').
                                               format(utils.properties.prediction_model, topology,tWin,T,obsrate,
                                               trial), delimiter=',')
            np.savetxt(errorpath +
                       'meanMAE_activeInf_model={}_topology={}_window={}_T={}_obsRate={}_trial={}.csv'.
                       format(utils.properties.prediction_model, topology,tWin,T,obsrate, 'mean'),
                       np.mean(errResults,axis=0), delimiter=',')
            np.savetxt(errorpath +
                       'stderrMAE_activeInf_model={}_topology={}_window={}_T={}_obsRate={}_trial={}.csv'.
                       format(utils.properties.prediction_model, topology,tWin,T,obsrate, 'mean'),
                       standard_error(errResults, axis=0), delimiter=',')
    print 'End of process, duration: {} secs'.format(time() - start)


def trialFuncStar(allParams):
    trialFunc(**allParams)


def trialFunc(trial, prediction_model, selection_strategy_name, T, tWin, testset, Y_test_allT, sampleSize, burnInCount,
              topology, obsrate, obsCount, evidencepath, predictionpath, errorpath, sensormeans=None):
    print 'obsrate {} trial {}'.format(obsrate, trial)
    evidMat = np.zeros(shape=(prediction_model.rvCount, T), dtype=np.bool_)
    selectionStrategy = StrategyFactory.generate_selection_strategy(selection_strategy_name, seed=trial,
                                                                    pool=prediction_model.sortedids,
                                                                    parentDict=prediction_model.parentDict,
                                                                    childDict=prediction_model.childDict,
                                                                    cpdParams=prediction_model.cpdParams,
                                                                    rvCount=prediction_model.rvCount)
    predResults = np.empty(shape=(prediction_model.rvCount, T))
    errResults = np.empty(shape=(T, 6))
    for t in range(T):
        selectees = selectionStrategy.choices(count_selectees=obsCount, t=t, evidMat=evidMat)
        evidMat[selectees, t] = True
        if t < tWin:
            Y_test = Y_test_allT[:, :t+1]
            testMat = testset[:, :t+1]
            curEvidMat = evidMat[:, :t+1]
        else:
            Y_test = Y_test_allT[:, t+1-tWin:t+1]
            testMat = testset[:, t+1-tWin:t+1]
            curEvidMat = evidMat[:, t+1-tWin:t+1]
        if sensormeans is not None:
            startupVals = np.repeat(sensormeans.reshape(-1, 1), Y_test.shape[1], axis=1)
        else:
            startupVals = None
        Y_pred = prediction_model.predict(testMat, curEvidMat, sampleSize=sampleSize, burnInCount=burnInCount,
                                          startupVals=startupVals, obsrate=obsrate, trial=trial, t=t)
        predResults[:, t] = Y_pred[:, -1]
        errResults[t, 0] = prediction_model.compute_mean_absolute_error(testMat[:, -1], Y_pred[:, -1],
                                                                        type_=0, evidence_mat=curEvidMat[:, -1])
        errResults[t, 1] = prediction_model.compute_mean_squared_error(testMat[:, -1], Y_pred[:, -1],
                                                                       type_=0, evidence_mat=curEvidMat[:, -1])
        errResults[t, 2] = prediction_model.compute_mean_absolute_error(testMat[:, -1], Y_pred[:, -1],
                                                                        type_=1, evidence_mat=curEvidMat[:, -1])
        errResults[t, 3] = prediction_model.compute_mean_squared_error(testMat[:, -1], Y_pred[:, -1],
                                                                       type_=1, evidence_mat=curEvidMat[:, -1])
        errResults[t, 4] = prediction_model.compute_mean_absolute_error(testMat[:, -1], Y_pred[:, -1],
                                                                        type_=2, evidence_mat=curEvidMat[:, -1])
        errResults[t, 5] = prediction_model.compute_mean_squared_error(testMat[:, -1], Y_pred[:, -1],
                                                                       type_=2, evidence_mat=curEvidMat[:, -1])
    np.savetxt(evidencepath +
               '{}_activeInf_gaussianDBN_T={}_trial={}_obsrate={}.csv'.
               format('evidMat', T, trial, obsrate),
               evidMat, delimiter=',')
    np.savetxt(predictionpath +
               '{}_activeInf_gaussianDBN_T={}_trial={}_obsRate={}.csv'.
               format('predResults', T, trial, obsrate),
               predResults, delimiter=',')
    np.savetxt(errorpath +
               '{}_activeInfo_gaussianDBN_topology={}_window={}_T={}_obsRate={}_trial={}.csv'.
               format('mae', topology, tWin, T, obsrate, trial),
               errResults, delimiter=',')
