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
from independent.multivariate_discrete_kalman_filter import MultivariateDiscreteKalmanFilter
from independent.linear_chain import LinearChain


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
    elif 'kf' == utils.properties.prediction_model:
        prediction_model = MultivariateDiscreteKalmanFilter()
    elif 'lc-linear' == utils.properties.prediction_model:
        prediction_model = LinearChain(regressionMethod='linear')
    elif 'lc-ridge' == utils.properties.prediction_model:
        prediction_model = LinearChain(regressionMethod='ridge')
    elif 'lc-lasso' == utils.properties.prediction_model:
        prediction_model = LinearChain(regressionMethod='lasso')
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
        meanPredictionPath = utils.properties.outputDirPath + str(obsrate) + '/predictions/mean/'
        if not os.path.exists(meanPredictionPath):
            os.makedirs(meanPredictionPath)
        varPredictionPath = utils.properties.outputDirPath + str(obsrate) + '/predictions/var/'
        if not os.path.exists(varPredictionPath):
            os.makedirs(varPredictionPath)
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
                                  'meanPredictionPath': meanPredictionPath, 'varPredictionPath': varPredictionPath,
                                  'errorpath': errorpath})
        else:
            for trial in range(numTrials):
                parameterList.append({'trial': trial, 'prediction_model': prediction_model,
                                      'selection_strategy_name': selection_strategy_name, 'T': T, 'tWin': tWin,
                                      'testset': testset, 'Y_test_allT': Y_test_allT, 'sampleSize': sampleSize,
                                      'burnInCount': burnInCount, 'topology': topology, 'obsrate': obsrate,
                                      'obsCount': obsCount, 'evidencepath': evidencepath,
                                      'meanPredictionPath': meanPredictionPath, 'varPredictionPath': varPredictionPath,
                                      'errorpath': errorpath})

    print 'Tasks for parallel computation were created.'
    pool = mp.Pool(processes=utils.properties.numParallelThreads)
    print 'Tasks in parallel are being started.'
    pool.map(trialFuncStar, parameterList)
    # trialFuncStar(parameterList[0])

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
              topology, obsrate, obsCount, evidencepath, meanPredictionPath, varPredictionPath, errorpath,
              sensormeans=None):
    print 'obsrate {} trial {}'.format(obsrate, trial)
    evidMat = np.zeros(shape=(prediction_model.rvCount, T), dtype=np.bool_)
    selectionStrategy = StrategyFactory.generate_selection_strategy(selection_strategy_name, seed=trial,
                                                                    pool=prediction_model.sortedids,
                                                                    parentDict=prediction_model.parentDict,
                                                                    childDict=prediction_model.childDict,
                                                                    cpdParams=prediction_model.cpdParams,
                                                                    rvCount=prediction_model.rvCount)
    meanPredResults = np.empty(shape=(prediction_model.rvCount, T))
    varPredResults = np.empty(shape=(prediction_model.rvCount, T))
    errResults = np.empty(shape=(T, 6))
    for t in range(T):
        testMat = testset[:, :t+1]
        selectees = selectionStrategy.choices(count_selectees=obsCount, evidMat=evidMat, t=t,
                                              predictionModel=prediction_model, testMat=testMat, sampleSize=sampleSize,
                                              burnInCount=burnInCount, tWin=tWin)
        # selectees = selectionStrategy.choices(count_selectees=obsCount, predictionModel=prediction_model, t=t,
        #                                       testMat=testMat, evidMat=evidMat, sampleSize=sampleSize,
        #                                       burnInCount=burnInCount, startupVals=startupVals)
        evidMat[selectees, t] = True
        curEvidMat = evidMat[:, :t+1]
        # if sensormeans is not None:
        #     startupVals = np.repeat(sensormeans.reshape(-1, 1), testMat.shape[1], axis=1)
        # else:
        #     startupVals = None
        Ymeanpred, Yvarpred = prediction_model.predict(testMat, curEvidMat, tWin=tWin, sampleSize=sampleSize,
                                                       burnInCount=burnInCount, t=t)
                                                       # obsrate=obsrate, trial=trial, startupVals=startupVals)
        meanPredResults[:, t] = Ymeanpred[:, -1]
        varPredResults[:, t] = Yvarpred[:, -1]
        errResults[t, 0] = prediction_model.compute_mean_absolute_error(testMat[:, -1], Ymeanpred[:, -1],
                                                                        type_=0, evidence_mat=curEvidMat[:, -1])
        errResults[t, 1] = prediction_model.compute_mean_squared_error(testMat[:, -1], Ymeanpred[:, -1],
                                                                       type_=0, evidence_mat=curEvidMat[:, -1])
        errResults[t, 2] = prediction_model.compute_mean_absolute_error(testMat[:, -1], Ymeanpred[:, -1],
                                                                        type_=1, evidence_mat=curEvidMat[:, -1])
        errResults[t, 3] = prediction_model.compute_mean_squared_error(testMat[:, -1], Ymeanpred[:, -1],
                                                                       type_=1, evidence_mat=curEvidMat[:, -1])
        errResults[t, 4] = prediction_model.compute_mean_absolute_error(testMat[:, -1], Ymeanpred[:, -1],
                                                                        type_=2, evidence_mat=curEvidMat[:, -1])
        errResults[t, 5] = prediction_model.compute_mean_squared_error(testMat[:, -1], Ymeanpred[:, -1],
                                                                       type_=2, evidence_mat=curEvidMat[:, -1])
    np.savetxt(evidencepath +
               '{}_activeInf_model={}_T={}_trial={}_obsrate={}.csv'.
               format('evidMat', utils.properties.prediction_model, T, trial, obsrate),
               evidMat, delimiter=',')
    np.savetxt(meanPredictionPath +
               '{}_activeInf_model={}_T={}_trial={}_obsRate={}.csv'.
               format('predResults', utils.properties.prediction_model, T, trial, obsrate),
               meanPredResults, delimiter=',')
    np.savetxt(varPredictionPath +
               '{}_activeInf_model={}_T={}_trial={}_obsRate={}.csv'.
               format('predResults', utils.properties.prediction_model, T, trial, obsrate),
               varPredResults, delimiter=',')
    np.savetxt(errorpath +
               '{}_activeInfo_model={}_topology={}_window={}_T={}_obsRate={}_trial={}.csv'.
               format('mae', utils.properties.prediction_model, topology, tWin, T, obsrate, trial),
               errResults, delimiter=',')


def computeErrorIndependently():
    # rvSet = range(0, 50)
    rvSet = range(50, 100)
    T = utils.properties.timeSpan
    tWin = utils.properties.tWin
    topology = utils.properties.dbn_topology
    numTrials = utils.properties.numTrials
    trainset, testset = DataProvider.provide_data()
    predictionModel = MultivariateDiscreteKalmanFilter()
    for obsrate in utils.properties.obsrateList:
        evidencepath = utils.properties.outputDirPath + str(obsrate) + '/evidences/'
        predictionpath = utils.properties.outputDirPath + str(obsrate) + '/predictions/'
        errorpath = utils.properties.outputDirPath + str(obsrate) + '/errors/'
        if 0.0 == obsrate:
            currentNumTrials = 1
        else:
            currentNumTrials = numTrials
        errResults = np.empty(shape=(currentNumTrials, T, 3))
        for trial in range(currentNumTrials):
            evidMat = np.loadtxt(evidencepath + '{}_activeInf_model={}_T={}_trial={}_obsrate={}.csv'.
                                 format('evidMat', utils.properties.prediction_model, T, trial, obsrate),
                                 delimiter=',').astype(np.bool_)
            predResults = np.loadtxt(predictionpath + 'mean/{}_activeInf_model={}_T={}_trial={}_obsRate={}.csv'.
                                     format('predResults', utils.properties.prediction_model, T, trial, obsrate),
                                     delimiter=',')
            for t in range(T):
                errResults[trial, t, 0] = predictionModel.compute_mean_absolute_error(testset[rvSet, t],
                                                                                      predResults[rvSet, t], type_=0,
                                                                                      evidence_mat=evidMat[rvSet, t])
                errResults[trial, t, 1] = predictionModel.compute_mean_absolute_error(testset[rvSet, t],
                                                                                      predResults[rvSet, t], type_=1,
                                                                                      evidence_mat=evidMat[rvSet, t])
                errResults[trial, t, 2] = predictionModel.compute_mean_absolute_error(testset[rvSet, t],
                                                                                      predResults[rvSet, t], type_=2,
                                                                                      evidence_mat=evidMat[rvSet, t])
            np.savetxt(errorpath +
                       '{}_activeInfo_model={}_topology={}_window={}_T={}_obsRate={}_trial={}.csv'.
                       format('mae_humid', utils.properties.prediction_model, topology, tWin, T, obsrate,
                              trial), errResults[trial], delimiter=',')
        np.savetxt(errorpath +
                   '{}_activeInf_model={}_topology={}_window={}_T={}_obsRate={}_trial={}.csv'.
                   format('meanMAE_humid', utils.properties.prediction_model, topology,tWin,T,obsrate, 'mean'),
                   np.mean(errResults, axis=0), delimiter=',')
        np.savetxt(errorpath +
                   '{}_activeInf_model={}_topology={}_window={}_T={}_obsRate={}_trial={}.csv'.
                   format('stderrMAE_humid', utils.properties.prediction_model, topology, tWin, T, obsrate,
                          'mean'), standard_error(errResults, axis=0), delimiter=',')
