import numpy as np
import pickle
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

from models.ml_reg_model import MLRegModel
from models.LinearRegressionExt import LinearRegressionExt


class LinearChain(MLRegModel):
    def __init__(self, regressionMethod='linear'):
        super(LinearChain, self).__init__()
        self.regressionMethod = regressionMethod
        self.rvCount = 0
        self.sortedids = np.array([], dtype=np.int_)
        self.cpdParams = list()

    def fit(self, trainset, **kwargs):
        self.rvCount = trainset.shape[0]
        self.sortedids = range(self.rvCount)
        self.__computeCpdParams(trainset)

    def fitFromFile(self, **kwargs):
        self.rvCount = 50
        self.sortedids = range(self.rvCount)
        self.__loadCpdParams()

    def __loadCpdParams(self):
        mu, initialSigmasq = pickle.load(open(
                'C:\\Users\\ckomurlu\\Documents\\workbench\\experiments\\20160301\\initialSliceParameters.pkl', 'rb'
        ))
        if 'linear' == self.regressionMethod:
            beta, sigmasq = pickle.load(open(
                    'C:\\Users\\ckomurlu\\Documents\\workbench\\experiments\\20160301\\linearChainParameters.pkl', 'rb'
            ))
        elif 'ridge' == self.regressionMethod:
            beta, sigmasq = pickle.load(open(
                    'C:\\Users\\ckomurlu\\Documents\\workbench\\experiments\\20160301\\ridgeChainParameters.pkl',
                    'rb'))
        elif 'lasso' == self.regressionMethod:
            raise NotImplementedError('Lasso regression for chain parameters has not been yet implemented.')
        else:
            raise ValueError('Improper regression method name for linear chain parameters.')
        for sensor in self.sortedids:
            self.cpdParams.append([(mu[sensor], [], initialSigmasq[sensor]), (beta[sensor, 0], beta[sensor, 1],
                                                                              sigmasq[sensor])])

    def __computeCpdParams(self, trainset):
        if 'linear' == self.regressionMethod:
            regressionModel = LinearRegressionExt
        elif 'ridge' == self.regressionMethod:
            regressionModel = Ridge
        elif 'lasso' == self.regressionMethod:
            regressionModel = Lasso
        else:
            raise ValueError('Improper regression method name for linear chain parameters.')
        trainmat = np.vectorize(lambda x: x.true_label)(trainset)
        mu = np.mean(trainmat, axis=1)
        initialSigmasq = np.var(trainmat, axis=1)
        self.rvCount = trainmat.shape[0]
        beta = np.zeros(shape=(self.rvCount, 2), dtype=np.float_)
        sigmasq = np.zeros(shape=self.rvCount, dtype=np.float_)
        for sensor in range(self.rvCount):
            X = trainmat[sensor, :-1].reshape(-1, 1)
            y = trainmat[sensor, 1:].reshape(-1, 1)
            reg = regressionModel()
            reg.fit(X, y)
            beta[sensor, 0] = reg.intercept_
            beta[sensor, 1:] = reg.coef_
            ypred = reg.predict(X)
            err = y - ypred
            var = np.cov(err.reshape(-1))
            sigmasq[sensor] = var
            self.cpdParams.append([(mu[sensor], [], initialSigmasq[sensor]), (beta[sensor, 0], beta[sensor, 1],
                                                                              sigmasq[sensor])])

    def predict(self, testMat, evidMat, **kwargs):
        ytest = np.vectorize(lambda x: x.true_label)(testMat)
        T = ytest.shape[1]
        predictedMean = np.empty(shape=(self.rvCount, T))
        predictedVariance = np.empty(shape=(self.rvCount, T))
        for sensor in self.sortedids:
            if evidMat[sensor, 0]:
                    predictedMean[sensor, 0] = ytest[sensor, 0]
                    predictedVariance[sensor, 0] = 0.0
            else:
                predictedMean[sensor, 0] = self.cpdParams[sensor][0][0]
                predictedVariance[sensor, 0] = self.cpdParams[sensor][0][2]
            for t in range(1, T):
                if evidMat[sensor, t]:
                    predictedMean[sensor, t] = ytest[sensor, t]
                    predictedVariance[sensor, t] = 0.0
                else:
                    predictedMean[sensor, t] = self.cpdParams[sensor][1][0] +\
                                               predictedMean[sensor, t-1] * self.cpdParams[sensor][1][1]
                    predictedVariance[sensor, t] = predictedVariance[sensor, t-1] * \
                                                   (self.cpdParams[sensor][1][1] ** 2) + self.cpdParams[sensor][1][2]
        return predictedMean, predictedVariance

    def computeVar(self, evidMat):
        varVec = np.empty(shape=self.rvCount, dtype=np.float_)
        T = evidMat.shape[1]
        for sensor in self.sortedids:
            betasq = self.cpdParams[sensor][1][1] ** 2
            sigmasq = self.cpdParams[sensor][1][2]
            if 0 == evidMat[sensor, ::-1].nonzero()[0].size:
                initialSigmasq = self.cpdParams[sensor][0][2]
                remainingTime = T - 1
            else:
                initialSigmasq = 0
                # remainingTime = T - (evidMat[sensor, ::-1].nonzero()[0][-1] + 1)
                remainingTime = T - (T - evidMat[sensor, ::-1].nonzero()[0][-1])
            varVec[sensor] = sigmasq * (((betasq ** remainingTime) - 1) / (betasq - 1)) + \
                             initialSigmasq * (betasq ** remainingTime)
        return varVec

    # def computeVar(self, evidMat):
    #     varVec = np.empty(shape=self.rvCount, dtype=np.float_)
    #     T = evidMat.shape[1]
    #     for sensor in self.sortedids:
    #         betasq = self.cpdParams[sensor][1][1] ** 2
    #         sigmasq = self.cpdParams[sensor][1][2]
    #         try:
    #             remainingTime = T - evidMat[sensor, ::-1].nonzero()[0][-1]
    #         except IndexError:
    #             remainingTime = T
    #         varVec[sensor] = sigmasq * (((betasq ** remainingTime) - 1) / (betasq - 1))
    #     return varVec

    def compute_accuracy(self, Y_test, Y_pred):
        raise NotImplementedError

    def compute_confusion_matrix(self, Y_test, Y_pred):
        raise NotImplementedError

#
# lc = LinearChain()
# lc.fit
# lc.loadCpdParams()
