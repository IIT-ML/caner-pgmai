import numpy as np
import pickle

from models.ml_reg_model import MLRegModel


class LinearChain(MLRegModel):
    def __init__(self):
        super(LinearChain, self).__init__()
        self.rvCount = 0
        self.sortedids = np.array([], dtype=np.int_)
        self.cpdParams = list()

    def fit(self, train_mat, **kwargs):
        self.rvCount = train_mat.shape[0]
        self.sortedids = range(self.rvCount)
        self.__loadCpdParams()

    def __loadCpdParams(self):
        mu, initialSigmasq = pickle.load(open(
                'C:\\Users\\ckomurlu\\Documents\\workbench\\experiments\\20160301\\initialSliceParameters.pkl', 'rb'
        ))
        beta, sigmasq = pickle.load(open(
                'C:\\Users\\ckomurlu\\Documents\\workbench\\experiments\\20160301\\linearChainParameters.pkl', 'rb'
        ))
        for sensor in self.sortedids:
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
                                               ytest[sensor, t-1] * self.cpdParams[sensor][1][1]
                    predictedVariance[sensor, t] = predictedVariance[sensor, t-1] *\
                                                   (self.cpdParams[sensor][1][1] ** 2) + self.cpdParams[sensor][1][2]
        return predictedMean  #, predictedVariance

    def predict_backup(self, testMat, evidMat, **kwargs):
        ytest = np.vectorize(lambda x: x.true_label)(testMat)
        predictedMean = np.empty(shape=self.rvCount)
        predictedVariance = np.empty(shape=self.rvCount)
        for sensor in self.sortedids:
            deltaT = 0
            initialMean = self.cpdParams[sensor][0][0]
            initialVariance = self.cpdParams[sensor][0][2]
            for t in range(evidMat.shape[1] - 1, -1, -1):
                if evidMat[sensor, t]:
                    initialMean = ytest[sensor, t]
                    initialVariance = 0
                    break
                deltaT += 1
            currentMean = initialMean
            currentVariance = initialVariance
            for t in range(deltaT):
                currentMean = self.cpdParams[sensor][1][0] + currentMean * self.cpdParams[sensor][1][1]
                currentVariance = currentVariance * (self.cpdParams[sensor][1][1] ** 2) + self.cpdParams[sensor][1][2]
            predictedMean[sensor] = currentMean
            predictedVariance[sensor] = currentVariance
        return predictedMean.reshape(-1, 1)

    def compute_accuracy(self, Y_test, Y_pred):
        raise NotImplementedError

    def compute_confusion_matrix(self, Y_test, Y_pred):
        raise NotImplementedError

#
# lc = LinearChain()
# lc.fit
# lc.loadCpdParams()
