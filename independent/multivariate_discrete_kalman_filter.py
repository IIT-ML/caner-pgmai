from pykalman import KalmanFilter

from models.ml_reg_model import MLRegModel

import numpy as np
import math

class MultivariateDiscreteKalmanFilter(MLRegModel):
    def __init__(self):
        super(MultivariateDiscreteKalmanFilter, self).__init__()
        self.kf = []  # np.array([], dtype=object)
        self.sortedids = np.array([], dtype=np.int_)
        self.observations = np.array([[]], dtype=np.float_)
        self.rvCount = 0
        self.initial_state_means = np.array([])
        self.initial_state_covariances = np.array([])

    def fit(self, train_mat, **kwargs):
        self.rvCount = train_mat.shape[0]
        self.sortedids = range(self.rvCount)
        self.observations = np.empty(shape=train_mat.shape, dtype=np.float_)
        self.initial_state_means = np.empty(shape=(self.rvCount,), dtype=np.float_)
        self.initial_state_covariances = np.empty(shape=(self.rvCount,), dtype=np.float_)
        for sensorid in self.sortedids:
            ytrain = np.vectorize(lambda instance: instance.true_label)(train_mat[sensorid])
            dataMeans = np.mean(ytrain.reshape(-1, 48).T, axis=1)
            self.observations[sensorid] = np.tile(dataMeans, 3)
            # self.observations[sensorid] = np.vectorize(lambda instance: instance.local_feature_vector)(
            #         train_mat[sensorid])
            # self.observations[sensorid] = np.tile(np.arange(0.0, 2 * math.pi, 2 * math.pi / 48), 3)
            data = np.vstack((ytrain[:-1], ytrain[1:], self.observations[sensorid, 1:]))
            cpdParams = self.__getCpdParams(data)
            transition_matrix = cpdParams[1][1]
            observation_matrix = 0  # cpdParams[2][1]
            transition_offset = cpdParams[1][0][0, 0]
            observation_offset = cpdParams[2][0][0, 0]
            transition_covariance = cpdParams[1][2]
            observation_covariance = cpdParams[2][2]
            initial_state_covariance = cpdParams[0][2]
            initial_state_mean = ytrain[-1]
            self.kf.append(KalmanFilter(transition_matrix, observation_matrix, transition_covariance,
                                        observation_covariance, transition_offset, observation_offset,
                                        initial_state_mean, initial_state_covariance))
            self.initial_state_means[sensorid] = initial_state_mean
            self.initial_state_covariances[sensorid] = initial_state_covariance

    def predict(self, test_mat, evid_mat, **kwargs):
        xtest = np.vectorize(lambda x: x.local_feature_vector)(test_mat)
        ytest = np.vectorize(lambda x: x.true_label)(test_mat)
        n_time_steps = test_mat.shape[1]
        ypred = np.zeros((self.rvCount, n_time_steps), dtype=np.float_)
        covpredict = np.zeros((self.rvCount, n_time_steps), dtype=np.float_)
        for sensorid in self.sortedids:
            ypred[sensorid, 0], covpredict[sensorid, 0] = (
                    self.kf[sensorid].filter_update(self.initial_state_means[sensorid],
                                                    self.initial_state_covariances[sensorid],
                                                    self.observations[sensorid, -1]))
            for t in range(n_time_steps-1):
                ypred[sensorid, t+1], covpredict[sensorid, t+1] = (
                    self.kf[sensorid].filter_update(ypred[sensorid, t], covpredict[sensorid, t],
                                                    self.observations[sensorid, t+1]))
                if evid_mat[sensorid, t+1]:
                    ypred[sensorid, t+1] = ytest[sensorid, t+1]
        return ypred


    def compute_accuracy(self, Y_test, Y_pred):
        raise NotImplementedError

    def compute_confusion_matrix(self, Y_test, Y_pred):
        raise NotImplementedError

    @staticmethod
    def __computeCondGauss(sensid, parentDict, mea, cova, initial=False):
        parents = parentDict[sensid]
        if initial:
            parents = parents[:-1]
            if parents == []:
                return mea[sensid],np.array([]), cova[sensid, sensid]
        firstInd = np.tile(tuple(parents), len(parents))
        secondInd = np.repeat(tuple(parents), len(parents))
        YY = cova[sensid, sensid].reshape(1, 1)
        YX = cova[sensid, tuple(parents)].reshape(1, -1)
        XY = cova[tuple(parents), sensid].reshape(-1, 1)
        XXinv = np.linalg.inv(cova[firstInd, secondInd].reshape(len(parents),
                                                                len(parents)))
        b0 = mea[sensid].reshape(1, 1) - np.dot(np.dot(YX, XXinv),
                                            mea[list(parents)].reshape(-1, 1))
        b = np.dot(XXinv, XY)
        sigsq = YY - np.dot(np.dot(YX,XXinv),XY.reshape(-1, 1))
        return b0, b, sigsq[0]

    def __getCpdParams(self, a):
        parentDict = {0: [], 1: [0], 2: [1]}
        mu = np.mean(a,axis=1)
        cova = np.cov(a)
        cpdParams = list()
        cpdParams.append(MultivariateDiscreteKalmanFilter.__computeCondGauss(0, parentDict, mu, cova, True))
        cpdParams.append(MultivariateDiscreteKalmanFilter.__computeCondGauss(1, parentDict, mu, cova))
        cpdParams.append(MultivariateDiscreteKalmanFilter.__computeCondGauss(2, parentDict, mu, cova))
        return cpdParams