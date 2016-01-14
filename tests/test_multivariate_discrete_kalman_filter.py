from data.data_provider import DataProvider
from independent.multivariate_discrete_kalman_filter import MultivariateDiscreteKalmanFilter
import numpy as np


def generateEvidVec(rvCount, timeStampCount, observationRate):
    timeStampIndices = np.arange(1, timeStampCount, dtype=np.int_)
    evidVec = np.zeros(shape=(timeStampCount,), dtype=np.bool_)
    evidVec[0] = True
    if observationRate == 0:
        return evidVec
    else:
        rs = np.random.RandomState(seed=0)
        rs.shuffle(timeStampIndices)
        countOfObserved = np.round(timeStampCount * observationRate) - 1
        selectees = timeStampIndices[:countOfObserved]
        evidVec[selectees] = True
        return evidVec

train_mat, test_mat = DataProvider.provide_data()

mdkf = MultivariateDiscreteKalmanFilter()
mdkf.fit(train_mat)
evid_mat = np.zeros(shape=test_mat.shape, dtype=np.bool_)
t = 10
ypred = mdkf.predict(test_mat[:, :t], evid_mat[:, :t])
print 'End of test'