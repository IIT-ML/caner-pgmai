from data.data_provider import DataProvider

import numpy as np
from sklearn.linear_model import Lasso

trainset, testset = DataProvider.provide_data()
trainmat = np.vectorize(lambda x: x.true_label)(trainset)
testmat = np.vectorize(lambda x: x.true_label)(testset)
# np.savetxt('C:/Users/ckomurlu/Documents/workbench/experiments/20160329/trainmat.csv', trainmat,
#            fmt='%.18e', delimiter=',')
# np.savetxt('C:/Users/ckomurlu/Documents/workbench/experiments/20160329/testmat.csv', testmat,
#            fmt='%.18e', delimiter=',')

regressionModel = Lasso

weights = np.empty(shape=(100, 100), dtype=np.float_)
bias = np.empty(shape=(100,), dtype=np.float_)
regModels = list()
for i in range(trainset.shape[0]):
    X = trainmat[:, :-1].T
    y = trainmat[i, 1:]
    reg = regressionModel(alpha=0.1, max_iter=10000)
    reg.fit(X, y)
    regModels.append(reg)
    weights[i] = reg.coef_
# np.savetxt('C:/Users/ckomurlu/Documents/workbench/experiments/20160329/weightsLasso_alpha5.csv', weights,
#            fmt='%.18e', delimiter=',')

predictions = np.empty(shape=(100, 144), dtype=np.float_)
for i in range(trainset.shape[0]):
    for t in range(1, 144):
        predictions[i, t] = regModels[i].predict(trainmat[:, t-1].reshape(-1,1))
        # predictions[i, t] = np.dot(weights[i], trainmat[:, t-1]) + bias[i]


pass