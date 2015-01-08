'''
Created on Jan 7, 2015

@author: ckomurlu
'''

import numpy as np
import cPickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from utils.readdata import convert_digitized_to_feature_matrix

# sorted_mat, bin_feature_mat = convert_digitized_to_feature_matrix(
#             to_be_pickled=True)
sorted_mat = cPickle.load(open('../data/sorted_mat.pickle','rb'))
bin_feature_mat = cPickle.load(open('../data/bin_feature_mat.pickle','rb'))

trainDays = [2,3,4]
testDays = [5,6]
X_train = np.empty((0,4))
Y_train = np.empty((0,1))
X_test = np.empty((0,4))
Y_test = np.empty((0,1))
for i in trainDays:
    X_train = np.append(X_train,bin_feature_mat[np.where((\
            [i*48 <= x < (i+1)*48 for x in sorted_mat[:,0]]))],axis=0)
    Y_train = np.append(Y_train,sorted_mat[np.where((\
            [i*48 <= x < (i+1)*48 for x in sorted_mat[:,0]])),1])
for i in testDays:
    X_test = np.append(X_test,bin_feature_mat[np.where((\
            [i*48 <= x < (i+1)*48 for x in sorted_mat[:,0]]))],axis=0)
    Y_test = np.append(Y_test,sorted_mat[np.where((\
            [i*48 <= x < (i+1)*48 for x in sorted_mat[:,0]])),1])

C = [10**i for i in range(-3,7)]
results = dict()
for c in C:
    print c
    ## Test accuracy
    lr = LogisticRegression(C=c)
    lr.fit(X_train,Y_train)
    Y_pred = lr.predict(X_test)
    testAcc = accuracy_score(Y_test, Y_pred)
    ## Train accuracy
    lr = LogisticRegression(C=c)
    lr.fit(X_train,Y_train)
    Y_pred2 = lr.predict(X_train)
    trainAcc = accuracy_score(Y_train, Y_pred2)
    results[c] = (trainAcc,testAcc)    
print 'C\ttrain\t\ttest'
print '======================================'
# for key,value in results.items():
for c in C:
    print c,'\t',results[c][0],'\t',results[c][1]