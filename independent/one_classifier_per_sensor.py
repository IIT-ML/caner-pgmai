'''
Created on Jan 7, 2015

@author: ckomurlu
'''
import numpy as np
import pandas as pd
import cPickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from utils.readdata import partition_feature_mat_into_sensors

def extZero(dim,pos,arr):
    arrDim = arr.shape
    assert(arrDim[0] + pos[0] + 1 <= dim[0])
    assert(arrDim[1] + pos[1] + 1 <= dim[1])
    zeroMat = np.zeros(dim)
    zeroMat[pos[0]:pos[0]+arrDim[0],pos[1]:pos[1]+arrDim[1]] = arr
    return zeroMat

# one_sensor_data_dict,bin_feature_dict = partition_feature_mat_into_sensors(
#         to_be_pickled=True)
one_sensor_data_dict,bin_feature_dict = cPickle.load(
    open('../data/one_sensor_data.pickle','rb'))
trainDays = [2,3,4]
testDays = [5,6]
sensor_IDs = one_sensor_data_dict.keys()
X_train_dict = dict()
Y_train_dict = dict()
X_test_dict = dict()
Y_test_dict = dict()
for current_sensor_ID in sensor_IDs:
    X_train = np.empty((0,4))
    Y_train = np.empty((0,1))
    for i in trainDays:
        X_train = np.append(X_train,bin_feature_dict[current_sensor_ID]
                            [np.where(([i*48 <= x < (i+1)*48 for x in
                            one_sensor_data_dict[current_sensor_ID]
                            [:,0]]))],axis=0)
        Y_train = np.append(Y_train,one_sensor_data_dict[current_sensor_ID]
                            [np.where(([i*48 <= x < (i+1)*48 for x in 
                            one_sensor_data_dict[current_sensor_ID][:,0]])),1])
    X_train_dict[current_sensor_ID] = X_train
    Y_train_dict[current_sensor_ID] = Y_train
    X_test = np.empty((0,4))
    Y_test = np.empty((0,1))
    for i in testDays:
        X_test = np.append(X_test,bin_feature_dict[current_sensor_ID]
                            [np.where(([i*48 <= x < (i+1)*48 for x in
                            one_sensor_data_dict[current_sensor_ID]
                            [:,0]]))],axis=0)
        Y_test = np.append(Y_test,one_sensor_data_dict[current_sensor_ID]
                            [np.where(([i*48 <= x < (i+1)*48 for x in 
                            one_sensor_data_dict[current_sensor_ID][:,0]])),1])
    X_test_dict[current_sensor_ID] = X_test
    Y_test_dict[current_sensor_ID] = Y_test
    

C = [10**i for i in range(-3,7)]
accResults = dict() #accuracy results
cmResults = dict() #confusion matrix results
for c in C:
    for sensor in sensor_IDs:
        lr = LogisticRegression(C=c)
        lr.fit(X_train_dict[sensor],Y_train_dict[sensor])
        Y_pred = lr.predict(X_train_dict[sensor])
        accResults['train',c,sensor] = accuracy_score(Y_train_dict[sensor], Y_pred)
        cmResults['train',c,sensor] = confusion_matrix(Y_train_dict[sensor], Y_pred)
        Y_pred = lr.predict(X_test_dict[sensor])
        accResults['test',c,sensor] = accuracy_score(Y_test_dict[sensor], Y_pred)  
        cmResults['test',c,sensor] = confusion_matrix(Y_test_dict[sensor], Y_pred)
        
accDf = pd.DataFrame(accResults.values(), index=pd.MultiIndex.from_tuples(
                accResults.keys(), names=['dataset', 'C', 'sensor']))
cmDf = pd.DataFrame(cmResults.values(), index=pd.MultiIndex.from_tuples(
                cmResults.keys(), names=['dataset', 'C', 'sensor']))

accDf.reset_index(level=2, inplace=True)
accDf.reset_index(level=1, inplace=True)
accDf.reset_index(level=0, inplace=True)
cmDf.reset_index(level=2, inplace=True)
cmDf.reset_index(level=1, inplace=True)
cmDf.reset_index(level=0, inplace=True)
accDf.rename(columns={0: 'outcome'}, inplace=True)
cmDf.rename(columns={0: 'outcome'}, inplace=True)
aggCm = dict()

dataset = 'train'
for c in C:
    aggCm[c] = np.zeros((4,4))
    for row in cmDf['outcome'][(cmDf.dataset == dataset)&(cmDf.C == c)]:
        currentCm = row
        if currentCm.shape != (4L,4L):
            currentCm = extZero((4,4),(0,0),currentCm)
        aggCm[c] += currentCm
aggAccTrain = dict()
for key in aggCm.keys():
    aggAccTrain[key] = np.sum(aggCm[key][(0,1,2,3),(0,1,2,3)])/np.sum(aggCm[key])


aggCm = dict()
dataset = 'test'
for c in C:
    aggCm[c] = np.zeros((4,4))
    for row in cmDf['outcome'][(cmDf.dataset == dataset)&(cmDf.C == c)]:
        currentCm = row
        if currentCm.shape != (4L,4L):
            currentCm = extZero((4,4),(0,0),currentCm)
        aggCm[c] += currentCm
aggAccTest = dict()
for key in aggCm.keys():
    aggAccTest[key] = np.sum(aggCm[key][(0,1,2,3),(0,1,2,3)])/np.sum(aggCm[key])
print 'C\ttrain\t\ttest'
print '======================================'
# for key,value in results.items():
for c in C:
    print c,'\t',aggAccTrain[c],'\t',aggAccTest[c]