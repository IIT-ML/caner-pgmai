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
from ai.selection_strategy import RandomStrategy

def extZero(dim,pos,arr):
    arrDim = arr.shape
    assert(arrDim[0] + pos[0] + 1 <= dim[0])
    assert(arrDim[1] + pos[1] + 1 <= dim[1])
    zeroMat = np.zeros(dim)
    zeroMat[pos[0]:pos[0]+arrDim[0],pos[1]:pos[1]+arrDim[1]] = arr
    return zeroMat

def classify_noai():
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
    cmResults = dict() #confusion matrix results
    for c in C:
        for sensor in sensor_IDs:
            lr = LogisticRegression(C=c)
            lr.fit(X_train_dict[sensor],Y_train_dict[sensor])
            Y_pred = lr.predict(X_train_dict[sensor])
            cmResults['train',c,sensor] = confusion_matrix(Y_train_dict[sensor], Y_pred)
            Y_pred = lr.predict(X_test_dict[sensor])
            cmResults['test',c,sensor] = confusion_matrix(Y_test_dict[sensor], Y_pred)
            
    cmDf = pd.DataFrame(cmResults.values(), index=pd.MultiIndex.from_tuples(
                    cmResults.keys(), names=['dataset', 'C', 'sensor']))
    
    cmDf.reset_index(level=2, inplace=True)
    cmDf.reset_index(level=1, inplace=True)
    cmDf.reset_index(level=0, inplace=True)
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
    for c in C:
        print c,'\t',aggAccTrain[c],'\t',aggAccTest[c]

def classify_ai_random():
    # one_sensor_data_dict,bin_feature_dict = partition_feature_mat_into_sensors(
    #         to_be_pickled=True)
    one_sensor_data_dict,bin_feature_dict = cPickle.load(
        open('../data/one_sensor_data.pickle','rb'))
    trainDays = [2,3,4]
    testDays = [5,6]
    sensor_IDs = np.array(one_sensor_data_dict.keys())
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
    cmResults = dict() #confusion matrix results
    for c in C:
        random_strategy = RandomStrategy()
        k = 480
#         pool = np.arange(Y_test.shape[0])
        pool = 4800
        selection_list = random_strategy.choices(pool, k)
        bins = np.arange(0,4896,96)
        selected_sensors = np.digitize(selection_list,bins) - 1
        inner_indices = selection_list % 96
        pool_by_sensor_dict = dict()
        for sens in sensor_IDs[np.unique(selected_sensors)]:
            pool_by_sensor_dict[sens] = list()
        for i in range(k):
            pool_by_sensor_dict[sensor_IDs[selected_sensors[i]]].append(inner_indices[i])
        for sensor in sensor_IDs:
            lr = LogisticRegression(C=c)
            lr.fit(X_train_dict[sensor],Y_train_dict[sensor])
            Y_pred = lr.predict(X_train_dict[sensor])            
            cmResults['train',c,sensor] = confusion_matrix(Y_train_dict[sensor], Y_pred)
            Y_pred = lr.predict(X_test_dict[sensor])
            for i in pool_by_sensor_dict[sensor]:
                Y_pred[i] = Y_test_dict[sensor][i]
            cmResults['test',c,sensor] = confusion_matrix(Y_test_dict[sensor], Y_pred)
#         for i in range(k):
#             cmResults['test',c,sensor] = 
    cmDf = pd.DataFrame(cmResults.values(), index=pd.MultiIndex.from_tuples(
                    cmResults.keys(), names=['dataset', 'C', 'sensor']))
    
    cmDf.reset_index(level=2, inplace=True)
    cmDf.reset_index(level=1, inplace=True)
    cmDf.reset_index(level=0, inplace=True)
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
    for c in C:
        print c,'\t',aggAccTrain[c],'\t',aggAccTest[c]

        
# classify_noai()
classify_ai_random()