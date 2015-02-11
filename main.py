'''
Created on Jan 19, 2015

@author: ckomurlu
'''

from utils.readdata import convert_time_window_df_randomvar
from joint.iterative_classifier import ICAModel
from ai.selection_strategy import RandomStrategy

import numpy as np
import cPickle
from time import time


def main():
    begin = time()
    neighborhood_def = all_others_current_time
    train_set,test_set = convert_time_window_df_randomvar(True,
                                                          neighborhood_def)
    use_local_features=True
    use_current_time=True
    is_relat_feature_binary=False
    local_classifier_name = 'svm'
    relat_classifier_name = 'svm'
    print 'immediate update\t', 'No'
    print 'use local feature\t', 'Yes' if use_local_features else 'No'
    print 'previous time/current time\t', 'C' if use_current_time else 'P'
    print 'relative features\t', 'binary' if is_relat_feature_binary else 'ordinal' 
    C_values = [10]
#     C_values = [0.01, 0.02, 0.05, 0.1, 1, 10, 100, 1000, 10000]
    random_strategy = RandomStrategy()
    pool = [(i,j) for i in range(test_set.shape[0])
            for j in range(test_set.shape[1])]
#     for iter_count in xrange(11):

    iter_count = 10
    num_trials = 10
    rate_range = np.arange(0,1.1,0.1)
    
    results = np.empty(shape=(num_trials,rate_range.shape[0],3))
    
    for rate_idx in range(rate_range.shape[0]):
        rate = rate_range[rate_idx]
        print 'rate: ',rate
        k = rate * test_set.size
        for current_trial in range(num_trials):
            print '\ttrial:',current_trial
            selecteds = random_strategy.choices(pool, k)
            evidence_mat = apply_func_to_coords(selecteds, test_set.shape)
            for C in C_values:
                icaModel = ICAModel(local_classifier_name=local_classifier_name,
                                    relat_classifier_name=relat_classifier_name,
                                    use_local_features=use_local_features,
                                    use_current_time=use_current_time,
                                    relat_classifier_C=C,
                                    is_relat_feature_binary=is_relat_feature_binary)
                icaModel.fit(train_set)
#                 Y_pred = icaModel.predict(test_set, maxiter=iter_count,
#                                           evidence_mat=evidence_mat)
                Y_pred = icaModel.predict_by_local_classifiers(test_set)
#                 Y_pred = icaModel.predict_with_neighbors_true_labels(train_set)
#                 print icaModel.compute_accuracy(
#                         train_set, Y_pred, type_=0,evidence_mat=evidence_mat)
#                 print
#                 print icaModel.compute_confusion_matrix(test_set, Y_pred)
                results[current_trial,rate_idx,0] = icaModel.compute_accuracy(
                        test_set, Y_pred, type_=0,evidence_mat=evidence_mat)
                results[current_trial,rate_idx,1] = icaModel.compute_accuracy(
                        test_set, Y_pred, type_=1,evidence_mat=evidence_mat)
                results[current_trial,rate_idx,2] = icaModel.compute_accuracy(
                        test_set, Y_pred, type_=2,evidence_mat=evidence_mat)
    cPickle.dump(results,open(
        'dataDays2_3_4-5_6/resultsRandomSamplingLocalOnly.pickle','wb'))
    print 'Duration: ',time() - begin
    print 'Process ended'


#neighborhood function list
def independent_back(self_id, sensor_IDs):
    neighbors = [(self_id,-1)]
    return neighbors

def all_nodes_current_time(self_id, sensor_IDs):
    neighbors = []
    neighbors += zip(sensor_IDs,[0]*len(sensor_IDs))
    return neighbors

def all_others_current_time(self_id, sensor_IDs):
    neighbors = []
    neighbors += zip(np.setdiff1d(sensor_IDs, [self_id]),[0]*
                    (len(sensor_IDs)-1))
    return neighbors

def itself_previous_others_current(self_id, sensor_IDs):
    neighbors = [(self_id,-1)]
    neighbors += zip(np.setdiff1d(sensor_IDs, [self_id]),[0]*
                    (len(sensor_IDs)-1))
    return neighbors

def itself_current_only(self_id, sensor_IDs):
    neighbors = [(self_id,0)]
    return neighbors

def apply_func_to_coords(coord_list, shape, func=None):
    row_count, col_count = shape
    marked_mat = np.zeros(shape=(row_count,col_count),dtype=np.bool8)
    if func is None:
        for coord in coord_list:
            marked_mat[coord[0],coord[1]] = True
    else:
        for coord in coord_list:
            marked_mat[coord[0],coord[1]] = func(marked_mat[coord[0],coord[1]])
    return marked_mat

main()


#Old code:
#                 Y_pred = icaModel.predict(train_set, maxiter=10)
            #     Y_pred = icaModel.predict_by_local_classifiers(train_set)
    #             Y_pred = icaModel.predict_with_neighbors_true_labels(train_set)
    #             train_acc = icaModel.compute_accuracy(train_set, Y_pred)
            #         train_acc = icaModel.compute_accuracy(train_set, Y_pred)
    #             Y_pred = icaModel.predict_with_neighbors_true_labels(test_set)
            #         Y_pred = icaModel.predict_by_local_classifiers(test_set)

#     neighbors += zip(np.setdiff1d(sensor_IDs, [self_id]),[0]*
#                     (len(sensor_IDs)-1))

    #             print rate, '\t', C,'\t',train_acc,'\t',test_acc
#     print icaModel.compute_confusion_matrix(test_set[:,1:], Y_pred[:,1:])