'''
Created on Jan 19, 2015

@author: ckomurlu
'''

from utils.readdata import convert_time_window_df_randomvar
from joint.iterative_classifier import ICAModel
from joint.iterative_regressor import IRAModel
from ai.selection_strategy import RandomStrategy,UNCSampling
from utils.node import Neighborhood

import numpy as np
import cPickle
from time import time
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.preprocessing import PolynomialFeatures
from models.LinearRegressionExt import LinearRegressionExt

def main_IRA():
    begin = time()
    neighborhood_def = Neighborhood.itself_previous_others_current
    train_set,test_set = convert_time_window_df_randomvar(True,
                                                          neighborhood_def)
    use_local_features=True
#     local_regressor_name = 'lasso'
#     relat_regressor_name = 'lasso'
#     print 'use local feature\t', 'Yes' if use_local_features else 'No'
#     print 'local regression algorithm: ',local_regressor_name
#     print 'relational regression algorithm: ',relat_regressor_name
#      = ['lineReg','lasso','ridge','svr']
    regressor_list = list()

#     degree = 1
#     reg = LinearRegressionExt(degree=degree)
#     tag = 'Linear Regression'
#     regressor_list.append((tag,reg))

    for kern in ['linear']:
        for C in [10**pwr for pwr in range(-2,5)]:
            reg = SVR(kernel=kern,C=C)
            tag = 'SVR kernel=' + kern + ' C=' + str(C)
            regressor_list.append((tag,reg))

#     for alpha in [10**pwr for pwr in range(-7,5)]:
#         reg = Lasso(alpha=alpha)
#         tag = 'Lasso alpha=' + str(alpha)
#         regressor_list.append((tag,reg))

#     for alpha in [10**pwr for pwr in range(-2,5)]:
#         reg = Ridge(alpha=alpha)
#         tag = 'Ridge alpha=' + str(alpha)
#         regressor_list.append((tag,reg))

    for tag,regressor in regressor_list:
        print tag,'\t',
        iraModel = IRAModel(local_regressor=regressor,
                            relat_regressor=regressor,
                            use_local_features=use_local_features)
        iraModel.fit(train_set)
        Y_pred = iraModel.predict_with_neighbors_true_labels(train_set)
        print iraModel.compute_mean_absolute_error(train_set, Y_pred, type_=0),'\t',
        Y_pred = iraModel.predict_with_neighbors_true_labels(test_set)
        print iraModel.compute_mean_absolute_error(test_set, Y_pred, type_=0)
    print

def main_classify():
    begin = time()
    neighborhood_def = Neighborhood.all_others_current_time
    train_set,test_set = convert_time_window_df_randomvar(True,
                                                          neighborhood_def)
    use_local_features=True
    use_current_time=True
    is_relat_feature_binary=False
    local_classifier_name = 'ridge'
    relat_classifier_name = 'ridge'
    print 'immediate update\t', 'No'
    print 'use local feature\t', 'Yes' if use_local_features else 'No'
    print 'previous time/current time\t', 'C' if use_current_time else 'P'
    print 'relative features\t', 'binary' if is_relat_feature_binary else 'ordinal' 
#     random_strategy = RandomStrategy()
    unc_sampling = UNCSampling()
    pool = [(i,j) for i in range(test_set.shape[0])
            for j in range(test_set.shape[1])]
    iter_count = 10
    num_trials = 10
    rate_range = np.arange(0,1.1,0.1)
    results = np.empty(shape=(num_trials,rate_range.shape[0],3))
    C = 10
    for current_trial in range(num_trials):
        print '\ttrial:',current_trial
        active_inf_loop(local_classifier_name,
                    relat_classifier_name,
                    use_local_features,
                    use_current_time,
                    C,
                    is_relat_feature_binary,
                    train_set,
                    test_set, pool,
                    results,
                    unc_sampling,
                    rate_range,
                    iter_count,
                    current_trial)
    cPickle.dump(results,open(
        'dataDays2_3_4-5_6/resultsUNCSampling_maxProb.pickle','wb'))
    print 'Duration: ',time() - begin
    print 'Process ended'

def active_inf_loop(local_classifier_name,
                    relat_classifier_name,
                    use_local_features,
                    use_current_time,
                    C,
                    is_relat_feature_binary,
                    train_set,
                    test_set, pool,
                    results,
                    selection_strategy,
                    rate_range,
                    iter_count,
                    current_trial):
    selectees = []
    initial_pool_size = len(pool)
    icaModel = ICAModel(local_classifier_name=local_classifier_name,
                        relat_classifier_name=relat_classifier_name,
                        use_local_features=use_local_features,
                        use_current_time=use_current_time,
                        relat_classifier_C=C,
                        is_relat_feature_binary=is_relat_feature_binary)
    icaModel.fit(train_set)
    for rate_idx in range(rate_range.shape[0]):
        if rate_idx == 0:
            rate_increment = 0
        else:
            rate_increment = int(round(initial_pool_size*
                            (rate_range[rate_idx] - rate_range[rate_idx-1])))
        print rate_idx, rate_increment
        current_selectees,pool = selection_strategy.choices(icaModel, test_set,
                                                pool, rate_increment)
        selectees += current_selectees
        evidence_mat = apply_func_to_coords(selectees, test_set.shape)
#         Y_pred = icaModel.predict_by_local_classifiers(test_set)
        Y_pred = icaModel.predict(test_set, maxiter=iter_count,
                                        evidence_mat=evidence_mat)
        results[current_trial,rate_idx,0] = icaModel.compute_accuracy(
                        test_set, Y_pred, type_=0,evidence_mat=evidence_mat)
        results[current_trial,rate_idx,1] = icaModel.compute_accuracy(
                        test_set, Y_pred, type_=1,evidence_mat=evidence_mat)
        results[current_trial,rate_idx,2] = icaModel.compute_accuracy(
                        test_set, Y_pred, type_=2,evidence_mat=evidence_mat)

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


main_IRA()

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