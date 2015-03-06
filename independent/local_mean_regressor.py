'''
Created on Feb 24, 2015

@author: ckomurlu
'''

import numpy as np

from models.ml_model import MLModel
from utils import constants
from utils.decorations import deprecated

from sklearn.metrics.metrics import mean_squared_error, mean_absolute_error

class LocalMeanRegressor(MLModel):
    def __init__(self):
        self.mean_mat = -np.ones((1,))
    
    def fit(self,train_mat):
        row_count = train_mat.shape[0]
        col_count = len(train_mat[0,0].local_feature_vector)
        self.mean_mat = np.zeros((row_count,col_count))
        time_mat = np.vectorize(lambda x: np.sum(
                        x.local_feature_vector * np.arange(1,5)))(train_mat)
        self.time_values = np.unique(time_mat)
        for current_row in range(row_count):
            for current_val_idx in range(col_count):
                current_val = self.time_values[current_val_idx]
                self.mean_mat[current_row,current_val_idx] = np.average(
                    np.vectorize(lambda x: x.true_label)(train_mat[current_row]
                    [np.where(time_mat[current_row] == current_val)]))
    
    def predict(self,test_mat):
        Y_pred = np.ones(shape=test_mat.shape,dtype=np.float_)* \
                    constants.FLOAT_INF
        row_count,col_count = test_mat.shape
        for row_idx in range(row_count):
            for col_idx in range(col_count):
                time_ = np.sum(test_mat[row_idx,col_idx].\
                               local_feature_vector*np.arange(1,5))
                time_idx = np.where(self.time_values == time_)
                Y_pred[row_idx,col_idx] = self.mean_mat[row_idx,time_idx]
        return Y_pred
    
    def __prepare_prediction_for_evaluation(self,test_set,Y_pred, type_=0,
                                            evidence_mat=None):
        assert test_set.shape == Y_pred.shape,'Test set and predicted label'+\
               ' matrix shape don\'t match'
#         Y_true = [node.true_label for row in test_set for node in row]
        if type_ == 0:
            Y_true = np.vectorize(lambda x: x.true_label)(test_set)
        else:
            assert evidence_mat is not None,'The evidence mat is ' + \
                'not given whereas type_ 1 or 2 accuracy is selected. ' + \
                'There\'s bug in the flow'
            if type_ == 1:
                if np.sum(evidence_mat) == np.prod(evidence_mat.shape):
                    Y_true = np.array([])
                    Y_pred = np.array([])
                else:
                    Y_true = np.vectorize(lambda x: x.true_label)(test_set[
                                                        np.invert(evidence_mat)])
                    Y_pred = Y_pred[np.invert(evidence_mat)]
            elif type_ == 2:
                Y_true = np.vectorize(lambda x: x.true_label)(test_set)
                Y_pred[evidence_mat] = Y_true[evidence_mat]
            else:
                raise ValueError('type_ is set a non legit value, it must ' + \
                                 'be {0,1,2}')
        legit_indices = np.where(Y_pred != constants.FLOAT_INF)
        return Y_true,Y_pred,legit_indices    
    
    def compute_mean_absolute_error(self,test_set,Y_pred, type_=0, evidence_mat=None):
        # type_ 0 is the accuracy in which we include everyone with their
        #     predicted labels
        # type_ 1 is the accuracy in which we include only the non evidence
        #     instances
        # type_ 2 is the one we include again everyone but evidence instances
        #     participate with true labels.
        Y_true,Y_pred,legit_indices = self.__prepare_prediction_for_evaluation(
                    test_set,Y_pred, type_=type_, evidence_mat=evidence_mat)
        return mean_absolute_error(np.ravel(Y_true[legit_indices]),
                                  np.ravel(Y_pred[legit_indices]))
    
    def compute_mean_squared_error(self,test_set,Y_pred, type_=0, evidence_mat=None):
        # type_ 0 is the accuracy in which we include everyone with their
        #     predicted labels
        # type_ 1 is the accuracy in which we include only the non evidence
        #     instances
        # type_ 2 is the one we include again everyone but evidence instances
        #     participate with true labels.
        Y_true,Y_pred,legit_indices = self.__prepare_prediction_for_evaluation(
                    test_set,Y_pred, type_=type_, evidence_mat=evidence_mat)
        return mean_squared_error(np.ravel(Y_true[legit_indices]),
                                  np.ravel(Y_pred[legit_indices]))
    
    @deprecated
    def compute_accuracy(self,test_set,Y_pred, type_=0, evidence_mat=None):
        # type_ 0 is the accuracy in which we include everyone with their
        #     predicted labels
        # type_ 1 is the accuracy in which we include only the non evidence
        #     instances
        # type_ 2 is the one we include again everyone but evidence instances
        #     participate with true labels.
        assert test_set.shape == Y_pred.shape,'Test set and predicted label'+\
               ' matrix shape don\'t match'
#         Y_true = [node.true_label for row in test_set for node in row]
        if type_ == 0:
            Y_true = np.vectorize(lambda x: x.true_label)(test_set)
        else:
            assert evidence_mat is not None,'The evidence mat is ' + \
                'not given whereas type_ 1 or 2 accuracy is selected. ' + \
                'There\'s bug in the flow'
            if type_ == 1:
                if np.sum(evidence_mat) == np.prod(evidence_mat.shape):
                    return float('nan')
                Y_true = np.vectorize(lambda x: x.true_label)(test_set[
                                                    np.invert(evidence_mat)])
                Y_pred = Y_pred[np.invert(evidence_mat)]
            elif type_ == 2:
                Y_true = np.vectorize(lambda x: x.true_label)(test_set)
                Y_pred[evidence_mat] = Y_true[evidence_mat]
            else:
                raise ValueError('type_ is set a non legit value, it must ' + \
                                 'be {0,1,2}')
        legit_indices = np.where(Y_pred != constants.FLOAT_INF)
#         return mean_squared_error(np.ravel(Y_true[legit_indices]),
#                                   np.ravel(Y_pred[legit_indices]))
        return mean_absolute_error(np.ravel(Y_true[legit_indices]),
                                  np.ravel(Y_pred[legit_indices]))
    
    def compute_confusion_matrix(self,Y_test,Y_pred):
        raise NotImplemented()