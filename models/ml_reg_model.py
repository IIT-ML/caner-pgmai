'''
Created on Mar 14, 2015

@author: ckomurlu
'''

from models.ml_model import MLModel
from utils import constants

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import copy
from abc import ABCMeta

class MLRegModel(MLModel):
    __metaclass__ = ABCMeta
    '''
    classdocs
    '''

    def __init__(self, **kwargs):
        self.rvCount = None
        self.sortedids = None
        self.parentDict = None
        self.childDict = None
        self.cpdParams = None

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

    def computeVar(self):
        raise NotImplementedError

    def compute_mean_absolute_error(self,test_set,Y_pred, type_=0, evidence_mat=None):
        # type_ 0 is the accuracy in which we include everyone with their
        #     predicted labels
        # type_ 1 is the accuracy in which we include only the non evidence
        #     instances
        # type_ 2 is the one we include again everyone but evidence instances
        #     participate with true labels.
        Y_pred_local = copy.deepcopy(Y_pred)
        Y_true,Y_pred_local,legit_indices = self.__prepare_prediction_for_evaluation(
                    test_set,Y_pred_local, type_=type_,
                    evidence_mat=evidence_mat)
        return mean_absolute_error(np.ravel(Y_true[legit_indices]),
                                  np.ravel(Y_pred_local[legit_indices]))
    
    def compute_mean_squared_error(self,test_set,Y_pred, type_=0, evidence_mat=None):
        # type_ 0 is the accuracy in which we include everyone with their
        #     predicted labels
        # type_ 1 is the accuracy in which we include only the non evidence
        #     instances
        # type_ 2 is the one we include again everyone but evidence instances
        #     participate with true labels.
        Y_pred_local = copy.deepcopy(Y_pred)
        Y_true,Y_pred_local,legit_indices = self.__prepare_prediction_for_evaluation(
                    test_set,Y_pred_local, type_=type_,
                    evidence_mat=evidence_mat)
        return mean_squared_error(np.ravel(Y_true[legit_indices]),
                                  np.ravel(Y_pred_local[legit_indices]))
        