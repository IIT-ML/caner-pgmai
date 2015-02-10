'''
Created on Feb 4, 2015

@author: ckomurlu
'''

from joint.iterative_classifier import ICAModel
from utils.node import RandomVarNode

import numpy as np

def main():
    rs = np.random.RandomState()
    rs.seed(seed = 0)
    Y_test = rs.randint(0,2,12).reshape(4,-1)
    print Y_test
#     Y_pred = rs.randint(0,2,12).reshape(4,-1)
    Y_pred = Y_test.copy()
    Y_pred[0] = 1 - Y_pred[0]
    Y_pred[1,:] = 1 - Y_pred[1,:]
    print Y_pred
    test_set = np.vectorize(lambda x: RandomVarNode(true_label=x))(Y_test)
    evidence_mat = np.zeros(Y_test.shape, dtype=bool)
#     evidence_mat[0] = 1 
    print np.sum(evidence_mat)
    
#     ica_model = ICAModel()
#     acc = ica_model.compute_accuracy(test_set, Y_pred, 1, evidence_mat)
#     print acc
main()