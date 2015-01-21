'''
Created on Jan 19, 2015

@author: ckomurlu
'''

from utils.readdata import convert_time_window_df_randomvar
from joint.iterative_classifier import ICAModel

def main():
    train_set,test_set = convert_time_window_df_randomvar(True)
    icaModel = ICAModel(use_local_features=True)
    icaModel.fit(train_set)
    Y_pred = icaModel.predict(test_set, maxiter=3)
    print icaModel.compute_accuracy(test_set, Y_pred)
    print icaModel.compute_confusion_matrix(test_set, Y_pred)

main()