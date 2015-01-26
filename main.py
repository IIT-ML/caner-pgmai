'''
Created on Jan 19, 2015

@author: ckomurlu
'''

from utils.readdata import convert_time_window_df_randomvar
from joint.iterative_classifier import ICAModel

def main():
    train_set,test_set = convert_time_window_df_randomvar(True)
    use_local_features=True
    use_current_time=True
    icaModel = ICAModel(use_local_features=use_local_features,
                        use_current_time=use_current_time)
    icaModel.fit(train_set)
#     Y_pred = icaModel.predict(test_set, maxiter=10)
#     Y_pred = icaModel.predict_by_local_classifiers(train_set)
    Y_pred = icaModel.predict_with_neighbors_true_labels_current_time(
                                                        test_set)
    print 'immediate update\t', 'No'
    print 'use local feature\t', 'Yes' if use_local_features else 'No'
    print 'previous time/current time\t', 'C' if use_current_time else 'P'
    print icaModel.compute_accuracy(test_set[:,1:], Y_pred[:,1:])
    print icaModel.compute_confusion_matrix(test_set[:,1:], Y_pred[:,1:])

main()