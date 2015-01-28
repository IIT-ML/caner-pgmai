'''
Created on Jan 19, 2015

@author: ckomurlu
'''

from utils.readdata import convert_time_window_df_randomvar
from joint.iterative_classifier import ICAModel

def main():
    train_set,test_set = convert_time_window_df_randomvar(True)
    use_local_features=False
    use_current_time=False
    is_relat_feature_binary=True
    print 'immediate update\t', 'No'
    print 'use local feature\t', 'Yes' if use_local_features else 'No'
    print 'previous time/current time\t', 'C' if use_current_time else 'P'
    print 'relative features\t', 'binary' if is_relat_feature_binary else 'ordinal' 
#     C_values = [0.1, 1, 10, 100]
    C_values = [0.01, 0.02, 0.05, 0.1, 1, 10, 100]
    for C in C_values:
        icaModel = ICAModel(use_local_features=use_local_features,
                            use_current_time=use_current_time,
                            relat_classifier_C=C,
                            is_relat_feature_binary=is_relat_feature_binary)
        icaModel.fit(train_set)
#         Y_pred = icaModel.predict(test_set, maxiter=10)
#     Y_pred = icaModel.predict_by_local_classifiers(train_set)
        Y_pred = icaModel.predict_with_neighbors_true_labels_previous_time(
                                                        train_set)
        train_acc = icaModel.compute_accuracy(train_set, Y_pred)
        Y_pred = icaModel.predict_with_neighbors_true_labels_previous_time(
                                                        test_set)
        test_acc = icaModel.compute_accuracy(test_set, Y_pred)
        print C,'\t',train_acc,'\t',test_acc
#     print icaModel.compute_confusion_matrix(test_set[:,1:], Y_pred[:,1:])

main()