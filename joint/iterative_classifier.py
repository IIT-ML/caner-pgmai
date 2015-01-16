'''
Created on Jan 14, 2015

@author: ckomurlu
'''

from ml_model import MLModel
from utils.readdata import train_test_split_by_day

from sklearn.linear_model import LogisticRegression
from sklearn.metrics.metrics import accuracy_score, confusion_matrix
from sklearn.svm.classes import SVC
import numpy as np
from bsddb.test.test_pickle import cPickle

class ICAModel(MLModel):
    
    LR = 'lr'
    SVM = 'svm'
    
    def __init__(self, local_classifier_name=LR,
                 relat_classifier_name=LR):
        if local_classifier_name == ICAModel.LR:
            self.local_classifier_name = LogisticRegression
        elif local_classifier_name == ICAModel.SVM:
            self.local_classifier_name = SVC
        else:
            raise ValueError('Wrong classifier name is provided for the '+
                             'local classifier; it should be either \'lr\'' + 
                             ' or \'svm\'')
        if relat_classifier_name == ICAModel.LR:
            self.relat_classifier_name = LogisticRegression
        elif relat_classifier_name == ICAModel.SVM:
            self.relat_classifier_name = SVC
        else:
            raise ValueError('Wrong classifier name is provided for the '+
                             'relational classifier; it should be either' + 
                             ' \'lr\' or \'svm\'')
        self.local_classifier = dict()
        self.relat_classifier = dict()
    
    def fit(self,X_train,Y_train,max_iter):
        self.local_classifier.fit(X_train, Y_train)
    
    def fit_by_df(self,traindays):
        sensor_ids = np.sort(traindays.moteid.unique())
        for sensor in sensor_ids:
            X_local_train = traindays[traindays.moteid == sensor]. \
                as_matrix(['morning','afternoon','evening','night'])
            Y_local_train = traindays[traindays.moteid == sensor]. \
                as_matrix(['digTemp']).reshape(1,-1)[0]
            self.local_classifier[sensor] = self.local_classifier_name()
            self.local_classifier[sensor].fit(X_local_train, Y_local_train)
        digTime = traindays.digTime.unique()
        for sensor in sensor_ids:
            X_relat_train = list()
            Y_relat_train = list()
            for current_time in digTime:
                X_relat_train.append(traindays[(traindays.digTime ==
                                     current_time) & (traindays.moteid !=
                                     sensor)].sort(['moteid']).\
                                     as_matrix(['digTemp']).reshape(-1))
                Y_relat_train.append(traindays['digTemp'][(traindays.digTime ==
                                     current_time) & (traindays.moteid ==
                                                      sensor)].as_matrix()[0])
            X_relat_train = np.vstack(X_relat_train)
            Y_relat_train = np.array(Y_relat_train)
            self.relat_classifier[sensor] = self.relat_classifier_name()
            self.relat_classifier[sensor].fit(X_relat_train,
                                              Y_relat_train)
    
    def predict(self, testdays, maxiter):
        Y_pred = np.empty((len(testdays)),dtype = np.int8)
        sensor_ids = np.sort(testdays.moteid.unique())
        for sensor in sensor_ids:
            X_local_test = testdays[testdays.moteid == sensor]. \
                as_matrix(['morning','afternoon','evening','night'])
            Y_pred[(testdays.moteid == sensor).as_matrix()] = \
                self.local_classifier[sensor].predict(X_local_test)
        Y_pred_temp = np.ndarray(Y_pred.shape,dtype = np.int8)
        
        for i in range(maxiter):
            for i in range(Y_pred.shape[0]):
                current_sensor = testdays.iloc[i]['moteid']
                current_time = testdays.iloc[i]['digTime']
                feature_vect = Y_pred[((testdays.moteid != current_sensor) &
                                      (testdays.digTime == current_time)).\
                                      as_matrix()]
                Y_pred_temp[((testdays.moteid == current_sensor) &
                            (testdays.digTime == current_time)).as_matrix()] = \
                            self.relat_classifier[current_sensor].predict(feature_vect)
        return Y_pred

    def compute_accuracy(self,Y_test,Y_pred):
        return accuracy_score(Y_test,Y_pred)
    
    def compute_confusion_matrix(self, Y_test, Y_pred, labels=None):
        return confusion_matrix(Y_test, Y_pred, labels)
    
##test
# myIca = ICAModel()
# traindays,testdays = train_test_split_by_day()
# traindays,testdays = cPickle.load(open('../data/traintestdays.pickle','rb')) 
# myIca.fit_by_df(traindays)
# myIca.predict(testdays, maxiter = 3)
# print 'end of iterative classifier'
