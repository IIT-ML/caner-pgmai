'''
Created on Jan 14, 2015

@author: ckomurlu
'''

from ml_model import MLModel
from utils.decorations import deprecated,cheating
from utils import constants

from sklearn.linear_model import LogisticRegression
from sklearn.metrics.metrics import accuracy_score, confusion_matrix
from sklearn.svm.classes import SVC
import numpy as np

class ICAModel(MLModel):
    
    def __init__(self, local_classifier_name=constants.LR,
                 local_classifier_C=1.0, relat_classifier_name=constants.LR,
                 relat_classifier_C=1.0, use_local_features=False,
                 use_current_time=True, is_relat_feature_binary = False,
                 immediate_update=False):
        if local_classifier_name == constants.LR:
            self.local_classifier_name = LogisticRegression
        elif local_classifier_name == constants.SVM:
            self.local_classifier_name = SVC
        else:
            raise ValueError('Wrong classifier name is provided for the '+
                             'local classifier; it should be either \'lr\'' + 
                             ' or \'svm\'')
        self.local_classifier_C = local_classifier_C
        if relat_classifier_name == constants.LR:
            self.relat_classifier_name = LogisticRegression
        elif relat_classifier_name == constants.SVM:
            self.relat_classifier_name = SVC
        else:
            raise ValueError('Wrong classifier name is provided for the '+
                             'relational classifier; it should be either' + 
                             ' \'lr\' or \'svm\'')
        self.relat_classifier_C = relat_classifier_C
        self.local_classifier = dict()
        self.relat_classifier = dict()
        self.use_local_features = use_local_features
        self.use_current_time = use_current_time
        self.is_relat_feature_binary = is_relat_feature_binary 
        self.immediate_update = immediate_update
    
    def fit(self, train_set):
        idlist = np.vectorize(lambda x: x.sensor_id)(train_set[:,0])
        sensor_ID_dict = dict(zip(idlist,np.arange(len(idlist))))
        local_feature_vector_len = len(train_set[0,0].local_feature_vector)
        X_local_train = np.empty(shape=train_set.shape+
                                 (local_feature_vector_len,),dtype=np.int8)
        Y_train = np.empty(shape=train_set.shape,dtype=np.int8)
        row_count,col_count = train_set.shape
        for i in range(row_count):
            for j in range(col_count):
                X_local_train[i,j] = train_set[i,j].local_feature_vector
                Y_train[i,j] = train_set[i,j].true_label
            self.local_classifier[i] = self.local_classifier_name(
                                        C=self.local_classifier_C)
            self.local_classifier[i].fit(X_local_train[i], Y_train[i])
        self.fit_current_time(train_set, sensor_ID_dict, X_local_train,
                              Y_train)
    
    def fit_current_time(self, train_set, sensor_ID_dict, X_local_train,
                         Y_train):
        row_count,col_count = train_set.shape
        relat_feature_vector_len = len(train_set[0,0].neighbors)
#         if self.is_relat_feature_binary:
#             X_relat_train_bin = np.empty(shape=train_set.shape+
#                                      (relat_feature_vector_len*4,),dtype=np.int8)
        for i in range(row_count):
            X_relat_train = np.empty(shape=(0,relat_feature_vector_len),
                                     dtype=np.int8)
            X_local_reduced = np.empty(shape=(0,X_local_train.shape[2]),
                                       dtype=np.int8)
            Y_relat_train = np.empty(shape=(0,1),dtype=np.int8)
            for j in range(col_count):
                current_relat_feature_vector = self.generate_relat_feature_vector(
                                            train_set,i,j,sensor_ID_dict)
                if current_relat_feature_vector[0] != constants.INF:
                    X_relat_train = np.append(X_relat_train,
                                              [current_relat_feature_vector],
                                              axis=0)
                    X_local_reduced = np.append(X_local_reduced,
                                                [X_local_train[i,j]], axis=0)
                    Y_relat_train = np.append(Y_relat_train,Y_train[i,j])
            self.relat_classifier[i] = self.relat_classifier_name(
                                        C=self.relat_classifier_C)
            if self.is_relat_feature_binary:
                X_relat_train_bin = self.convert_to_binary(X_relat_train)
                if self.use_local_features:
                    self.relat_classifier[i].fit(np.append(X_relat_train_bin,
                                X_local_reduced, axis=1), Y_relat_train)
                else:
                    self.relat_classifier[i].fit(X_relat_train_bin, Y_relat_train)
            else:
                if self.use_local_features:
                    self.relat_classifier[i].fit(np.append(X_relat_train,
                                X_local_reduced, axis=1), Y_relat_train)
                else:
                    self.relat_classifier[i].fit(X_relat_train, Y_relat_train)
    
    def generate_relat_feature_vector(self, data_set, node_i, node_j, node_ID_dict):
        current_node = data_set[node_i,node_j]
        relat_feature_vector_len = len(current_node.neighbors)
        relat_feature_vector = np.empty(shape=(relat_feature_vector_len,),
                                        dtype=np.int8)
        for k in range(relat_feature_vector_len):
            neighbor_id,neighbor_time_offset = current_node.neighbors[k]
            neighbor_time = node_j + neighbor_time_offset
            if neighbor_time < 0:
                relat_feature_vector = np.ones((relat_feature_vector_len,),
                                               dtype=np.int8) * constants.INF
                break
            relat_feature_vector[k] = data_set[
                    node_ID_dict[neighbor_id],neighbor_time].\
                    true_label
        return relat_feature_vector

    def predict_by_local_classifiers(self, test_set, evidence_mat=None):
        if evidence_mat is None:
            evidence_mat = np.zeros(test_set.shape,dtype=np.bool8)
        
        Y_pred = np.empty(shape=test_set.shape,dtype = np.int8)
        evid_indices = np.where(evidence_mat == 1)
        try:
            Y_pred[evid_indices] = np.vectorize(lambda x: x.true_label)(
                                                test_set[evid_indices])
        except(IndexError):
            pass
        nonevid_indices = np.where(evidence_mat == 0)
        for i,j in zip(nonevid_indices[0],nonevid_indices[1]):
            Y_pred[i,j] = self.local_classifier[i].predict(test_set[i,j].\
                                                        local_feature_vector)
        return Y_pred
        
    
    def predict(self, test_set, maxiter=5, evidence_mat=None):
        if evidence_mat is None:
            evidence_mat = np.zeros(test_set.shape,dtype=np.bool8)
        Y_pred = self.predict_by_local_classifiers(test_set, evidence_mat)
        idlist = np.vectorize(lambda x: x.sensor_id)(test_set[:,0])
        sensor_ID_dict = dict(zip(idlist,np.arange(len(idlist))))
        if self.use_current_time:
            return self.__predict_current_time(test_set, maxiter, evidence_mat,
                                             Y_pred, sensor_ID_dict)
        else:
            return self.__predict_previous_time(test_set, maxiter, evidence_mat,
                                             Y_pred, sensor_ID_dict)
            
    def __predict_current_time(self, test_set, maxiter, evidence_mat, Y_pred,
                             sensor_ID_dict):
        Y_pred_temp = np.empty(shape=test_set.shape,dtype = np.int8)
        nonevid_indices = np.where(evidence_mat == 0)
        for count in range(maxiter):
            for i,j in zip(nonevid_indices[0],nonevid_indices[1]):
                neighbor_indices = self.get_neighbor_indices(test_set[i,j],
                                             sensor_ID_dict)
                current_feature_vector = Y_pred[neighbor_indices,j]
                if self.is_relat_feature_binary:
                    current_feature_vector = self.convert_to_binary(
                                                current_feature_vector)
                if self.use_local_features:
                    current_feature_vector = np.append(
                        current_feature_vector,test_set[i,j].\
                        local_feature_vector)
                Y_pred_temp[i,j] = self.relat_classifier[i].predict(
                    current_feature_vector)
            Y_pred = Y_pred_temp.copy()
        return Y_pred
    
    @cheating
    def predict_with_neighbors_true_labels_current_time(self,test_set):
        idlist = np.vectorize(lambda x: x.sensor_id)(test_set[:,0])
        sensor_ID_dict = dict(zip(idlist,np.arange(len(idlist))))
        row_count,col_count = test_set.shape
        Y_pred = np.ones(shape=test_set.shape,dtype = np.int8)*(-1)
        for i in range(row_count):
            for j in range(0,col_count):
                current_feature_vector = self.generate_relat_feature_vector(
                                                test_set, i, j, sensor_ID_dict)
                if current_feature_vector[0] == constants.INF:
                    Y_pred[i,j] = constants.INF
                else:
                    if self.is_relat_feature_binary:
                        current_feature_vector = self.convert_to_binary(
                                                    current_feature_vector)
                    if self.use_local_features:
                        current_feature_vector = np.append(
                            current_feature_vector,test_set[i,j].\
                            local_feature_vector)
                    Y_pred[i,j] = self.relat_classifier[i].predict(
                                        current_feature_vector)
        return Y_pred

    def __predict_previous_time(self, test_set, maxiter, evidence_mat, Y_pred,
                             sensor_ID_dict):
        row_count,col_count = test_set.shape
        Y_pred_temp = np.empty(shape=test_set.shape,dtype = np.int8)
        Y_pred_first_col = Y_pred[:,0]
        for count in range(maxiter):
            for i in range(row_count):
                for j in range(1,col_count):
                    neighbor_indices = self.get_neighbor_indices(test_set[i,j],
                                             sensor_ID_dict)
                    neighbor_indices = np.append(i,neighbor_indices)
                    current_feature_vector = Y_pred[neighbor_indices,j-1]
                    if self.is_relat_feature_binary:
                        current_feature_vector = self.convert_to_binary(
                                                current_feature_vector)
                    if self.use_local_features:
                        current_feature_vector = np.append(
                            current_feature_vector,test_set[i,j].\
                            local_feature_vector)
                    Y_pred_temp[i,j] = self.relat_classifier[i].predict(
                        current_feature_vector)
            Y_pred = Y_pred_temp.copy()
            Y_pred[:,0] = Y_pred_first_col
        return Y_pred

    def get_neighbor_indices(self, node, sensor_ID_dict):
        num_neighbor = len(node.neighbors)
        neighbor_indices = np.empty((num_neighbor,),dtype=np.int8)
        for k in range(num_neighbor):
            neighbor_indices[k] = sensor_ID_dict[node.neighbors[k]]
        return neighbor_indices

    def compute_accuracy(self,test_set,Y_pred):
        assert test_set.shape == Y_pred.shape,'Test set and predicted label'+\
               ' matrix shape don\'t match'
        Y_true = [node.true_label for row in test_set for node in row]
        return accuracy_score(Y_true, np.ravel(Y_pred))
    
    def compute_confusion_matrix(self,test_set, Y_pred):
        assert test_set.shape == Y_pred.shape,'Test set and predicted label'+\
               ' matrix shape don\'t match'
        Y_true = [node.true_label for row in test_set for node in row]
        return confusion_matrix(Y_true, np.ravel(Y_pred))

    def convert_to_binary(self,feature_mat):
        if len(feature_mat.shape) > 2:
            raise TypeError('The input matrix dimension > 2')
        elif len(feature_mat.shape) == 1:
#             col_count, = feature_mat.shape
#             bin_mat = np.zeros((col_count,4),dtype=np.bool_)
#             bin_mat[np.arange(col_count),feature_mat] = 1
            feature_mat.shape = (1,) + feature_mat.shape 
        row_count,col_count = feature_mat.shape
        bin_mat = np.empty((row_count,0),dtype=np.bool_)
        for i in np.arange(col_count):
            col = np.zeros((row_count,4), dtype=np.bool_)
            col[np.arange(row_count),feature_mat[:,i]] = 1
            bin_mat = np.append(bin_mat, col, axis=1)
        
        return bin_mat

    @deprecated
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
            
        
    @deprecated
    def predict_by_df(self, testdays, maxiter):
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
##test
# myIca = ICAModel()
# traindays,testdays = train_test_split_by_day()
# traindays,testdays = cPickle.load(open('../data/traintestdays.pickle','rb')) 
# myIca.fit_by_df(traindays)
# myIca.predict(testdays, maxiter = 3)
# print 'end of iterative classifier'
