from abc import ABCMeta
import numpy as np

import utils.properties
from data.humidity_data_preprocess import HumidityProcessor
from utils.node import Neighborhood
from utils import readdata
from data import wunderground_data


class DataProvider(object):
    __metaclass__ = ABCMeta

    def __init__(self):
        raise NotImplementedError('One cannot instantiate the calss DataProvider. It\'s a static class')

    @staticmethod
    def provide_data():
        if 'temperature' == utils.properties.data:
            trainset, testset = readdata.convert_time_window_df_randomvar_hour(True,
                                                        Neighborhood.itself_previous_others_current)
        elif 'humidity' == utils.properties.data:
            hp = HumidityProcessor()
            # hp.read_data(to_be_pickled=True)
            # hp.digitize_data(to_be_pickled=True)
            # hp.window_data(to_be_pickled=True)
            # hp.create_time_window_df_hour_feature(to_be_pickled=True)
            # hp.add_day_to_time_window_df_hour()
            # hp.train_test_split_by_day_hour(to_be_pickled=True)
            trainset, testset = hp.convert_time_window_df_randomvar_hour(True,
                                                                         Neighborhood.itself_previous_others_current)
        elif 'temperature+humidity' == utils.properties.data:
            trainset1, testset1 = readdata.convert_time_window_df_randomvar_hour(True,
                                                        Neighborhood.itself_previous_others_current)
            hp = HumidityProcessor()
            # hp.read_data(to_be_pickled=True)
            # hp.digitize_data(to_be_pickled=True)
            # hp.window_data(to_be_pickled=True)
            # hp.create_time_window_df_hour_feature(to_be_pickled=True)
            # hp.add_day_to_time_window_df_hour()
            # hp.train_test_split_by_day_hour(to_be_pickled=True)
            trainset2, testset2 = hp.convert_time_window_df_randomvar_hour(True,
                                                                         Neighborhood.itself_previous_others_current)
            trainset = np.append(trainset1, trainset2, axis=0)
            testset = np.append(testset1, testset2, axis=0)
        elif 'wunderground' == utils.properties.data:
            trainset, testset = wunderground_data.split_train_test()
        else:
            raise ValueError('Unknown value for utils.properties.data: ' + utils.properties.data)
        return trainset, testset

    @staticmethod
    def get_topology_file_path():
        if 'k2_bin5' == utils.properties.dbn_topology:
            if 'temperature' == utils.properties.data:
                return utils.properties.temperature_k2_bin5_topology_ParentChildDictPath
            elif 'humidity' == utils.properties.data:
                return utils.properties.humidity_k2_bin5_topology_ParentChildDictPath
            elif 'temperature+humidity' == utils.properties.data:
                return utils.properties.temperature_humidity_k2_bin5_topology_ParentChildDictPath
            elif 'wunderground' == utils.properties.data:
                return utils.properties.wground_k2_bin5_topology_ParentChildDictPath
            else:
                raise NotImplementedError('Unknown data identifier: Available data are temperature, humidity,' +
                                          'and temperature+humidity for now.')
        elif 'k2_bin10' == utils.properties.dbn_topology:
            if 'temperature' == utils.properties.data:
                return utils.properties.temperature_k2_bin10_topology_ParentChildDictPath
            elif 'humidity' == utils.properties.data:
                raise NotImplementedError('Topology on 10 bin data is not generated for humidity data')
            elif 'temperature+humidity' == utils.properties.data:
                raise NotImplementedError('Topology on 10 bin data is not generated for temperature+humidity data')
            else:
                raise NotImplementedError('Unknown data identifier: Available data are temperature, humidity,' +
                                          'and temperature+humidity for now.')

    @staticmethod
    def get_train_test_csv():
        trainset_path = 'C:/Users/ckomurlu/PycharmProjects/pgmai/csvData/intelHumidityTrain.csv'
        testset_path = 'C:/Users/ckomurlu/PycharmProjects/pgmai/csvData/intelHumidityTest.csv'
        trainset, testset = DataProvider.provide_data()
        trainmat = np.vectorize(lambda x: x.true_label)(trainset)
        testmat = np.vectorize(lambda x: x.true_label)(testset)
        np.savetxt(trainset_path, trainmat, delimiter=', ')
        np.savetxt(testset_path, testmat, delimiter=', ')

