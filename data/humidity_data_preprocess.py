import pandas as pd
import numpy as np
import cPickle
import itertools

from utils.node import SensorRVNode, Neighborhood


__author__ = 'ckomurlu'


class HumidityProcessor(object):

    def __init__(self):
        self.DATA_DIR_PATH = 'C:\\Users\\ckomurlu\\PycharmProjects\\pgmai\\humidityData\\'
        self.humidf = None
        self.time_window_df = None
        self.train_df = None
        self.test_df = None
        self.train_set = None
        self.test_set = None

    def read_data(self, discarded_sensors=list([5, 15, 18, 49]), to_be_pickled=False):
        try:
            self.humidf = cPickle.load(open(self.DATA_DIR_PATH+'humidf.pickle', 'rb'))
        except IOError:
            self.humidf = pd.read_csv('C:/Users/ckomurlu/Documents/workbench/data/' +
                                      'TimeSeries/intelResearch/data/data2.txt', sep=' ', header=None,
                                      names=['date', 'time', 'epoch', 'moteid', 'temperature', 'humidity', 'light',
                                             'voltage'], parse_dates={'datentime': [0, 1]}, dtype={
                                             'datentime': np.datetime64,
                                             'epoch': np.int, 'moteid': np.int8, 'temperature': np.float64,
                                             'humidity': np.float64, 'light': np.float64, 'voltage': np.float64})
            del self.humidf['light']
            del self.humidf['voltage']
            del self.humidf['temperature']
            self.humidf = self.humidf.sort(columns='datentime')
            self.humidf = self.humidf[self.humidf.datentime < pd.Timestamp('2004-03-10 13:08:46.002832')]
            for sensor in discarded_sensors:
                self.humidf = self.humidf[self.humidf.moteid != sensor]
            if to_be_pickled:
                cPickle.dump(self.humidf, open(self.DATA_DIR_PATH+'humidf.pickle', 'wb'),
                             protocol=cPickle.HIGHEST_PROTOCOL)

    def digitize_data(self, bin_count=545, to_be_pickled=False):
        try:
            self.humidf = cPickle.load(open(self.DATA_DIR_PATH + 'digitizeddf.pickle', 'rb'))
        except IOError:
            self.humidf['unixtime'] = pd.Series(np.array([elem.valuefor for elem in self.humidf['datentime']]),
                                                index=self.humidf.index)
            time_bins = np.linspace(self.humidf.unixtime.min(),
                                    self.humidf.unixtime.max(), endpoint=False, num=bin_count)
            dig_time = np.digitize(self.humidf.unixtime, time_bins)

            self.humidf['digTime'] = dig_time
            if to_be_pickled:
                cPickle.dump(self.humidf, open(self.DATA_DIR_PATH+'digitizeddf.pickle', 'wb'),
                             protocol=cPickle.HIGHEST_PROTOCOL)

    def window_data(self, to_be_pickled=False, target_field='humidity'):
        try:
            self.time_window_df = cPickle.load(open(self.DATA_DIR_PATH + 'time_window_df.pickle', 'rb'))
        except IOError:
            groups = self.humidf.groupby(['moteid', 'digTime'])
            mean_series = groups[target_field].mean()
            self.time_window_df = pd.DataFrame(mean_series)
            self.time_window_df.reset_index(level=0, inplace=True)
            self.time_window_df.reset_index(level=1, inplace=True)
            if to_be_pickled:
                cPickle.dump(self.time_window_df,
                             open(self.DATA_DIR_PATH+'time_window_df.pickle', 'wb'),
                             protocol=cPickle.HIGHEST_PROTOCOL)

    '''
    The data returned by this method is a pandas data frame. It's the
    time windowed temporal data, that is 545 rows for each sensor. Columns are
    'digTime', 'moteid', 'temperature', 'digTemp', 'hour'
    The hour feature here is can be 3, 9, 15, 21
    '''
    def create_time_window_df_hour_feature(self, to_be_pickled=False):
        try:
            self.time_window_df = cPickle.load(open(self.DATA_DIR_PATH + 'time_window_df_hour_feature.pickle', 'rb'))
        except IOError:
            self.time_window_df.sort(columns=['digTime', 'moteid'], inplace=True)
            #     daytime = np.array(map(lambda x: x.hour/float(48), digitizeddf.datentime))
            daytime = np.array(map(lambda x: x.hour + (x.minute > 29)*.5, self.humidf.datentime))
            self.humidf['daytime'] = daytime
            day_time_list = np.empty((self.time_window_df.shape[0]), dtype=np.float_)
            for i in range(0, self.time_window_df.shape[0]):
                day_time_list[i] = self.humidf[self.humidf.digTime ==
                                               self.time_window_df.digTime.iloc[i]].daytime.iloc[0]
            self.time_window_df['hour'] = day_time_list
            if to_be_pickled:
                cPickle.dump(self.time_window_df,
                             open(self.DATA_DIR_PATH+'time_window_df_hour_feature.pickle', 'wb'))

    def add_day_to_time_window_df_hour(self):
        time_bins = np.arange(12)*48 + 1
        time_bins = time_bins[1:]
        days = np.digitize(self.time_window_df.digTime, time_bins)
        self.time_window_df['day'] = days

    def train_test_split_by_day_hour(self, to_be_pickled=False):
        try:
            self.train_df, self.test_df = cPickle.load(
                open(self.DATA_DIR_PATH+'traintestdayshour.pickle', 'rb'))
        except IOError:
            if 'day' not in self.time_window_df.columns:
                self.add_day_to_time_window_df_hour()
            train_days = [2, 3, 4]
            test_days = [5, 6]
            self.train_df = self.time_window_df[self.time_window_df.day.isin(train_days)]
            self.test_df = self.time_window_df[self.time_window_df.day.isin(test_days)]
            if to_be_pickled:
                cPickle.dump((self.train_df, self.test_df), open(self.DATA_DIR_PATH + 'traintestdayshour.pickle', 'wb'),
                             protocol=cPickle.HIGHEST_PROTOCOL)

    def convert_time_window_df_randomvar_hour(self, to_be_pickled=False,
                                              #  neighborhood_def=np.setdiff1d):
                                              neighborhood_def='dnm'):
        try:
            self.train_set, self.test_set = cPickle.load(
                open(self.DATA_DIR_PATH + neighborhood_def.__name__+'_hour.pickle','rb'))
            #         raise IOError
        except IOError:
            sensor_IDs = self.train_df.moteid.unique()
            digTime_list = self.train_df.digTime.unique()
            num_sensors = sensor_IDs.shape[0]
            num_dig_time = self.train_df.digTime.unique().shape[0]
            self.train_set = np.ndarray(shape=(num_sensors, num_dig_time),
                                   dtype=SensorRVNode)
            for sensor, digTime in itertools.product(sensor_IDs, digTime_list):
                row = self.train_df[(self.train_df.moteid == sensor) &
                                (self.train_df.digTime == digTime)].iloc[0]
                sensor_id = row.moteid
                sensor_idx = np.where(sensor_IDs == sensor_id)[0][0]
                dig_time = np.where(digTime_list == digTime)[0][0]
                local_feature_vector = row.hour
                neighbors = neighborhood_def(sensor_id, sensor_IDs)
                self.train_set[sensor_idx, dig_time] = \
                    SensorRVNode(sensor_id=row.moteid, dig_time=dig_time,
                                 day=row.day, true_label=row.humidity,
                                 local_feature_vector=local_feature_vector,
                                 is_observed=False, neighbors=neighbors)
            num_dig_time = self.test_df.digTime.unique().shape[0]
            self.test_set = np.ndarray(shape=(num_sensors, num_dig_time),
                                  dtype=SensorRVNode)
            digTime_list = self.test_df.digTime.unique()
            for i in range(len(self.test_df)):
                row = self.test_df.iloc[i]
                sensor_id = row.moteid
                sensor_idx = np.where(sensor_IDs == sensor_id)[0][0]
                dig_time = np.where(digTime_list == row.digTime)[0][0]
                #         print sensor_id,'\t',sensor_idx,'\t',dig_time
                local_feature_vector = row.hour
                neighbors = neighborhood_def(sensor_id, sensor_IDs)
                self.test_set[sensor_idx,dig_time] = \
                    SensorRVNode(sensor_id=sensor_id, dig_time=row.digTime,
                                 day=row.day, true_label=row.humidity,
                                 local_feature_vector=local_feature_vector,
                                 is_observed=False, neighbors=neighbors)
            if to_be_pickled:
                cPickle.dump((self.train_set,self.test_set),
                             open(self.DATA_DIR_PATH + neighborhood_def.__name__+'_hour.pickle',
                                  'wb'),protocol=cPickle.HIGHEST_PROTOCOL)
        return self.train_set, self.test_set

# hp = HumidityProcessor()
# hp.read_data(to_be_pickled=True)
# hp.digitize_data(to_be_pickled=True)
# hp.window_data(to_be_pickled=True)
# hp.create_time_window_df_hour_feature(to_be_pickled=True)
# hp.add_day_to_time_window_df_hour()
# hp.train_test_split_by_day_hour(to_be_pickled=True)
# trainset,testset = hp.convert_time_window_df_randomvar_hour(True,
#                                         Neighborhood.itself_previous_others_current)
# print 'Process ended.'