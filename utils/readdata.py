'''
Created on Jan 6, 2015

@author: ckomurlu
'''

import pandas as pd
import numpy as np
import cPickle

def read_data(binCount=545, discarded_sensors=[5, 15, 18, 49],
              to_be_pickled=False):
    tempdf = pd.read_csv('C:/Users/ckomurlu/Documents/workbench/data/' + \
    'TimeSeries/intelResearch/data/data2.txt', sep=' ', header=None, \
    names=['date','time','epoch','moteid','temperature', 'humidity','light', \
           'voltage'], parse_dates={'datentime' : [0,1]}, dtype={ \
           'datentime':np.datetime64, \
           'epoch':np.int,'moteid':np.int8,'temperature':np.float64, \
           'humidity':np.float64,'light':np.float64,'voltage':np.float64})
    del tempdf['light']
    del tempdf['voltage']
    del tempdf['humidity']
    tempdf = tempdf.sort(columns='datentime')
    tempdf = tempdf[tempdf.datentime < 
                    pd.Timestamp('2004-03-10 13:08:46.002832')]
    for sensor in discarded_sensors:
        tempdf = tempdf[tempdf.moteid != sensor]
    if to_be_pickled:
        cPickle.dump(tempdf, open('../data/tempdf.pickle','wb'),
                     protocol=cPickle.HIGHEST_PROTOCOL)
    return tempdf

def digitize_data(tempdf = None, binCount = 545,
                  discarded_sensors = [5, 15, 18, 49],
                  to_be_pickled=False):
    if tempdf is None:
        try:
            tempdf = cPickle.load(open('../data/tempdf.pickle','rb'))
        except(IOError):
            tempdf = read_data(binCount=binCount,
                           discarded_sensors=discarded_sensors,
                           to_be_pickled=to_be_pickled)
    tempdf['unixtime'] = pd.Series(np.array([elem.value \
            for elem in tempdf['datentime']]), index=tempdf.index)
    
    timeBins = np.linspace(tempdf.unixtime.min(),\
                           tempdf.unixtime.max(),endpoint=False,num=binCount)
    digTime = np.digitize(tempdf.unixtime,timeBins)
    
    tempdf['digTime'] = digTime
    if to_be_pickled:
        cPickle.dump(tempdf, open('../data/digitizeddf.pickle','wb'),
                     protocol=cPickle.HIGHEST_PROTOCOL)
    return tempdf

def window_data(digitizeddf = None, binCount = 545,
                discarded_sensors = [5, 15, 18, 49],
                to_be_pickled=False):
    if digitizeddf is None:
        try:
            digitizeddf = cPickle.load(open('../data/digitizeddf.pickle','rb'))
        except(IOError):
            digitizeddf = digitize_data(binCount = binCount,
                                    discarded_sensors=discarded_sensors,
                                    to_be_pickled=to_be_pickled)
    groups = digitizeddf.groupby(['moteid','digTime'])
    meanSeries = groups.temperature.mean()
    time_window_df = pd.DataFrame(meanSeries)
    time_window_df.reset_index(level=0, inplace=True)
    time_window_df.reset_index(level=1, inplace=True)
    time_window_df['digTemp'] = pd.Series(np.empty((len(time_window_df))),
                                          dtype = np.int16,
                                   index=time_window_df.index)
    time_window_df['digTemp'][time_window_df.temperature < 20] = 0
    time_window_df['digTemp'][np.logical_and(time_window_df.temperature >= 20,
                           time_window_df.temperature < 22.5)] = 1
    time_window_df['digTemp'][np.logical_and(time_window_df.temperature >= 22.5,
                           time_window_df.temperature < 25)] = 2
    time_window_df['digTemp'][time_window_df.temperature >= 25] = 3
    if to_be_pickled:
        cPickle.dump(time_window_df,
                     open('../data/time_window_df.pickle','wb'),
                     protocol=cPickle.HIGHEST_PROTOCOL)
    return time_window_df

def convert_digitized_to_feature_matrix(digitizeddf = None,
                                        time_window_df = None,
                                        binCount = 545,
                                        discarded_sensors = [5,15,18,49],
                                        to_be_pickled=False):
    if digitizeddf is None:
        try:
            digitizeddf = cPickle.load(open('../data/digitizeddf.pickle','rb'))
        except(IOError):
            digitizeddf = digitize_data(to_be_pickled=to_be_pickled)
    try:
        sorted_mat = cPickle.load(open('../data/sorted_mat.pickle','rb'))
    except(IOError):
        if time_window_df is None:
            try:
                time_window_df = cPickle.\
                                    load(open('../data/time_window_df.pickle',
                                              'rb'))
            except(IOError):
                time_window_df = window_data(digitizeddf=digitizeddf,
                                             to_be_pickled=to_be_pickled)
        sorted_mat = time_window_df.sort(columns='digTime').\
                                    as_matrix()[:,(0,3)]
        if to_be_pickled:
            cPickle.dump(sorted_mat, open('../data/sorted_mat.pickle','wb'),
                         protocol=cPickle.HIGHEST_PROTOCOL)
    daytime = np.array(map(lambda x: x.hour/6, digitizeddf.datentime))
    digitizeddf['daytime']=daytime
    day_time_list = np.empty((sorted_mat.shape[0]),dtype='int32')
    for i in range(0,sorted_mat.shape[0]):
        day_time_list[i] = digitizeddf[digitizeddf.digTime ==
                                        sorted_mat[i,0]].daytime.iloc[0]
    bin_feature_mat = np.zeros((len(day_time_list),4))
    for i in range(len(day_time_list)):
        bin_feature_mat[i,day_time_list[i]] = 1
    if to_be_pickled:
        cPickle.dump(bin_feature_mat, open('../data/bin_feature_mat.pickle','wb'),
                     protocol=cPickle.HIGHEST_PROTOCOL)
    return sorted_mat,bin_feature_mat

def partition_feature_mat_into_sensors(digitizeddf = None,
                                       time_window_df = None,
                                       to_be_pickled=False):
    if digitizeddf is None:
        try:
            digitizeddf = cPickle.load(open('../data/digitizeddf.pickle','rb'))
        except(IOError):
            digitizeddf = digitize_data(to_be_pickled=to_be_pickled)
    if time_window_df is None:
        try:
            time_window_df = cPickle.\
                                load(open('../data/time_window_df.pickle',
                                          'rb'))
        except(IOError):
            if digitizeddf is None:
                try:
                    digitizeddf = cPickle.load(open('../data/digitizeddf.pickle','rb'))
                except(IOError):
                    digitizeddf = digitize_data(to_be_pickled=to_be_pickled)
            time_window_df = window_data(digitizeddf=digitizeddf,
                                         to_be_pickled=to_be_pickled)
    sensor_IDs = np.unique(time_window_df.moteid)
    sorted_mat = time_window_df.sort(columns='digTime').\
                                    as_matrix()[:,(0,3)]
    daytime = np.array(map(lambda x: x.hour/6, digitizeddf.datentime))
    digitizeddf['daytime']=daytime
    dayTimeList = np.empty((sorted_mat.shape[0]),dtype=np.int64)
    for i in range(0,sorted_mat.shape[0]):
        dayTimeList[i] = digitizeddf[digitizeddf.digTime == sorted_mat[i,0]].daytime.iloc[0]
    time_window_df.sort('digTime',inplace=True)
    time_window_df['dayTime'] = dayTimeList
    bin_feature_dict = dict()
    one_sensor_data_dict = dict()
    for current_sensor_ID in sensor_IDs:
        one_sensor_data = time_window_df[time_window_df['moteid']==
                                       current_sensor_ID]
        one_sensor_mat = one_sensor_data.as_matrix()[:,(0,3,4)]
        one_sensor_data_dict[current_sensor_ID] = one_sensor_mat
        bin_feature_mat = np.zeros((one_sensor_mat.shape[0],4))
        for i in range(one_sensor_mat.shape[0]):
            bin_feature_mat[i,one_sensor_mat[i,2]] = 1
        bin_feature_dict[current_sensor_ID] = bin_feature_mat
    if to_be_pickled:
        cPickle.dump((one_sensor_data_dict,bin_feature_dict), open(
                '../data/one_sensor_data.pickle','wb'),
                     protocol=cPickle.HIGHEST_PROTOCOL)
    return one_sensor_data_dict,bin_feature_dict

def create_time_window_df_bin_feature(digitizeddf=None,
                           time_window_df=None,
                           to_be_pickled=False):
    '''The data returned by this method is a pandas data frame. It's the
    time windowed temporal data, that is 545 rows for each sensor. Columns are
        'digTime', 'moteid', 'temperature', 'digTemp', 'morning',
        'afternoon', 'evening', 'night'
    '''
    if digitizeddf is None:
        try:
            digitizeddf = cPickle.load(open('../data/digitizeddf.pickle','rb'))
        except(IOError):
            digitizeddf = digitize_data(to_be_pickled=to_be_pickled)
    if time_window_df is None:
        try:
            time_window_df = cPickle.\
                                load(open('../data/time_window_df.pickle',
                                          'rb'))
        except(IOError):            
            time_window_df = window_data(digitizeddf=digitizeddf,
                                         to_be_pickled=to_be_pickled)
    time_window_df.sort(columns='digTime',inplace=True)
    daytime = np.array(map(lambda x: x.hour/6, digitizeddf.datentime))
    digitizeddf['daytime']=daytime
    dayTimeList = np.empty((time_window_df.shape[0]),dtype=np.int64)
    for i in range(0,time_window_df.shape[0]):
        dayTimeList[i] = digitizeddf[digitizeddf.digTime ==
                time_window_df.digTime.iloc[i]].daytime.iloc[0]
    bin_feature_mat = np.zeros((time_window_df.shape[0],4))
    bin_feature_mat[np.arange(time_window_df.shape[0]),dayTimeList] = 1
    time_window_df['morning'] = bin_feature_mat[:,0]
    time_window_df['afternoon'] = bin_feature_mat[:,1]
    time_window_df['evening'] = bin_feature_mat[:,2]
    time_window_df['night'] = bin_feature_mat[:,3]
    cPickle.dump(time_window_df,
        open('../data/time_window_df_bin_feature.pickle','wb'))
    return time_window_df

def add_day_to_time_window_df(time_window_df=None):
    if time_window_df is None:
        time_window_df = cPickle.load(open(
            '../data/time_window_df_bin_feature.pickle','rb'))
    timeBins = np.arange(12)*48 + 1
    timeBins = timeBins[1:]
    days = np.digitize(time_window_df.digTime,timeBins)
    time_window_df['day'] = days
    return time_window_df

def train_test_split_by_day():
    train_days = range(5)
    test_days = range(5,10)
    time_window_df = add_day_to_time_window_df()
    train_df = time_window_df[time_window_df.day.isin(train_days)]
    test_df = time_window_df[time_window_df.day.isin(test_days)]
    return train_df,test_df
    
    
    
#test1
# tempdf = read_data()
# digitizeddf = digitize_data(tempdf=tempdf)
# time_window_df = window_data(digitizeddf=digitizeddf)
# feature_mat = convert_digitized_to_feature_matrix(tempdf, time_window_df)

#test2
# feature_mat = convert_digitized_to_feature_matrix(to_be_pickled=True)
# print 'end of process'


# partition_feature_mat_into_sensors(to_be_pickled=False)
# create_time_window_df_bin_feature()
traindays,testdays = train_test_split_by_day()
cPickle.dump((traindays,testdays),
             open('../data/traintestdays.pickle','wb'),
             protocol=cPickle.HIGHEST_PROTOCOL)