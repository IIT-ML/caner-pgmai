'''
Created on Jan 6, 2015

@author: ckomurlu
'''

import pandas as pd
import numpy as np

def read_data(binCount = 545,discarded_sensors = [5, 15, 18, 49]):
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
    return tempdf

def digitize_data(tempdf = None, binCount = 545,
                  discarded_sensors = [5, 15, 18, 49]):
    if tempdf is None:
        tempdf = read_data(binCount=binCount,
                           discarded_sensors=discarded_sensors)
    tempdf['unixtime'] = pd.Series(np.array([elem.value \
            for elem in tempdf['datentime']]), index=tempdf.index)
    
    timeBins = np.linspace(tempdf.unixtime.min(),\
                           tempdf.unixtime.max(),endpoint=False,num=binCount)
    digTime = np.digitize(tempdf.unixtime,timeBins)
    
    tempdf['digTime'] = digTime
    return tempdf

def window_data(digitizeddf = None, binCount = 545,
                discarded_sensors = [5, 15, 18, 49]):
    if digitizeddf is None:
        digitizeddf = digitize_data(binCount = binCount,
                                    discarded_sensors=discarded_sensors)
    groups = digitizeddf.groupby(['moteid','digTime'])
    meanSeries = groups.temperature.mean()
    time_window_df = pd.DataFrame(meanSeries)
    time_window_df.reset_index(level=0, inplace=True)
    time_window_df.reset_index(level=1, inplace=True)
    time_window_df['digTemp'] = pd.Series(np.empty((len(time_window_df))), dtype = np.int16,
                                   index=time_window_df.index)
    time_window_df['digTemp'][time_window_df.temperature < 20] = 0
    time_window_df['digTemp'][np.logical_and(time_window_df.temperature >= 20,\
                           time_window_df.temperature < 22.5)] = 1
    time_window_df['digTemp'][np.logical_and(time_window_df.temperature >= 22.5,\
                           time_window_df.temperature < 25)] = 2
    time_window_df['digTemp'][time_window_df.temperature >= 25] = 3
    return time_window_df

def convert_digitized_to_feature_matrix(time_window_df,tempdf):
    sorted_mat = time_window_df.sort(columns='digTime').as_matrix()[:,(0,3)]
    daytime = np.array(map(lambda x: x.hour/6, tempdf.datentime))
    tempdf['daytime']=daytime
    day_time_list = np.empty((sorted_mat.shape[0]),dtype='int32')
    for i in range(0,sorted_mat.shape[0]):
        day_time_list[i] = tempdf[tempdf.digTime == sorted_mat[i,0]].daytime.iloc[0]
    feature_mat = np.zeros((len(day_time_list),4))
    for i in range(len(day_time_list)):
        feature_mat[i,day_time_list[i]] = 1
    return feature_mat


#test
tempdf = read_data()
digitizeddf = digitize_data(tempdf=tempdf)
time_window_df = window_data(digitizeddf=digitizeddf)
feature_mat = convert_digitized_to_feature_matrix(time_window_df, tempdf)

print 'end of process'
