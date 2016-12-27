from utils import readdata
from utils.node import Neighborhood
from data.humidity_data_preprocess import HumidityProcessor

import numpy as np

__author__ = 'ckomurlu'


def digitize_data():
    # trainset, testset = readdata.convert_time_window_df_randomvar_hour(True, Neighborhood.all_others_current_time)
    hp = HumidityProcessor()
    trainset, testset = hp.convert_time_window_df_randomvar_hour(True,
                                                                 Neighborhood.itself_previous_others_current)
    X = np.vectorize(lambda x: x.true_label)(trainset)
    xMin = X.min()
    xMax = X.max()
    bins = np.linspace(xMin, xMax, 5)
    digitized = list()
    for row in X:
        digitized.append(np.digitize(row, bins))
    digitized = np.array(digitized)
    np.savetxt('C:/Users/ckomurlu/Documents/workbench/experiments/20151102/humidity_digitized_bins=5_uniqMaxMin.txt',
               digitized, delimiter=',')

def combine_digitized_data():
    digital_temp = np.loadtxt('C:/Users/ckomurlu/Documents/workbench/experiments/20151023/' +
                              'digitized_bins=5_uniqMaxMin.txt', delimiter=',')
    digital_humid = np.loadtxt('C:/Users/ckomurlu/Documents/workbench/experiments/20151102/' +
                               'humidity_digitized_bins=5_uniqMaxMin.txt', delimiter=',')
    digital_combined = np.append(digital_temp, digital_humid, axis=0)
    np.savetxt('C:/Users/ckomurlu/Documents/workbench/experiments/20151103/humidity_digitized_bins=5_uniqMaxMin.txt',
               digital_combined, delimiter=',')

combine_digitized_data()