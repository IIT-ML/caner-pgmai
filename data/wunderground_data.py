import pandas as pd
import numpy as np
import cPickle as cpk
from datetime import datetime, timedelta

from utils.node import SensorRVNode


def create_data_matrix():
    # dq = pd.read_csv('C:/Users/ckomurlu/Documents/workbench/data/wunderground/IL/dq.csv')
    dq = pd.read_csv('C:/Users/ckomurlu/Documents/workbench/data/wunderground/countrywide/dq.csv')
    dq.drop('Unnamed: 0', axis=1, inplace=True)
    loc_groups = dq.groupby('location')
    n_sensors = len(loc_groups)
    n_timestamps = len(dq.groupby(['month', 'day', 'quarter']))
    data_mat = np.empty(shape=(n_sensors, n_timestamps), dtype=SensorRVNode)
    count = 0
    for name, group in loc_groups:
        t = 0
        for index, row in group.iterrows():
            month = int(row['month'])
            day = int(row['day'])
            quarter = int(row['quarter'])
            data_mat[count, t] = SensorRVNode(row['location'], None, None, true_label=row['Tmean'],
                                              local_feature_vector=quarter, is_observed=False)
            t += 1
        count += 1
    return data_mat


def read_data_matrix(fname):
    with open(fname, 'rb') as inputfile:
        data_mat = cpk.load(inputfile)
    return data_mat


def split_train_test(data_name='IL'):
    trainstart = 0
    trainend = 4 * (datetime(2015, 2, 28) - datetime(2014, 12, 31)).days
    teststart = trainend
    testend = teststart + 4 * 12
    if data_name == 'IL':
        fname = 'C:/Users/ckomurlu/PycharmProjects/pgmai/csvData/wunderground/IL/3months/wund_3months_mat.pkl'
    elif data_name == 'countrywide':
        fname = 'C:/Users/ckomurlu/PycharmProjects/pgmai/csvData/wunderground/countrywide/3months/wund_3months_mat.pkl'
    else:
        raise ValueError()
    data_mat = read_data_matrix(fname)
    trainset = data_mat[:, trainstart:trainend]
    testset = data_mat[:, teststart:testend]
    return trainset, testset


def main():
    # fname = 'C:/Users/ckomurlu/PycharmProjects/pgmai/csvData/wunderground/IL/3months/wund_3months_mat.pkl'
    fname = 'C:/Users/ckomurlu/PycharmProjects/pgmai/csvData/wunderground/countrywide/3months/wund_3months_mat.pkl'
    data_mat = create_data_matrix()
    with open(fname, 'wb') as outputfile:
        cpk.dump(data_mat, outputfile)
    data_mat2 = read_data_matrix(fname)
    data_mat[0, 0] == data_mat2[0, 0]
    print np.array_equal(data_mat, data_mat2)


def printTrueLabelMat():
    # fnamein = 'C:/Users/ckomurlu/PycharmProjects/pgmai/csvData/wunderground/IL/3months/wund_3months_mat.pkl'
    # fnameout = 'C:/Users/ckomurlu/PycharmProjects/pgmai/csvData/wunderground/IL/3months/wund_3months_mat.csv'
    fnamein = 'C:/Users/ckomurlu/PycharmProjects/pgmai/csvData/wunderground/countrywide/3months/wund_3months_mat.pkl'
    fnameout = 'C:/Users/ckomurlu/PycharmProjects/pgmai/csvData/wunderground/countrywide/3months/wund_3months_mat.csv'
    with open(fnamein, 'rb') as inputfile:
        data_mat = cpk.load(inputfile)
        with open(fnameout, 'wb') as outpufile:
            np.savetxt(fnameout, np.vectorize(lambda x: x.true_label)(data_mat), delimiter=',')


def createParentChildDict():
    path = 'C:\Users\ckomurlu\Documents\workbench\data\wunderground\countrywide\digitized\wgroundcwide_dag_bin5.txt'. \
        replace('\\', '/')
    adjmat = np.loadtxt(path, delimiter=',')
    parentDict = dict()
    for i in range(adjmat.shape[1]):
        parentDict[i] = np.flatnonzero(adjmat[:, i]).astype(int).tolist()
    childDict = dict()
    for i in range(adjmat.shape[0]):
        childDict[i] = np.flatnonzero(adjmat[i]).astype(int).tolist()

    cpk.dump((parentDict, childDict),
             open(r'C:\Users\ckomurlu\Documents\workbench\data\wunderground\countrywide'
                  r'\wground_parentChildDicts_k2_bin5.pkl', 'wb'))
    # (sampleParentDict, sampleChildDict) = cpk.load(
    #     open(r'C:\Users\ckomurlu\Documents\workbench\data\wunderground\IL\wground_parentChildDicts_k2_bin5.pkl', 'rb'))
    # sampleParentDict, sampleChildDict


createParentChildDict()
# printTrueLabelMat()

# main()
# print 'End of process'
