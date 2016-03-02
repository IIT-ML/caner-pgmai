__author__ = 'ckomurlu'

import utils.properties

import numpy as np
import types
import datetime
import os


def standard_error_depr(sample):
    flat = sample.flatten()
    stddev = flat.std()
    sample_size = flat.shape[0]
    return stddev/(sample_size**.5)


def standard_error(sample_array, axis=0):
    stddev = np.std(sample_array, axis=axis)
    size = sample_array.shape[axis]
    return stddev/(size**.5)


def print_experiment_parameters_to_file():
    file_name = utils.properties.outputDirPath + 'experimentParameters' + utils.properties.timeStamp + '.txt'
    if not os.path.exists(utils.properties.outputDirPath):
            os.makedirs(utils.properties.outputDirPath)
    f = open(file_name,'wb')
    # f.write('mh_sample')
    # for key in utils.properties.__dict__:
    #     print key, utils.properties.__dict__[key]
    properties_dict = utils.properties.__dict__
    for item in dir(utils.properties):
        if not item.startswith('__') and \
           not isinstance(properties_dict[item], types.ModuleType) and \
           item != 'ts':
            f.write(item + ': ' + str(properties_dict[item]) + '\n')
        elif item == 'ts':
            f.write(item + ': ' + str(properties_dict[item]) + ' ' +
                    datetime.datetime.fromtimestamp(properties_dict[item]).strftime('%H:%M %m/%d/%Y') + '\n')