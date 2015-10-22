__author__ = 'ckomurlu'

from utils.toolkit import *

import numpy as np


def test_standard_error_depr():
    sample = np.arange(1, 10, dtype=np.int_)
    print standard_error_depr(sample)


def test_standard_error():
    sample_array = np.random.rand(3, 4, 5)
    mymean = sample_array.mean(0)
    mymean = np.array([mymean]*3)
    myvar = (np.sum((sample_array - mymean)**2, axis=0))/3
    stddev = myvar**.5
    stderr = stddev/(3**.5)
    assert np.array_equal(stderr, standard_error(sample_array))


def test_print_experiment_parameters_to_file():
    print_experiment_parameters_to_file()