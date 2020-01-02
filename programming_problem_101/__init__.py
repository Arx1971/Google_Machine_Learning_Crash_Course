from __future__ import print_function
import math
from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset


def preprocess_targets(california_housing_dataframe):
    targets = pd.DataFrame()
    targets['median_house_value'] = (california_housing_dataframe["median_house_value"] / 1000.0)
    return targets


def preprocess_features(california_housing_dataframe):
    features = california_housing_dataframe[[
        'latitude',
        'longitude',
        'housing_median_age',
        'total_rooms',
        'total_bedrooms',
        'population',
        'households',
        'median_income'
    ]]

    processed_features = features.copy()

    processed_features['room_per_person'] = (
            california_housing_dataframe['total rooms'] / california_housing_dataframe['population'])

    return processed_features


tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

california_housing_dataframe = pd.read_csv(
    "https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv", sep=",")

# california_housing_dataframe = california_housing_dataframe.reindex(
#     np.random.permutation(california_housing_dataframe.index))
