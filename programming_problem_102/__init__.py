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
from sklearn.model_selection import train_test_split

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format


def preprocess_features(california_housing_dataframe):
    selected_features = california_housing_dataframe[
        ["latitude",
         "longitude",
         "housing_median_age",
         "total_rooms",
         "total_bedrooms",
         "population",
         "households",
         "median_income"]]
    processed_features = selected_features.copy()
    # Create a synthetic feature.
    processed_features["rooms_per_person"] = (
            california_housing_dataframe["total_rooms"] /
            california_housing_dataframe["population"])
    return processed_features


def preprocess_targets(california_housing_dataframe):
    output_targets = pd.DataFrame()
    output_targets["median_house_value"] = (
            california_housing_dataframe["median_house_value"] / 1000.0)
    return output_targets


california_housing_dataframe = pd.read_csv(
    "https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv", sep=",")

print(california_housing_dataframe)

california_housing_dataframe = california_housing_dataframe.reindex(
    np.random.permutation(california_housing_dataframe.index))

preprocess_data = preprocess_features(california_housing_dataframe)
preprocess_target = preprocess_targets(california_housing_dataframe)

train_data, train_val, test_data, test_val = train_test_split(preprocess_data, preprocess_target, train_size=0.7,
                                                              test_size=0.3, random_state=1)
train_data = pd.DataFrame(train_data)
train_val = pd.DataFrame(train_val)
test_data = pd.DataFrame(test_data)
test_val = pd.DataFrame(test_val)
