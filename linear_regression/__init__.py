import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


def data_preprocess(data):
    features = data.columns[data.isnull().mean() < 0.8]

    new_data = data[features]

    return new_data


train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

new_train_data = data_preprocess(train_data)
new_test_data = data_preprocess(test_data)
