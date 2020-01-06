import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm


def data_preprocess(data):
    col = data.columns[data.isnull().mean() < 0.8]
    new_data = data[col]
    return new_data


train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

new_train_data = data_preprocess(train_data)
new_test_data = data_preprocess(test_data)

y = new_train_data['SalePrice']  # this is our target

# which feature is had best correlation score

corr_sale = new_train_data.corr().SalePrice
corr_sale = corr_sale.sort_values(ascending=False)

features = [
    'OverallQual',
    '1stFlrSF',
    'TotalBsmtSF',
    'YearBuilt',
    'FullBath',
    'TotRmsAbvGrd'
]
