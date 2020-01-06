import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')


