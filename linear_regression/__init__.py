import pandas as pd
import numpy as np

data = pd.read_csv('../house_prices_train.csv')

print(data.columns)

features = [
    'MSSubClass',
    'LotArea',
    'LotFrontage',
    'LotShape',
    'OverallQual',
    'OverallCond',
    'BsmtUnfSF',
    '1stFlrSF',
    '2ndFlrSF',
    'GrLivArea'
]

y = data['SalePrice']
X = data[features]

print(X)
