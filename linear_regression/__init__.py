import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

data = pd.read_csv('../house_prices_train.csv')

features = [
    'MSSubClass',
    'LotArea',
    'OverallQual',
    'OverallCond',
    'BsmtUnfSF',
    '1stFlrSF',
    '2ndFlrSF',
    'GrLivArea'
]

y = data['SalePrice']
X = data[features]

train_data, test_data, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

x_train = train_data.fillna(train_data.mean)
x_test = test_data.fillna(test_data.mean)

linear_model = LinearRegression()
model = linear_model.fit(x_train, y_train)

prediction = linear_model.predict(x_test)
print(prediction)
print(linear_model.score(X, y))
