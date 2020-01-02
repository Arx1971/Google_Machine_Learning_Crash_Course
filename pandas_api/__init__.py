import pandas as pd

city_names = pd.Series(['San Francisco', 'San Jose', 'Sacramento'])
population = pd.Series([852469, 1015785, 485199])

df = pd.DataFrame({'City Name': city_names, 'Population': population})

california_housing_dataframe = pd.read_csv(
    "https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv", sep=",")

print(california_housing_dataframe.columns)
print(california_housing_dataframe.describe())

cities = pd.DataFrame({'City Name': city_names, 'Population': population})
print(cities)
