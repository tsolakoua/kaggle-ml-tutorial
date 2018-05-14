import pandas as pd
from sklearn.tree import DecisionTreeRegressor

main_file_path = 'train.csv'
data = pd.read_csv(main_file_path)

y = data.SalePrice

data_predictors = ['YearBuilt', 'LotArea', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'YrSold', 'TotRmsAbvGrd']

X = data[data_predictors]

# Define model
data_model = DecisionTreeRegressor()

# Fit model
data_model.fit(X, y)

print("Making predictions for the following 5 houses:")
print(X.head())
print("The predictions are")
print(data_model.predict(X.head()))