import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

main_file_path = 'train.csv'
data = pd.read_csv(main_file_path)

y = data.SalePrice
data_predictors = []
X = data[['YearBuilt', 'LotArea', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'YrSold', 'TotRmsAbvGrd']]

train_X, val_X, train_y, val_y = train_test_split(X, y,random_state = 0)

# Define model
data_model = DecisionTreeRegressor()

# Fit model
data_model.fit(train_X, train_y)

predictions = data_model.predict(val_X)
print(mean_absolute_error(val_y, predictions))