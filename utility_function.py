import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor

def get_mae(max_leaf_nodes, predictors_train, predictors_val, targ_train, targ_val):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(predictors_train, targ_train)
    preds_val = model.predict(predictors_val)
    mae = mean_absolute_error(targ_val, preds_val)
    return(mae)

main_file_path = 'train.csv'
data = pd.read_csv(main_file_path)

y = data.SalePrice
X = data[['YearBuilt', 'LotArea', '1stFlrSF', 
                        '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'YrSold', 'TotRmsAbvGrd']]

train_X, val_X, train_y, val_y = train_test_split(X, y,random_state = 0)

for max_leaf_nodes in [5, 40, 100, 1000, 2000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))