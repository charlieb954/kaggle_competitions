import pandas as pd 
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# 1. Gather data
# Import sample data
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# 2. Prepare Data
# Fix missing values -> should this be the mean or 0?
[train_data[each].fillna(train_data[each].mean(), inplace=True) for each in train_data.columns if train_data[each].dtype in ['int64', 'float64']]
[test_data[each].fillna(train_data[each].mean(), inplace=True) for each in test_data.columns if test_data[each].dtype in ['int64', 'float64']]

# TODO find a better way of filling these N/As
[train_data[each].fillna('None', inplace=True) for each in train_data.columns if train_data[each].dtype not in ['int64', 'float64']]
[test_data[each].fillna('None', inplace=True) for each in test_data.columns if test_data[each].dtype not in ['int64', 'float64']]

# Split the data ready to train the model
X = train_data.drop(columns=['Id', 'SalePrice']).to_numpy()
y = train_data['SalePrice'].to_numpy()
X_data = test_data.drop(columns='Id').to_numpy()

# Create the encoder to convert text to arrays
# handle_unknwon = "ignore", means that if there are any values the encoder doesn't know
# for example it's in the test data but not train data, it'll fill with 0s
encoder = OneHotEncoder(handle_unknown="ignore")
encoder.fit(X)

# Apply the encoder to convert text to arrays
X = encoder.transform(X)
X_data = encoder.transform(X_data)

X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.20)

# 3. Choose a model
# Select the model -> Regression chosen as the data is continuous and not categorical
lin_reg = LinearRegression()

# 4. Train the model using just the train data
lin_reg.fit(X_train, y_train)

# 5. Evaluate the mode
print(f'Accuracy of LR on test set: {lin_reg.score(X_test, y_test):.2f}')

# 6. Parameter tuning
# TODO, work out which parameters we can change to affect the validity of the model
# Use the rest of the data to train the model
#lin_reg.fit(X_test, y_test) -> this will replace the model, not add to it
lin_reg.fit(X, y)

# 7. Prediction
# Using the model, predict the test data ready to submit to Kaggle
y_pred = lin_reg.predict(X_data)

# Assign results to a DataFrame and merge with the test data
lin_reg_result = pd.DataFrame(y_pred)
lin_reg_result.rename(columns={0:'SalePrice'}, inplace=True)

lin_reg_result = test_data.merge(lin_reg_result, 
                                        how = 'left', 
                                        left_index = True, 
                                        right_index = True)

# Output results
lin_reg_result[['Id', 'SalePrice']].to_csv('results/results7.csv', index = False)