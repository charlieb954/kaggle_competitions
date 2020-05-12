import pandas as pd
import numpy as np 
import os

# Import models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

## Read in the data, select the columns of interest
## All Columns == 'PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'
test_data = pd.read_csv('test.csv', 
                        usecols = ['PassengerId', 
                                   'Pclass', 
                                   'Sex', 
                                   'Age', 
                                   'SibSp', 
                                   'Parch', 
                                   'Fare']
                        )
train_data = pd.read_csv('train.csv', 
                         usecols = ['PassengerId', 
                                    'Survived', 
                                    'Pclass', 
                                    'Sex', 
                                    'Age', 
                                    'SibSp', 
                                    'Parch', 
                                    'Fare']
                         )

# Does randomising the data make any difference?
test_data = test_data.sample(frac=1).reset_index(drop=True)
train_data = train_data.sample(frac=1).reset_index(drop=True)

## Preprocessing the data, ensure values are usable and remove NaN values
train_data['Sex'] = train_data['Sex'].map({'female':0,'male':1})
test_data['Sex'] = test_data['Sex'].map({'female':0,'male':1})

age_mean_train = train_data['Age'].mean()
age_mean_test = test_data['Age'].mean()

train_data['Age'].fillna(age_mean_train, 
                         inplace=True)

test_data['Age'].fillna(age_mean_test, 
                        inplace=True)

train_data.loc[(train_data['Age'] < 16),'Age_cat'] = 'C'
train_data.loc[(train_data['Age'] >= 16),'Age_cat'] = 'A'
#train_data.loc[(train_data['Age'].isnull()),'Age_cat'] = 'U'

test_data.loc[(test_data['Age'] < 16),'Age_cat'] = 'C'
test_data.loc[(test_data['Age'] >= 16),'Age_cat'] = 'A'
#test_data.loc[(test_data['Age'].isnull()),'Age_cat'] = 'U'

train_data.drop(columns='Age', 
                inplace=True)

test_data.drop(columns='Age', 
               inplace=True)

train_data.loc[(train_data['Fare'].isnull()), 'Fare'] = 0
test_data.loc[(test_data['Fare'].isnull()), 'Fare'] = 0

train_data = pd.get_dummies(train_data, 
                            columns=['Age_cat'], 
                            prefix = ['Age_cat'])

test_data = pd.get_dummies(test_data, 
                           columns=['Age_cat'], 
                           prefix = ['Age_cat'])

## Split the data to train and test data and convert to array
X_train = train_data.drop(columns=['Survived', 'PassengerId']).to_numpy()
y_train = train_data['Survived'].to_numpy()
X_test = test_data.drop(columns='PassengerId').to_numpy()

## Decision Tree
dtc = DecisionTreeClassifier()
model = dtc.fit(X_train, y_train)
dtc_y_pred = model.predict(X_test)
dtc_y_pred_df = pd.DataFrame(dtc_y_pred).rename(columns={0:'Survived'})

print(f'Accuracy of DT classifier on training set: {model.score(X_train, y_train):.2f}')
print(f'Accuracy of DT classifier on test set: {model.score(X_test, dtc_y_pred):.2f}')

result = test_data.merge(dtc_y_pred_df, how = 'left', left_index = True, right_index = True)
submission = result[['PassengerId', 'Survived']].to_csv('result.csv', index = False)