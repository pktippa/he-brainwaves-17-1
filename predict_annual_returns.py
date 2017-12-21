# Predicting Annual returns
from pathlib import Path
# Importing numpy
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from lxml.ElementInclude import include

# Setting up train data file path and resolving absoulte path.
train_data_file_path = 'data/predict_annual_returns/train.csv'
# Resolving the path
train_data_file_path = Path(train_data_file_path).resolve()
# Converting to String
train_data_file_path = str(train_data_file_path)

# Setting up test data file path and resolving absoulte path.
test_data_file_path = 'data/predict_annual_returns/test.csv'
test_data_file_path = Path(test_data_file_path).resolve()
test_data_file_path = str(test_data_file_path)

# Reading data files into numpy array
#train_data = np.genfromtxt(train_data_file_path, dtype=None, delimiter=',', names=True)
#test_data = np.genfromtxt(test_data_file_path, dtype=None, delimiter=',', names=True)

train_data = pd.read_csv(train_data_file_path)
test_data = pd.read_csv(test_data_file_path)
# Printing columns to know the column name of label.
# print(train_data.columns)

# Required label name
label_name = 'return'

# Extracting label from data 
#y = train_data[label_name]

# Extracting the last column from slicing
y = train_data.iloc[:,-1]

# Printing Lables y to see whether it is same.
# print(y)

# Printing columns before dropping last column
# print(train_data.columns)

# Dropping last column and assigning to X as features
X = train_data.drop(label_name, 1)

# Printing columns after dropping last column
# print(X.columns)

# Printing shape before splitting
"""
print('Shape before splitting data ', 
    'X  ' , X.shape, ' y ', y.shape)
"""

# Now splitting the training set into train and validation sets.
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1)

# Printing shape after splitting
"""
print('Shape after splitting train and validation data ', 
    'X_train ' , X_train.shape, ' y_train ', y_train.shape,
    'X_valid ' ,X_valid.shape, ' y_valid ', y_valid.shape,)
"""

# Printing Description / column datatypes of Training set, to identify different variables in data
print(X_valid.describe(include='all'))
print(X_valid.dtypes)
