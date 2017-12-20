# Predicting Annual returns
from pathlib import Path
# Importing numpy
import numpy as np
import pandas as pd
from sklearn.cross_validation import KFold

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
train_data = np.genfromtxt(train_data_file_path, dtype=None, delimiter=',', names=True)
test_data = np.genfromtxt(test_data_file_path, dtype=None, delimiter=',', names=True)

print(train_data)
print(train_data.shape)
#print('First Row', train_data[:,0])

# Required label name
label_name = 'return'

# Extracting label from data 
y = train_data[label_name]

# Removing label from train_data using list compression
X = train_data[[b for b in list(train_data.dtype.names) if b != label_name]]

print('original shape X ', X.shape, ' y ', y.shape)

# Now splitting the training set into train and validation sets.

eval_size = 0.10
kf = KFold(len(y), round(1. / eval_size))
train_indices, valid_indices = next(iter(kf))
X_train, y_train = X[train_indices], y[train_indices]
X_valid, y_valid = X[valid_indices], y[valid_indices]

print('Shape after splitting train and validation data ', 
    'X_train ' , X_train.shape, ' y_train ', y_train.shape,
    'X_valid ' ,X_valid.shape, ' y_valid ', y_valid.shape,)

