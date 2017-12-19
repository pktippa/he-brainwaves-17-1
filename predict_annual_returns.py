# Importing numpy
import numpy as np
from pathlib import Path

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

# Required label name
label_name = 'return_'

# Extracting label from data 
y_train = train_data[label_name]

# Removing label from train_data using list compression
X_train = train_data[[b for b in list(train_data.dtype.names) if b != label_name]]

# Now splitting the training set into train and validation sets.