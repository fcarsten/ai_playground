from sklearn.linear_model import LinearRegression
import pickle
import numpy as np

pickle_file = 'notMNIST.pickle'

with open(pickle_file, 'rb') as f:
  save = pickle.load(f)
  train_dataset = save['train_dataset']
  train_labels = save['train_labels']
  valid_dataset = save['valid_dataset']
  valid_labels = save['valid_labels']
  test_dataset = save['test_dataset']
  test_labels = save['test_labels']
  del save  # hint to help gc free up memory
  print('Training set', train_dataset.shape, train_labels.shape)
  print('Validation set', valid_dataset.shape, valid_labels.shape)
  print('Test set', test_dataset.shape, test_labels.shape)

lr = LinearRegression()
# lr = LinearRegression(False, False, True, -1)
X_test = test_dataset.reshape(test_dataset.shape[0], 28 * 28)
y_test = test_labels

sample_size = 10000
X_train = train_dataset[:sample_size].reshape(sample_size, 784)
y_train = train_labels[:sample_size]
lr.fit(X_train, y_train)
score= lr.score(X_test, y_test)
print("Regression score: {}".format(score))

train_data_subset=[]
train_lable_subset= []

for i in range(10000):
    train_data_subset.append(np.reshape(train_dataset[i], -1))
    train_lable_subset.append(train_labels[i])

lr.fit(train_data_subset, train_lable_subset)

train_data_subset=[]
train_lable_subset= []


print("Regression score: {}".format(lr.score(X_test, y_test)))

