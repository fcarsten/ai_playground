from prepare_data import train_dataset, train_labels, test_dataset, test_labels, valid_dataset, valid_labels
from sklearn.linear_model import LinearRegression
import numpy as np

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

# train_data_subset=[]
# train_lable_subset= []
#
# for i in range(500):
#     train_data_subset.append(np.reshape(train_dataset[i], -1))
#     train_lable_subset.append(train_labels[i])
#
# lr.fit(train_data_subset, train_lable_subset)
#
# train_data_subset=[]
# train_lable_subset= []
#
#
# print("Regression score: {}".format(lr.score(X_test, y_test)))

