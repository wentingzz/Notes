import numpy as np
import matplotlib.pyplot as plot
import csv

### (a)
test = np.genfromtxt('hw2q1_test.csv', dtype=None, skip_header=1, delimiter=',')
train = np.genfromtxt('hw2q1_train.csv', dtype=None, skip_header=1, delimiter=',')
Rcounttrain = sum(1 for i in range(0, len(train)) if train[i][60].decode() == 'R')
Rcounttest = sum(1 for i in range(0, len(test)) if test[i][60].decode() == 'R')
print("a) testing size = ", len(test), "; testing R = ", Rcounttest, "; testing M = ", len(test) - Rcounttest)
print("   training size = ", len(train), "; training R = ", Rcounttrain, "; training M = ", len(train) - Rcounttrain)

max = np.zeros(len(train[0]) - 1)
min = np.zeros(len(train[0]) - 1)
for i in range(len(train[0]) - 1):
    max[i] = (train[0][i])  # copy the first row to the max
    min[i] = max[i]         # copy the first row to the min
    for j in range(len(train)):     # iterate through each column and update the min and max
        if max[i] < train[j][i]:
            max[i] = train[j][i]
        if min[i] > train[j][i]:
            min[i] = train[j][i]

### (b)
ntrain = []
ntest = []
for j in range(len(train)):
    for i in range(len(train[0]) - 1):
        ntrain.append((train[j][i] - min[i])/(max[i] - min[i]))
for k in range(len(test)):
    for i in range(len(train[0]) - 1):
        ntest.append((test[k][i] - min[i]) / (max[i] - min[i]))
ntrain = np.reshape(ntrain, (len(train),len(train[0]) - 1))
ntest = np.reshape(ntest, (len(test), len(test[0]) - 1))

### (b) i
covtrain = np.cov(ntrain)
covtest = np.cov(ntest)
# print('b)i. covariance of training')
# print(covtrain)

### (b) ii
eig_train = np.linalg.eig(covtrain)     #[0] is eigenvalues, [1] is eigenvectors
eigenvalues = eig_train[0][:]
eigenvalues.sort()
# eig_test = np.linalg.eig(covtest)
print('b) ii. size of covariance matrix = ', covtrain.shape)
print('       5 largest eigenvalues are: ', eigenvalues[-1], ', ', eigenvalues[-2], ', ', eigenvalues[-3], ', ', eigenvalues[-4], ', ', eigenvalues[-5], '.')

