#####################################################
#       TEAM NUMBER: 10                             #
#       TEAM MEMBER: Malik Majette  (mamajett)      #
#                    Qua Jones      (qyjones)       #
#                    Wenting Zheng  (wzheng8)       #
#####################################################
import numpy as np
import xlrd
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt


#####################################################
#       This part is to read the player files       #
#####################################################

# Headers: 
xl = pd.ExcelFile("NBA Data.xlsx")
data14 = pd.read_excel(xl, sheet_name="2014-2015")
data15 = pd.read_excel(xl, sheet_name="2015-2016")
data16 = pd.read_excel(xl, sheet_name="2016-2017")
test17 = pd.read_excel(xl, sheet_name="2017-2018")

# Separate x and y for the training datasets
x_train14 = np.array(data14.loc[:, data14.columns[2:-1]])
x_train15 = np.array(data15.loc[:, data15.columns[2:-1]])
x_train16 = np.array(data16.loc[:, data16.columns[2:-1]])
# all three year data
x_train = np.append(x_train14, x_train15, axis = 0)
x_train = np.append(x_train, x_train16, axis = 0)

# x_train = np.array(data.loc[:,data.columns[2:-1]])

y_train14 = []
for data in data14['ALL-STAR']:
    if data == 'Yes':
        y_train14.append(1)
    else:
        y_train14.append(0)
y_train15 = []
for data in data15['ALL-STAR']:
    if data == 'Yes':
        y_train15.append(1)
    else:
        y_train15.append(0)
y_train16 = []
for data in data16['ALL-STAR']:
    if data == 'Yes':
        y_train16.append(1)
    else:
        y_train16.append(0)
y_train = np.append(y_train14, y_train15, axis = 0)
y_train = np.append(y_train, y_train16, axis = 0)


# Separate x and y for the testing dataset
x_test = np.array(test17.loc[:, test17.columns[2:-1]])
y_test = []
for data in test17['ALL-STAR']:
    if data == 'Yes':
        y_test.append(1)
    else:
        y_test.append(0)
# print(x_train16)
#####################################################
#           This part is for normalization          #
#####################################################
# normalize the training datasets
norm14 = (x_train14 - x_train14.min(axis=0)) / (x_train14.max(axis=0) - x_train14.min(axis=0))
norm15 = (x_train15 - x_train15.min(axis=0)) / (x_train15.max(axis=0) - x_train15.min(axis=0))
norm16 = (x_train16 - x_train16.min(axis=0)) / (x_train16.max(axis=0) - x_train16.min(axis=0))
norm_all = (x_train - x_train.min(axis=0)) / (x_train.max(axis=0) - x_train.min(axis=0))

# normalize the testing dataset
normtest14 = (x_test - x_train14.min(axis=0)) / (x_train14.max(axis=0) - x_train14.min(axis=0))
normtest15 = (x_test - x_train15.min(axis=0)) / (x_train15.max(axis=0) - x_train15.min(axis=0))
normtest16 = (x_test - x_train16.min(axis=0)) / (x_train16.max(axis=0) - x_train16.min(axis=0))
normtest_all = (x_test - x_train.min(axis=0)) / (x_train.max(axis=0) - x_train.min(axis=0))

# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# scaler.fit(x_train)
#
# norm_all = scaler.transform(x_train)
# normtest_all = scaler.transform(x_test)
# print(norm_all)

# print(y_train)
