import numpy as np
from sklearn import linear_model

def readdata():
    fname = 'StatQuizData.csv'
    
    # Find csv file dimensions
    with open(fname) as f:
        for nrows, row_string in enumerate(f):
            if nrows == 0:
                ncols = row_string.count(',') + 1
            pass

    data = np.zeros((nrows, ncols))

    # read data
    with open(fname) as f:
        for nrows, row_string in enumerate(f):
            if nrows == 0:
                pass
            else:
                data[nrows-1, :] = np.array(row_string.rstrip().split(',')).astype('float')

    X = data[:, 1:-1]
    y = data[:, -1]

    return X, y

def normalize(X):
    '''
    Preprocess data of vastly different dimensions.
    Return X with columns of mean=0 and std=1.
    '''

    means = []
    stds = []
 
    X_norm = X
 
    nc = X.shape[1]
    for i in range(nc):
        m = np.mean(X[:, i])
        s = np.std(X[:, i])
        means.append(m)
        stds.append(s)
        X_norm[:, i] = (X_norm[:, i] - m) / s
 
    return X_norm, means, stds

def linfit(X,y):
    clf = linear_model.LinearRegression()
    clf.fit(X,y)
    return clf.coef_

