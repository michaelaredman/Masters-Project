import pystan
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt 

X = pd.read_csv('X.csv', index_col=0)
expected = pd.read_csv('expected.csv', index_col=0)
cases = pd.read_csv('cases.csv', index_col=0)
adj = pd.read_csv('adj.csv', index_col=0)

X = np.matrix(X)
expected = np.array(expected['expected'])
cases = np.array(cases['observed'])
adj = np.matrix(adj)

cancer_dat = {'n': X.shape[0],
              'p': X.shape[1],
              'X': X,
              'y': cases,
              'log_offset': np.log(expected),
              'W': adj}


fit = pystan.stan(file='scot.stan', data=cancer_dat, iter=1000, chains=4)

fit.plot()
plt.show()
