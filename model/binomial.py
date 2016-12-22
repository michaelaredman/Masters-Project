import pymc3 as pm
import numpy as np
import pandas as pd
import theano
import theano.tensor as T
from theano.printing import pydotprint

theano.config.compute_test_value = 'raise'

temp_expected = pd.read_csv('~/4th Year/project/data/csv/expected.csv')
E = np.array(temp_expected['x'])

temp_simulated = pd.read_csv('~/4th Year/project/data/csv/simulated.csv')
temp_times = temp_simulated[['Time1', 'Time2', 'Time3', 'Time4', 'Time5', 'Time6', 'Time7', 'Time8', 'Time9', 'Time10', 'Time11', 'Time12', 'Time13', 'Time14', 'Time15']]
observed_values = np.matrix(temp_times)

adj = pd.read_csv('/Users/Mike/4th Year/project/data/csv/adjacency.csv', index_col=0)
W = np.matrix(adj)


model = pm.Model()
numRegions = observed_values.shape[0] #number of regions
nt = observed_values.shape[1] #number of time points
prob_z = 0.95 #probability of a region following the area specific model
Q = np.diag(np.ones(nt-1), k=1) #'adjacency matrix' in time

#making the inverse covariance matricies for the CAR models (ignoring their variances)
alpha = 0.75 #this was 1 in the model but that makes the covariance matrix singular
D = np.diag(np.array(W.sum(0))[0]) #diag(d_1,..,d_numRegions) with d_i the number of neighbours of region i
Tau_v_unscaled = np.array(D - alpha*W)
Tau_gamma_unscaled = np.identity(n=nt) - Q

with model:
    """
    BYM prior on the spatial component

        lambda ~ Normal(v_i, sigma_lambda)

        v ~ CAR(W, sigma_v)

    where W is the adjacency matrix.
    
    The CAR model is equivalent to,

        v ~ N(0, T^-1)

    with,  T = (D - alpha*W)sigma_v

    D = diag(d_1, ..., d_n)
    d_i = 'degree' of region i
    alpha = level of spatial dependence 

    We place vague priors on the variances.

    """
    Tau_v = Tau_v_unscaled*sigma_v #covariance matirx for v
    v = pm.MvNormal('v',mu=np.zeros(numRegions), tau=Tau_v, shape=numRegions)
    lmbda = np.empty(numRegions, dtype=object)
    for i in range(numRegions):
        lmbda[i] = pm.Normal('lambda_%i' % i,mu=v[i], sd=sigma_lambda)
