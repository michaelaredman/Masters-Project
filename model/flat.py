import matplotlib
matplotlib.use('Agg')
import pymc3 as pm
import scipy
import numpy as np
import pandas as pd
import theano
import theano.tensor as T
import time
from matplotlib import pyplot as plt

theano.config.compute_test_value = 'raise'
#theano.config.gcc.cxxflags = "-fbracket-depth=16000 -O0" # default is 256 | compilation optimizations are turned off for faster compilation
theano.config.exception_verbosity= 'high'
#theano.config.openmp = True # this isn't working with gcc for some reason
theano.config.optimizer = 'fast_compile'

def load_data():
    temp_expected = pd.read_csv('../data/csv/expected.csv')
    E = np.array(temp_expected['x'])
    
    temp_sim = pd.read_csv('../data/csv/simulated.csv')
    temp_times = temp_sim[['Time1', 'Time2', 'Time3', 'Time4', 'Time5', 'Time6', 'Time7', 'Time8', 'Time9', 'Time10', 'Time11', 'Time12', 'Time13', 'Time14', 'Time15']]
    observed_values = np.matrix(temp_times)
    
    adj = pd.read_csv('../data/csv/adjacency.csv', index_col=0)
    W = np.matrix(adj)
    
    numRegions = observed_values.shape[0] #number of regions
    nt = observed_values.shape[1] #number of time points
    Q = np.diag(np.ones(nt-1), k=1) #'adjacency matrix' in time
    
    #making the inverse covariance matricies for the CAR models (ignoring their variances)
    alpha = 0.75 #this was 1 in the model but that makes the covariance matrix singular
    D = np.diag(np.array(W.sum(0))[0]) #diag(d_1,..,d_numRegions) with d_i the number of neighbours of region i
    Tau_v_unscaled = np.array(D - alpha*W)
    Tau_gamma_unscaled = np.identity(n=nt) - Q

    return numRegions, nt, E, Tau_v_unscaled, Tau_gamma_unscaled, observed_values

prob_z = 0.95 #probability of a region following the area specific model

numRegions, nt, E, Tau_v_unscaled, Tau_gamma_unscaled, observed_values = load_data()

model = pm.Model()


print('Data loaded')

print('Starting time: ', time.ctime())


with model:
    """
    BYM prior on the spatial component

        lambda ~ Normal(v_i, sigma_lambda)

        v ~ CAR(W, sigma_v)

    where W is the adjacency matrix.
    
    The CAR model is equivalent to,

        v ~ N(0, T^-1)

    with,  T = (D - alpha*W)*sigma_v

    D = diag(d_1, ..., d_n)
    d_i = 'degree' of region i
    alpha = level of spatial dependence 

    We place vague priors on the variances.
    
    """
    sigma_v = pm.HalfNormal('sigma_v', sd=1) # change this to something more vague
    sigma_lambda = pm.HalfNormal('sigma_lambda', sd=1)
    
    Tau_v = Tau_v_unscaled*sigma_v #covariance matirx for v
    v = pm.MvNormal('v',mu=np.zeros(numRegions), tau=Tau_v, shape=numRegions)
    lmbda = pm.MvNormal('lambda', mu=v, cov=np.identity(numRegions)*sigma_lambda, shape=numRegions)

    """
    BYM prior on the time component
    
    
    
    """
    sigma_gamma = pm.HalfNormal('sigma_gamma', sd=1)
    sigma_xi = pm.HalfNormal('sigma_xi', sd=1)
    Tau_gamma = Tau_gamma_unscaled*sigma_gamma #covariance matrix for xi
    gamma = pm.MvNormal('gamma', mu=np.zeros(nt), cov=Tau_gamma, shape=nt)
    xi = pm.MvNormal('xi', mu=gamma, cov=np.identity(nt)*sigma_xi, shape=nt)


    """
    We attempt to measure a regions deviations
    """
    
    
    """
    mu_it = Expected*exp(lambda_i + xi_t)
    """
    
    mu_temp = [] #rate parameters over the time points
    for i in range(numRegions):
        mu_temp.append(T.stack([E[i]*T.exp(lmbda[i] + xi[t]) for t in range(nt)]))
    mu = T.stack(mu_temp)

    
    observed = pm.Poisson('observed_{}'.format(i), mu = mu, observed=observed_values, shape=(numRegions, nt))
    
print('Model defined at ', time.ctime())

with model:
    step = pm.Metropolis()
    print('Metropolis initialized')
    db = pm.backends.Text('baystdetect_trace')
    trace = pm.sample(draws=2000, trace=db, step=step)

print('End time: ', time.ctime())

#trace_burn = trace[:][3000:]
pm.traceplot(trace)
plt.savefig('trace.png')
