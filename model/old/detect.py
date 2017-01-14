import pymc3 as pm
import scipy
import numpy as np
import pandas as pd
import theano
import theano.tensor as T

theano.config.compute_test_value = 'raise'
#theano.config.gcc.cxxflags = "-fbracket-depth=16000" # default is 256

def load_data():
    temp_expected = pd.read_csv('~/4th Year/project/data/csv/expected.csv')
    E = np.array(temp_expected['x'])
    
    temp_sim = pd.read_csv('~/4th Year/project/data/csv/simulated.csv')
    temp_times = temp_sim[['Time1', 'Time2', 'Time3', 'Time4', 'Time5', 'Time6', 'Time7', 'Time8', 'Time9', 'Time10', 'Time11', 'Time12', 'Time13', 'Time14', 'Time15']]
    observed_values = np.matrix(temp_times)
    
    adj = pd.read_csv('/Users/Mike/4th Year/project/data/csv/adjacency.csv', index_col=0)
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

model = pm.Model()

prob_z = 0.95 #probability of a region following the area specific model

numRegions, nt, E, Tau_v_unscaled, Tau_gamma_unscaled, observed_values = load_data()


with model:
    """
    The model is broadly as specified in Baystdetect (Best et al, 2012)
    xi_i ~ CAR(Q, sigma_{i, xi}^2) 
    eta_i ~ N(v_i, sigma_eta^2)
    v ~ CAR(W, sigma_v^2)
    etc...
    """
    #priors on the variances
    print('Setting priors for variances...', end=' ')
    
    sigma_eta = pm.HalfNormal('sigma_eta', sd = 1)
    sigma_v = pm.HalfNormal('sigma_v', sd = 1)
    sigma_gamma = pm.HalfNormal('sigma_gamma', sd = 1)

    print('complete!')

    #priors on hyper parameters
    print('Setting priors on hyperparameters...', end=' ')
    
    a = pm.Normal('a',mu=0,tau=0.001) #diffuse prior on a
    b = pm.HalfNormal('b',sd=2.5) #informative prior on b
    u = np.empty(numRegions, dtype=object)
    sigma_xi = np.empty(numRegions, dtype=object)
    ##for i in range(numRegions):
    ##    u[i] = pm.Normal('u_%i' % i,mu=0, tau=0.001)
    ##    sigma_xi[i] = pm.Lognormal('sigma_xi_%i' % i,mu=a, sd=b)

    print('complete!')

    #print(np.linalg.matrix_rank(Tau_gamma_unscaled))

    print('Setting BYM prior...', end=' ')
    
    Tau_gamma = Tau_gamma_unscaled*sigma_gamma #covariance matirx for gamma
    Tau_v = Tau_v_unscaled*sigma_v #covariance matirx for v
    gamma = pm.MvNormal('gamma',mu=np.zeros(nt), tau=Tau_gamma, shape=nt)
    v = pm.MvNormal('v',mu=np.zeros(numRegions), tau=Tau_v, shape=numRegions)
    
    print('complete!')

    #print(sigma_eta.__dict__.keys())
    #print(sigma_eta.random())

    alpha0 = pm.Normal('alpha0',mu=0, tau=0.001) #diffuse prior (not flat like baystdetect)
    eta = np.empty(numRegions, dtype=object)
    #Tau_xi = np.empty(shape=(numRegions, nt, nt), dtype=object)
    Tau_xi = np.empty(numRegions, dtype=object)
    xi_i_unnormed = np.empty(shape=(numRegions, nt), dtype=object)
    print('made it further!')
    #make eta a mvnormal
    for i in range(numRegions):
        eta[i] = pm.Normal('eta_%i' % i,mu=v[i], sd=sigma_eta)

    print('eta defined')
    ##for i in range(numRegions):
    ##    Tau_xi[i] = np.empty(shape=(nt, nt), dtype=object)
    ##    Tau_xi[i] = Tau_gamma_unscaled*sigma_xi[i]
    ##    #Tau_xi[i,:,:] = Tau_gamma_unscaled*eta[i] #covariance matrix for eta_i
    ##    #xi_i_unnormed[i,:] = pm.MvNormal('xi_%i' % i,mu=np.zeros(nt), tau=Tau_xi[i,:,:], shape=nt)
    ##    xi_i_unnormed[i,:] = pm.MvNormal('xi_%i' % i,mu=np.zeros(nt), tau=Tau_xi[i], shape=nt)

    print('Defining mixture component...', end=' ')

        
    ##z = np.empty(numRegions, dtype=object) #mixture componenti
    ##for i in range(numRegions):
    ##    z[i] = pm.Bernoulli('z_%i' % i,prob_z)

    print('complete!')

    print('Setting rate parameters...', end=' ')
    mu_C = np.empty(shape=(numRegions, nt), dtype=object) #rate of the general trend
    for i in range(numRegions):
        mu_C[i,:] = alpha0 + eta[i] + gamma 

    ##mu_AC = np.empty(shape=(numRegions, nt), dtype=object)
    ##for i in range(numRegions):
    ##    mu_AC[i,:] = u[i] + xi_i_unnormed[i,:]
    
    mu = np.empty(shape=(numRegions, nt), dtype=object) #rates by region through time
    for i in range(numRegions):
        for t in range(nt):
            ##mu[i,t] = (mu_C[i,t]*z[i] + mu_AC[i,t]*(1 - z[i]))*E[i] #mixture of the two rates
            mu[i,t] = E[i]*mu_C[i,t]
    print('complete!')



    observed = np.empty(shape=numRegions, dtype=object)
    for i in range(numRegions):
            observed[i] = T.stack([pm.Poisson('observed_{}_{}'.format(i,t), mu = mu[i,t], observed=observed_values[i,t]) for t in range(nt)])

    print('Model created!')

    #pm.sample(draws=1000)

    strat = pm.find_MAP(fmin=scipy.optimize.fmin_powell)
    #print(start)
    
