import pystan
import pandas as pd
import numpy as np
import time
import pickle
import seaborn as sns
from matplotlib import pyplot as plt

def load_data():
    temp_expected = pd.read_csv('../../data/csv/expected.csv')
    E = np.array(temp_expected['x'])
    
    temp_sim = pd.read_csv('../../data/csv/simulated.csv')
    temp_times = temp_sim[['Time1', 'Time2', 'Time3', 'Time4', 'Time5', 'Time6', 'Time7', 'Time8', 'Time9', 'Time10', 'Time11', 'Time12', 'Time13', 'Time14', 'Time15']]
    observed_values = np.matrix(temp_times, dtype=np.int)
    
    adj = pd.read_csv('../../data/csv/adjacency.csv', index_col=0)
    W = np.matrix(adj)
    
    numRegions = observed_values.shape[0] #number of regions
    nt = observed_values.shape[1] #number of time points
    
    alpha = 0.9 #this was 1 in the model but that makes the covariance matrix singular

    return numRegions, nt, E, W, alpha, observed_values

numRegions, nt, E, W, alpha, observed_values = load_data()

model_data = {'numRegions': numRegions,
              'nt': nt,
              'observed': observed_values,
              'log_expected': np.log(E),
              'W_n': int(W.sum()/2.0),
              'W': W,
              'alpha': alpha}

print('Starting fit at: ', time.ctime())

iter_num = 2000

fit = pystan.stan(file='var.stan', data=model_data, iter=iter_num, chains=4)

print('Finished fit at: ', time.ctime())

trace = fit.extract()

ts = time.localtime()
file_name = "trace/model_{}-{}-{}__{}-{}.pkl".format(ts[2], ts[1], ts[0], ts[3], ts[4])

with open('summary.txt', 'w') as f:
    print(fit, file=f)

with open(file_name, 'wb') as f:
    pickle.dump(trace, f)

true_unusual = pd.read_csv('../../data/csv/unusual.csv')
unusual_regions = np.array(true_unusual['x'])

indicator = trace['indicator']
indicator_av = pd.DataFrame(indicator.mean(axis=0))

indicator_av.index = range(1, 211)
predictor = indicator_av.sort(0)
regions = np.array(predictor.index)

np.savetxt('predicted.csv', regions, delimiter=',', fmt='%i')

fifteen = regions[-15:]

regions_identified = set(unusual_regions).intersection(fifteen)

num_regions_identified = len(regions_identified)





