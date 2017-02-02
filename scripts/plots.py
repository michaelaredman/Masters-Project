
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

observed = pd.read_csv('../data/csv/simulated.csv')
observed.columns = range(16)
observed = np.array(observed[list(range(1, 16))])



for i in [2, 20, 200, 170]:
    plt.figure()
    plt.plot(observed[i-1, :])
    plt.title('Observed values for normal region {}'.format(i))
    plt.xlabel('Time point')
    plt.ylabel('Counts')
    plt.savefig('../figures/observed_normal_{}.png'.format(i))

for i in [6, 10, 82, 83]:
    plt.figure()
    plt.plot(observed[i-1, :])
    plt.title('Observed values for deviant region {}'.format(i))
    plt.xlabel('Time point')
    plt.ylabel('Counts')
    plt.savefig('../figures/observed_deviant_{}.png'.format(i))
