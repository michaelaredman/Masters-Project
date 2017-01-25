import seaborn as sns
from matplotlib import pyplot as plt

temporal = fit.extract('temporal')
temporal = temporal['temporal']
temporal = pd.DataFrame(temporal)

temporal.columns = [i for i in range(1, 16)]
temporal_means = [temporal[i].mean for i in range(1, 16)]

plt.figure(figsize=(20,10))
sns.violinplot(temporal)
plt.title('Temporal Trend - Naive Fit')
plt.xlabel('Time Point')
plt.ylabel('Value')
#plt.savefig('something.png')
