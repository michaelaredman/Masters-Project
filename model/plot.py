import seaborn as sns

temporal = fit.extract('temporal')
temporal = temporal['temporal']
temporal = pd.DataFrame(temporal)

temporal.columns = [i for i in range(1, 16)]
temporal_means = [temporal[i].mean for i in range(1, 16)]
sns.violinplot(temporal)
