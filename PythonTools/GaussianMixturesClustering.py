# gaussian mixture clustering
# weblink: https://machinelearningmastery.com/clustering-algorithms-with-python/
from numpy import unique
from numpy import where
from sklearn.datasets import make_classification
from sklearn.mixture import GaussianMixture
from matplotlib import pyplot

import pandas as pd

# read csv input file
input_data = pd.read_csv("All_Participents_DataFrame.csv", sep=";", decimal=',')
# input_data = pd.read_csv("All_Participents_WaveSum_DataFrame.csv", sep=";", decimal=',')

#print(input_data.head(1))
#print(input_data.dtypes)

# define dataset
#X, _ = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, random_state=4)
# define the model
model = GaussianMixture(n_components=2)

input_data["Cluster"] = model.fit_predict(input_data)
input_data["Cluster"] = input_data["Cluster"].astype("int")
#print(input_data.head(1)) 

""" # fit the model
model.fit(X)
# assign a cluster to each example
yhat = model.predict(X)
# retrieve unique clusters
clusters = unique(yhat)
# create scatter plot for samples from each cluster
for cluster in clusters:
	# get row indexes for samples with this cluster
	row_ix = where(yhat == cluster)
	# create scatter of these samples
	pyplot.scatter(X[row_ix, 0], X[row_ix, 1]) """

ax2 = input_data.plot.scatter(x='Cluster', y='pId', c='Cluster', colormap='viridis')
# show the plot
pyplot.show()


#-----------------------------------------------------------------------------------------
# read csv input file
input_data = pd.read_csv("All_Participents_Mean_DataFrame.csv", sep=";", decimal=',')
#input_data = pd.read_csv("All_Participents_WaveSum_Mean_DataFrame.csv", sep=";", decimal=',')

#print(input_data.head(1))
#print(input_data.dtypes)

# define the model
model = GaussianMixture(n_components=2)

input_data["Cluster"] = model.fit_predict(input_data)
input_data["Cluster"] = input_data["Cluster"].astype("int")
print(input_data.head(1)) 

ax2 = input_data.plot.scatter(x='Cluster', y='pId', c='Cluster', colormap='viridis')
# show the plot
pyplot.show()