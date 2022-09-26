# mini-batch k-means clustering
# weblink: https://machinelearningmastery.com/clustering-algorithms-with-python/
from numpy import unique
from numpy import where
from sklearn.datasets import make_classification
from sklearn.cluster import MiniBatchKMeans
# import KMeans
from sklearn.cluster import KMeans
from matplotlib import pyplot

import pandas as pd

# read csv input file
input_data = pd.read_csv("All_Participents_DataFrame.csv", sep=";", decimal=',')
#input_data = pd.read_csv("All_Participents_WaveSum_DataFrame.csv", sep=";", decimal=',')

#print(input_data.head(1))
#print(input_data.dtypes)

# define the model
model = MiniBatchKMeans(n_clusters=2)

input_data["Cluster"] = model.fit_predict(input_data)
input_data["Cluster"] = input_data["Cluster"].astype("int")
#print(input_data.head(1)) 

# fit the model
""" model.fit(input_data)
# assign a cluster to each example
yhat = model.predict(input_data)

print(yhat)

# retrieve unique clusters
clusters = unique(yhat)
test = []
#print(clusters[0])
#print(clusters[1])
# create scatter plot for samples from each cluster
for cluster in clusters:
	# get row indexes for samples with this cluster
	row_ix = where(yhat == cluster)
	test.append(row_ix)
	print(row_ix)
	# create scatter of these samples
	#pyplot.scatter(input_data[row_ix, 0], input_data[row_ix, 1])
	#pyplot.scatter(input_data[:, 0], input_data[:, 1], marker='o', s=2, c=C)
# show the plot
print(test) """

ax2 = input_data.plot.scatter(x='Cluster',
                       	      y='pId',
                              c='Cluster',
                              colormap='viridis')
#pyplot.scatter(input_data[:, 0], input_data[:, 1], marker='o', c=yhat)
#pyplot.scatter(test[0][0], test[1][0])
pyplot.show()



#-----------------------------------------------------------------------------------------
# read csv input file
input_data = pd.read_csv("All_Participents_Mean_DataFrame.csv", sep=";", decimal=',')
#input_data = pd.read_csv("All_Participents_WaveSum_Mean_DataFrame.csv", sep=";", decimal=',')

#print(input_data.head(1))
#print(input_data.dtypes)

# define the model
model = MiniBatchKMeans(n_clusters=2)

input_data["Cluster"] = model.fit_predict(input_data)
input_data["Cluster"] = input_data["Cluster"].astype("int")
print(input_data.head(1)) 

ax2 = input_data.plot.scatter(x='Cluster', y='pId', c='Cluster', colormap='viridis')
# show the plot
pyplot.show()



#input_data = pd.read_csv("All_Participents_DataFrame.csv", sep=";", decimal=',')
#input_Features = input_data
# create kmeans object
#kmeans = KMeans(n_clusters=2)
# fit kmeans object to data
#kmeans.fit(input_Features)
# print location of clusters learned by kmeans object
#print(kmeans.cluster_centers_)
# save new clusters for chart
#output_Features = input_Features
#output_Features["Cluster"] = kmeans.fit_predict(input_Features)
#output_Features["Cluster"] = output_Features["Cluster"].astype("int")
#print(output_Features["Cluster"])

#ax2 = input_data.plot.scatter(x='Cluster', y='pId', c='Cluster', colormap='viridis')
#pyplot.show()