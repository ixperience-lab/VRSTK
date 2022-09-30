# mini-batch k-means clustering
# weblink: https://machinelearningmastery.com/clustering-algorithms-with-python/
from numpy import unique
from numpy import where
import numpy as np
from sklearn.datasets import make_classification
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import pairwise_distances_argmin
# import KMeans
from sklearn.cluster import KMeans
from matplotlib import pyplot
import seaborn as sns
#sns.set_theme(color_codes=True)
import os
import pandas as pd

# read csv input file
input_data = pd.read_csv("All_Participents_EEG_Clusterd_WaveSum_DataFrame.csv", sep=";", decimal=',')
# drop columns conscientious, time, pId
input_x = input_data.drop(columns=['Conscientious', 'time', 'pId'])
# count rows and columns
#r_num = input_x.shape[0]
#print(r_num)
c_num = input_x.shape[1]
print(c_num)

# nomalize input data to create more varianz in the data (because of statictical values like mean, std, max, counts, ...)
scaler = StandardScaler()
scaler.fit(input_x)
#print(scaler.mean_)
transformed_input_x = scaler.transform(input_x)
print(transformed_input_x[0])

# define and fit the model
miniBatchKMeans = MiniBatchKMeans(init="k-means++", n_clusters=2, batch_size=10, n_init=10, max_no_improvement=10, verbose=0,).fit(transformed_input_x) #miniBatchKMeans = MiniBatchKMeans(n_clusters=2).fit(input_x)
input_score = miniBatchKMeans.score(transformed_input_x) #input_score = miniBatchKMeans.score(np.array(input_x)[0:1])
input_cluster_centers_ = miniBatchKMeans.cluster_centers_
print(input_score)
print(input_cluster_centers_)

input_x["Cluster"] = miniBatchKMeans.predict(transformed_input_x) #input_x["Cluster"] = miniBatchKMeans.predict(input_x)
input_x["Cluster"] = input_x["Cluster"].astype("int")

input_x["pId"] = input_data["pId"]
#input_x["ClusterFactor"] = pd.Series(data = np.zeros(input_x.shape[0]))

input_means_labels = pairwise_distances_argmin(transformed_input_x, input_cluster_centers_)
print(input_means_labels)

fig = pyplot.figure(figsize=(10, 10))
colors = ["#4EACC5", "#FF9C34"]

# KMeans
ax = fig.add_subplot(1, 1, 1)
for k, col in zip(range(2), colors):
	my_members = input_means_labels == k
	cluster_center = input_cluster_centers_[k]
	for i in range(c_num - 1):
		ax.plot(transformed_input_x[my_members, i], transformed_input_x[my_members, i+1], "w", markerfacecolor=col, marker=".")
	ax.plot(cluster_center[0], cluster_center[1], "o", markerfacecolor=col, markeredgecolor="k", markersize=6 )
ax.set_title("MiniBatchKMeans")
ax.set_xticks(())
ax.set_yticks(())
pyplot.text(-3.5, 1.8, "inertia: %f" % (miniBatchKMeans.inertia_))
pyplot.show()


ax2 = input_x.plot.scatter(x='Cluster',  y='pId', c='Cluster', colormap='viridis')
pyplot.show()

""" _Ids = [ 1,2,3,4,5,6,7,10,13,14,15,16,17,18,19,20,31,34]
for id in _Ids:
	temp = input_x.loc[input_x["pId"] == id]
	component_C0 =  temp[temp.Cluster == 0].shape[0]
	component_C1 =  temp[temp.Cluster == 1].shape[0]
	component_Sum = component_C0 + component_C1 
	faktor_C0 = component_C0 / component_Sum
	faktor_C1 = component_C1 / component_Sum
	print(faktor_C0)
	print(faktor_C1)
	if faktor_C0 > faktor_C1: 
		input_x.ClusterFactor[input_x.pId == id] = faktor_C0
	if faktor_C0 < faktor_C1:  
		input_x.ClusterFactor[input_x.pId == id] = faktor_C1

ax2 = input_x.plot.scatter(x='pId',  y='ClusterFactor', c='Cluster', colormap='viridis')
pyplot.show() """

# ------- pair (column_x_column) wise k-means cluster

""" input_data = pd.read_csv("All_Participents_Clusterd_WaveSum_DataFrame.csv", sep=";", decimal=',')
input_x = input_data.drop(columns=['Conscientious', 'time', 'pId'])
print(input_x.head(1))
r_num = input_x.shape[0]
print(r_num)
c_num = input_x.shape[1]
print(c_num)

# nomalize input data to create more varianz in the data
scaler = StandardScaler()
scaler.fit(input_x)
transformed_input_x = scaler.transform(input_x)
transformed_input_x = pd.DataFrame(data=transformed_input_x)

miniBatchKMeans = MiniBatchKMeans(n_clusters=2)
result_df = pd.DataFrame(columns=('column_index', 'compare_column_index', 'score', 'cluster_centers'))#, 'cluster_result'))
for i in range(c_num):
	c1 = transformed_input_x.iloc[:,i].values
	print('======================================================= ' + str(i))
	if i < (c_num - 1) :
		for j in range(i+1,c_num):
			print('=================================================================== ' + str(j))
			c2 = transformed_input_x.iloc[:,j].values
			tmp_input_x = pd.DataFrame(c1)#data=[c_1, c_2]).T
			tmp_input_x[1] = c2
			#break
			miniBatchKMeans.fit(tmp_input_x)
			tmp_score = miniBatchKMeans.score(tmp_input_x)
			tmp_cluster_centers_ = miniBatchKMeans.cluster_centers_
			#print(tmp_score)
			#print(tmp_cluster_centers_)
			#tmp_input_x["Cluster"] = miniBatchKMeans.predict(tmp_input_x)
			#tmp_input_x["Cluster"] = tmp_input_x["Cluster"].astype("int")
			result_df = pd.concat([result_df, pd.DataFrame({'column_index': i, 'compare_column_index' : j, 'score' : tmp_score, 'cluster_centers':  list(tmp_cluster_centers_)})])#, 'cluster_result': list(tmp_input_x["Cluster"].values)})])
			#break	

# mode
mode = 0o666
# with mode 0o666
path = "./output/K-Means-Feature-Compare".format(i)
if not os.path.exists(path):
	os.mkdir(path, mode)
generatet_file_name = "{}/Features_Analyse_with_Features{}".format(path, ".csv")
result_df.to_csv(generatet_file_name, sep=";")
 """
#-----------------------------------------------------------------------------------------
# read csv input file
#input_data = pd.read_csv("All_Participents_Mean_DataFrame.csv", sep=";", decimal=',')
#input_data = pd.read_csv("All_Participents_WaveSum_Mean_DataFrame.csv", sep=";", decimal=',')

#print(input_data.head(1))
#print(input_data.dtypes)

# define the model
#model = MiniBatchKMeans(n_clusters=2)

#input_data["Cluster"] = model.fit_predict(input_data)
#input_data["Cluster"] = input_data["Cluster"].astype("int")
#print(input_data.head(1)) 

#ax2 = input_data.plot.scatter(x='Cluster', y='pId', c='Cluster', colormap='viridis')
# show the plot
#pyplot.show()



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