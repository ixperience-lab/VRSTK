# mini-batch k-means clustering
# weblink: https://machinelearningmastery.com/clustering-algorithms-with-python/
import numpy as np
import numpy.matlib
from pandas import DataFrame
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from matplotlib import pyplot
import seaborn as sns
import os
import pandas as pd

# Source-Link: https://towardsdatascience.com/confidence-in-k-means-d7d3a13ca856
# Source-Link: https://github.com/drmattcrooks/Medium-SoftClusteringWeights
def soft_clustering_weights(data, cluster_centres, **kwargs):
    
    """
    Function to calculate the weights from soft k-means
    data: Array of data. Features arranged across the columns with each row being a different data point
    cluster_centres: array of cluster centres. Input kmeans.cluster_centres_ directly.
    param: m - keyword argument, fuzziness of the clustering. Default 2
    """
    
    # Fuzziness parameter m>=1. Where m=1 => hard segmentation
    m = 2
    if 'm' in kwargs:
        m = kwargs['m']
    
    Nclusters = cluster_centres.shape[0]
    Ndp = data.shape[0]
    Nfeatures = data.shape[1]

    # Get distances from the cluster centres for each data point and each cluster
    EuclidDist = np.zeros((Ndp, Nclusters))
    for i in range(Nclusters):
        EuclidDist[:,i] = np.sum((data-np.matlib.repmat(cluster_centres[i], Ndp, 1))**2,axis=1)
    

    
    # Denominator of the weight from wikipedia:
    invWeight = EuclidDist**(2/(m-1))*np.matlib.repmat(np.sum((1./EuclidDist)**(2/(m-1)),axis=1).reshape(-1,1),1,Nclusters)
    Weight = 1./invWeight
    
    return Weight

# read csv input file
#input_data = pd.read_csv("All_Participents_Clusterd_WaveSum_DataFrame.csv", sep=";", decimal=',')			# plan of sensors weighting:
#input_data = pd.read_csv("All_Participents_ECG_Clusterd_WaveSum_DataFrame.csv", sep=";", decimal=',') 		# weight with 1/9
#input_data = pd.read_csv("All_Participents_EDA_Clusterd_WaveSum_DataFrame.csv", sep=";", decimal=',') 		# weight with 2/9
#input_data = pd.read_csv("All_Participents_EEG_Clusterd_WaveSum_DataFrame.csv", sep=";", decimal=',') 		# weight with 1/9
#input_data = pd.read_csv("All_Participents_EYE_Clusterd_WaveSum_DataFrame.csv", sep=";", decimal=',') 		# weight with 2/9
input_data = pd.read_csv("All_Participents_PAGES_Clusterd_WaveSum_DataFrame.csv", sep=";", decimal=',') 	# weight with 3/9
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
transformed_input_x = scaler.transform(input_x)

# define and fit the model
miniBatchKMeans = MiniBatchKMeans(init="k-means++", n_clusters=2).fit(transformed_input_x) #miniBatchKMeans = MiniBatchKMeans(n_clusters=2).fit(input_x)
input_score = miniBatchKMeans.score(transformed_input_x) #input_score = miniBatchKMeans.score(np.array(input_x)[0:1])
input_cluster_centers_ = miniBatchKMeans.cluster_centers_
print(input_score)
print(input_cluster_centers_)

input_x["Conscientious"] = miniBatchKMeans.predict(transformed_input_x) #input_x["Conscientious"] = miniBatchKMeans.predict(input_x)
input_x["Conscientious"] = input_x["Conscientious"].astype("int")
input_x["pId"] = input_data["pId"]

df = DataFrame()
for i in range(2):
    df['p' + str(i)] = 0

df[['p0', 'p1']] = soft_clustering_weights(transformed_input_x, input_cluster_centers_)
df['confidence'] = np.max(df[['p0', 'p1']].values, axis = 1)
input_x["Confidence"] = df['confidence']

pyplot.figure(figsize=(15,7))
pyplot.hist(df['confidence'][input_x["Conscientious"]==0], bins=50, label='Cluster Conscientious', alpha=0.7, color='b')
pyplot.hist(df['confidence'][input_x["Conscientious"]==1], bins=50, label='Cluster None-Conscientious', alpha=0.7, color='r')
pyplot.xlabel('Calculated Probability', fontsize=25)
pyplot.ylabel('Number of records', fontsize=25)
pyplot.legend(fontsize=15)
pyplot.tick_params(axis='both', labelsize=25, pad=5)
pyplot.show() 

print(miniBatchKMeans.get_params(deep=True))
print(miniBatchKMeans.labels_)

# get probability score of each sample
loss = log_loss(input_data['Conscientious'], input_x['Conscientious'])
print(loss)

input_means_labels = pairwise_distances_argmin(transformed_input_x.copy(order='C'), input_cluster_centers_.copy(order='C'))
print(input_means_labels)

# ----------- miniBatchKMeans Cluster plot of transformed_input_x
fig = pyplot.figure(figsize=(10, 10))
ax = fig.add_subplot(1, 1, 1)
for i in range(c_num - 1):
	ax.plot(transformed_input_x[input_means_labels == 0, i], transformed_input_x[input_means_labels == 0, i+1], "w", markerfacecolor='b', marker=".", zorder=1)
	ax.plot(transformed_input_x[input_means_labels == 1, i], transformed_input_x[input_means_labels == 1, i+1], "w", markerfacecolor='r', marker=".", zorder=1)
cluster_center = input_cluster_centers_[0]
ax.plot(cluster_center[0], cluster_center[1], "o", markerfacecolor='b', markeredgecolor="k", markersize=6, zorder=2)
cluster_center = input_cluster_centers_[1]
ax.plot(cluster_center[0], cluster_center[1], "o", markerfacecolor='r', markeredgecolor="k", markersize=6, zorder=2)
ax.set_title("MiniBatchKMeans")
ax.set_xticks(())
ax.set_yticks(())
#pyplot.text(-3.5, 1.8, "inertia: %f" % (miniBatchKMeans.inertia_))
pyplot.show()

colors = {0:'b', 1:'r'}
pyplot.scatter(x=input_x['Conscientious'], y=input_x['pId'], alpha=0.5, c=input_x['Conscientious'].map(colors))
pyplot.show()

ax2 = input_x.plot.scatter(x='pId',  y='Confidence', alpha=0.5, c=input_x['Conscientious'].map(colors))
pyplot.show()

# ----------- miniBatchKMeans Cluster IDs plot with heighest confidence
_ids = [ 1,2,3,4,5,6,7,10,13,14,15,16,17,18,19,20,31,34]
for id in _ids:
	temp = input_x.loc[input_x["pId"] == id]
	max_confi = temp['Confidence'].max()
	highest_confidet_index = temp[temp.Confidence == max_confi].index.tolist()[0]
	temp['Conscientious'] = temp.at[highest_confidet_index, 'Conscientious']
	input_x.Conscientious[input_x.pId == id] = temp['Conscientious'].values[0]
	
ax2 = input_x.plot.scatter(x='Conscientious',  y='pId', c=input_x['Conscientious'].map(colors))
pyplot.show()

# ------- display roc_auc curve
model_roc_auc = roc_auc_score(input_data["Conscientious"], miniBatchKMeans.predict(transformed_input_x))
fpr, tpr, thresholds = roc_curve(input_data["Conscientious"], input_x["Confidence"])
pyplot.figure()
pyplot.plot(fpr, tpr, label='Mini-Batch-K-Means-Model (area = %0.2f)' % model_roc_auc)
pyplot.plot([0, 1], [0, 1],'r--')
pyplot.xlim([0.0, 1.0])
pyplot.ylim([0.0, 1.05])
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
pyplot.title('Receiver operating characteristic')
pyplot.legend(loc="lower right")
pyplot.savefig('Mini-Batch-K-Means-Model ROC curve')
pyplot.show()

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