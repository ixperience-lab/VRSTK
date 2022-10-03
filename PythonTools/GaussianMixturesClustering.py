# gaussian mixture clustering
# weblink: https://machinelearningmastery.com/clustering-algorithms-with-python/
from numpy import unique
from numpy import where
import numpy as np
import numpy.matlib
from sklearn.datasets import make_classification
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from matplotlib import pyplot
import pandas as pd

# read csv input file
input_data = pd.read_csv("All_Participents_Clusterd_WaveSum_DataFrame.csv", sep=";", decimal=',')			# plan of sensors weighting:
#input_data = pd.read_csv("All_Participents_ECG_Clusterd_WaveSum_DataFrame.csv", sep=";", decimal=',') 		# weight with 1/9
#input_data = pd.read_csv("All_Participents_EDA_Clusterd_WaveSum_DataFrame.csv", sep=";", decimal=',') 		# weight with 2/9
#input_data = pd.read_csv("All_Participents_EEG_Clusterd_WaveSum_DataFrame.csv", sep=";", decimal=',') 		# weight with 1/9
#input_data = pd.read_csv("All_Participents_EYE_Clusterd_WaveSum_DataFrame.csv", sep=";", decimal=',') 		# weight with 2/9
#input_data = pd.read_csv("All_Participents_PAGES_Clusterd_WaveSum_DataFrame.csv", sep=";", decimal=',') 	# weight with 3/9
# drop columns conscientious, time, pId
input_x = input_data.drop(columns=['Conscientious', 'time', 'pId'])
# count rows and columns
#r_num = input_x.shape[0]
#print(r_num)
c_num = input_x.shape[1]
#print(c_num)

# nomalize input data to create more varianz in the data (because of statictical values like mean, std, max, counts, ...)
scaler = StandardScaler()
scaler.fit(input_x)
#print(scaler.mean_)
transformed_input_x = scaler.transform(input_x)

gaussianMixture = GaussianMixture(n_components=2, init_params='k-means++').fit(transformed_input_x)
input_score = gaussianMixture.score(transformed_input_x) #input_score = miniBatchKMeans.score(np.array(input_x)[0:1])
input_score_sampels = gaussianMixture.score_samples(transformed_input_x)
input_mean = gaussianMixture.means_
print(input_score)
print(input_score_sampels)
print(input_mean)

input_x["Conscientious"] = gaussianMixture.predict(transformed_input_x)
input_x["Conscientious"] = input_x["Conscientious"].astype("int")

prediction=gaussianMixture.predict_proba(transformed_input_x)#[:,1]
print(prediction)
input_x["Confidence"] = np.max(prediction, axis = 1)

pyplot.figure(figsize=(15,7))
#pyplot.hist(prediction[:,1][input_x['Conscientious']==0], bins=50, label='Cluster Conscientious', alpha=0.7, color='b')
pyplot.hist(input_x['Confidence'][input_x["Conscientious"]==0], bins=50, label='Cluster Conscientious', alpha=0.7, color='b')
pyplot.hist(input_x['Confidence'][input_x["Conscientious"]==1], bins=50, label='Cluster None-Conscientious', alpha=0.7, color='r')
pyplot.xlabel('Calculated Probability', fontsize=25)
pyplot.ylabel('Number of records', fontsize=25)
pyplot.legend(fontsize=15)
pyplot.tick_params(axis='both', labelsize=25, pad=5)
pyplot.show() 

print(gaussianMixture.get_params(deep=True))

# get probability score of each sample
loss = log_loss(input_data['Conscientious'], input_x['Conscientious'])
print(loss)

input_means_labels = pairwise_distances_argmin(transformed_input_x.copy(order='C'), input_mean.copy(order='C'))
print(input_means_labels)

# GaussianMixture
fig = pyplot.figure(figsize=(10, 10))
colors = ["#4EACC5", "#FF9C34"]
ax = fig.add_subplot(1, 1, 1)
# for k, col in zip(range(2), colors):
#     my_members = input_means_labels == k
#     cluster_center = input_mean[k]
#     for i in range(c_num - 1):
#         ax.plot(transformed_input_x[my_members, i], transformed_input_x[my_members, i+1], "w", markerfacecolor=col, marker=".", zorder=1)
#     ax.plot(cluster_center[0], cluster_center[1], "o", markerfacecolor=col, markeredgecolor="k", markersize=6, zorder=2)
for i in range(c_num - 1):
	ax.plot(transformed_input_x[input_means_labels == 0, i], transformed_input_x[input_means_labels == 0, i+1], "w", markerfacecolor='b', marker=".", zorder=1)
	ax.plot(transformed_input_x[input_means_labels == 1, i], transformed_input_x[input_means_labels == 1, i+1], "w", markerfacecolor='r', marker=".", zorder=1)
cluster_center = input_mean[0]
ax.plot(cluster_center[0], cluster_center[1], "o", markerfacecolor='b', markeredgecolor="k", markersize=6, zorder=2)
cluster_center = input_mean[1]
ax.plot(cluster_center[0], cluster_center[1], "o", markerfacecolor='r', markeredgecolor="k", markersize=6, zorder=2)
ax.set_title("GaussianMixture")
ax.set_xticks(())
ax.set_yticks(())
#pyplot.text(-3.5, 1.8, "inertia: %f" % (gaussianMixture.inertia_))
pyplot.show()

input_x["pId"] = input_data["pId"]

colors = {0:'b', 1:'r'}
pyplot.scatter(x=input_x['Conscientious'], y=input_x['pId'], alpha=0.5, c=input_x['Conscientious'].map(colors))
pyplot.show()

ax2 = input_x.plot.scatter(x='pId',  y='Confidence', alpha=0.5, c=input_x['Conscientious'].map(colors))
pyplot.show()

# ----------- gaussianMixture Cluster IDs plot with heighest confidence
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
model_roc_auc = roc_auc_score(input_data["Conscientious"], gaussianMixture.predict(transformed_input_x))
fpr, tpr, thresholds = roc_curve(input_data["Conscientious"], input_x["Confidence"])
pyplot.figure()
pyplot.plot(fpr, tpr, label='Gaussian-Mixtures-Model (area = %0.2f)' % model_roc_auc)
pyplot.plot([0, 1], [0, 1],'r--')
pyplot.xlim([0.0, 1.0])
pyplot.ylim([0.0, 1.05])
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
pyplot.title('Receiver operating characteristic')
pyplot.legend(loc="lower right")
pyplot.savefig('Gaussian-Mixtures-Model ROC curve')
pyplot.show()

#-----------------------------------------------------------------------------------------
# read csv input file
#input_data = pd.read_csv("All_Participents_Mean_DataFrame.csv", sep=";", decimal=',')
#input_data = pd.read_csv("All_Participents_WaveSum_Mean_DataFrame.csv", sep=";", decimal=',')

# define the model
#model = GaussianMixture(n_components=2)

#input_data["Cluster"] = model.fit_predict(input_data)
#input_data["Cluster"] = input_data["Cluster"].astype("int")
#print(input_data.head(1)) 

#ax2 = input_data.plot.scatter(x='Cluster', y='pId', c='Cluster', colormap='viridis')
# show the plot
#pyplot.show()