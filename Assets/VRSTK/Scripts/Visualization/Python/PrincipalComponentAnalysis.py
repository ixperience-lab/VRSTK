# Source-Link: https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.cluster import MiniBatchKMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame
import numpy as np
import numpy.matlib
import sys
import os
from os.path import exists

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

def plot_roc_curve(true_positive_rate, false_positive_rate, legend_label, title, file_name, show=False, save=False):
    plt.figure(figsize=(15,10))
    plt.plot(false_positive_rate, true_positive_rate, label=legend_label)
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.title(title, fontsize=18)
    plt.grid(which="major", alpha=0.6)
    plt.grid(which="minor", alpha=0.6)
    plt.legend(bbox_to_anchor=(1, 0.5), loc='center left', fontsize=16)
    plt.tight_layout() 
    if save:
        plt.savefig(file_name)
    if show:
        plt.show()
    plt.close()

def plot_data_cluster(data, conscientious_indeces_list, none_conscientious_indeces_list, title, file_name, show=False, save=False):
    plt.figure(figsize=(15,10))
    plt.scatter(data[conscientious_indeces_list, 0], data[conscientious_indeces_list, 1], c="b")
    plt.scatter(data[none_conscientious_indeces_list, 0], data[none_conscientious_indeces_list, 1], c="r")
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    #plt.grid(which="major", alpha=0.6)
    #plt.grid(which="minor", alpha=0.6)
    plt.title(title, fontsize=18)
    plt.tight_layout() 
    if save:
        plt.savefig(file_name)
    if show:
        plt.show()
    plt.close()

def write_report(file_name, content):
    if exists(file_name):
        os.remove(file_name)
    file = open(file_name, "w")
    file.write(content)
    file.close()

# input_data_type = { all_sensors = 0, ecg = 1, eda = 2, eeg = 3, eye = 4, pages = 5 }
input_data_type = 0

# read csv train data as pandas data frame
input_data = pd.read_csv("All_Participents_Clusterd_WaveSum_DataFrame.csv", sep=";", decimal=',')			# plan of sensors weighting:
if input_data_type == 1: 
	input_data = pd.read_csv("All_Participents_ECG_Clusterd_WaveSum_DataFrame.csv", sep=";", decimal=',') 		# weight with 2/10
if input_data_type == 2: 
	input_data = pd.read_csv("All_Participents_EDA_Clusterd_WaveSum_DataFrame.csv", sep=";", decimal=',') 		# weight with 1/10
if input_data_type == 3: 
	input_data = pd.read_csv("All_Participents_EEG_Clusterd_WaveSum_DataFrame.csv", sep=";", decimal=',') 		# weight with 1/10
if input_data_type == 4: 
	input_data = pd.read_csv("All_Participents_EYE_Clusterd_WaveSum_DataFrame.csv", sep=";", decimal=',') 		# weight with 2/10
if input_data_type == 5: 
	input_data = pd.read_csv("All_Participents_PAGES_Clusterd_WaveSum_DataFrame.csv", sep=";", decimal=',') 	# weight with 4/10

# updates Conscientious to subjektive 
# for i in range(input_data.shape[1]):
#     if input_data['pId'].values[i] == 14 or input_data['pId'].values[i] == 15 or input_data['pId'].values[i] == 16: # or load_test_data['pId'].values[i] == 28:
#         input_data['Conscientious'].values[i] = 1

# ------- fitler columns of train data
train_data = input_data.drop(columns=['Conscientious', 'time', 'pId'])
# count rows and columns
#r_num = train_data.shape[0]
#print(r_num)
c_num = train_data.shape[1]
print(c_num)

# ------ Normalizing
# Separating out the features
x = train_data.loc[:, :].values
# Separating out the target
y_result_output = np.array(input_data[["Conscientious"]].values.flatten())
# Standardizing the features of train data
x = StandardScaler().fit_transform(x)

# set sensor and validity score weights
weight_ecg = 1       #train_data.loc[:,1:26]                                 -> count() = 26
weight_eda = 1       #train_data.loc[:,27:31]                                -> count() = 5
weight_eeg = 1       #train_data.loc[:,32:107]  , train_data.loc[:,141:145]  -> count() = 76, 5
weight_eye = 1       #train_data.loc[:,108:117] , train_data.loc[:,130:137]  -> count() = 10, 8
weight_pages = 1       #train_data.loc[:,118:129] , train_data.loc[:,138:140]  -> count() = 12, 3
# ------
# weight_ecg = 1/5       #train_data.loc[:,1:26]                                 -> count() = 26
# weight_eda = 2/5       #train_data.loc[:,27:31]                                -> count() = 5
# weight_eeg = 1/5       #train_data.loc[:,32:107]  , train_data.loc[:,141:145]  -> count() = 76, 5
# weight_eye = 3/5     #train_data.loc[:,108:117] , train_data.loc[:,130:137]  -> count() = 10, 8
# weight_pages = 1       #train_data.loc[:,118:129] , train_data.loc[:,138:140]  -> count() = 12, 3

if input_data_type == 0:
	x[:,0:26]    = x[:,0:26]    * weight_ecg
	x[:,26:31]   = x[:,26:31]   * weight_eda
	x[:,31:107]  = x[:,31:107]  * weight_eeg
	x[:,152:157] = x[:,152:157] * weight_eeg
	x[:,107:129] = x[:,107:129] * weight_eye
	x[:,141:149] = x[:,141:149] * weight_eye
	x[:,129:141] = x[:,129:141] * weight_pages
	x[:,149:152] = x[:,149:152] * weight_pages

if input_data_type == 1:
	x[:,:] = x[:,:] * weight_ecg
if input_data_type == 2:
	x[:,:] = x[:,:] * weight_eda
if input_data_type == 3:
	x[:,:] = x[:,:] * weight_eeg
if input_data_type == 4:
	x[:,:] = x[:,:] * weight_eye
if input_data_type == 5:
	x[:,:] = x[:,:] * weight_pages

print("Create output directory")
# --- create dir
mode = 0o666
if not os.path.exists("./output"):
    os.mkdir("./output", mode)
path = "./output/Principal_Component_Analysis_{}".format(input_data_type)
if not os.path.exists(path):
    os.mkdir(path, mode)

print("------ Principal Component Analysis test explainable variance of given features in train data")
# test explainable variance of given features
pca = PCA()
print(pca.get_params(True))
test_principal_components = pca.fit_transform(x)
test_pca_explained_variance = pca.explained_variance_
print(test_pca_explained_variance)
print(test_pca_explained_variance.shape)

content = "pca_explained_variance with all features: {}\n".format(test_pca_explained_variance)

test_pca_explained_variance_ratio = pca.explained_variance_ratio_
print(test_pca_explained_variance_ratio)
print(test_pca_explained_variance_ratio.shape)
content = "{}\npca_explained_variance_ratio with all features: {}\n".format(content, test_pca_explained_variance_ratio)

kaiser_rule_mean = np.mean(pca.singular_values_)
print(kaiser_rule_mean)
print(pca.singular_values_)
content = "{}\nsingular_values_ of all all features: {}\n".format(content, pca.singular_values_)
content = "{}\nkaiser_rule_mean of all singular_values_: {}\n".format(content, kaiser_rule_mean)

component_counter = 1
for factor in pca.singular_values_:
    if factor > kaiser_rule_mean:
        component_counter += 1

print(component_counter)
content = "{}\ncomponent_counter : {}\n".format(content, component_counter)

print(range(c_num - 2))

plt.figure(figsize=(15, 10))
plt.grid(which="major", alpha=0.6)
plt.grid(which="minor", alpha=0.6)
plt.bar(range(157), test_pca_explained_variance, align='center', label='individual variance', color="b")
plt.legend(bbox_to_anchor=(1, 0.5), loc='center left', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.ylabel('Variance', fontsize=16)
plt.xlabel('Principal components', fontsize=16)
plt.title("Principal Component Analysis explained variance train data", fontsize=18)
plt.tight_layout() 
file_name = '{}/Tested_pca_explained_variance_on_train_data_plot.png'.format(path)
plt.savefig(file_name)
#plt.show()
plt.close()

plt.figure(figsize=(15, 10))
plt.grid(which="major", alpha=0.6)
plt.grid(which="minor", alpha=0.6)
plt.bar(range(157), test_pca_explained_variance_ratio, align='center', label='individual variance %', color="b")
plt.legend(bbox_to_anchor=(1, 0.5), loc='center left', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.ylabel('Variance ratio', fontsize=16)
plt.xlabel('Principal components', fontsize=16)
plt.title("Principal Component Analysis explained variance ratio train data", fontsize=18)
plt.tight_layout() 
file_name = '{}/Tested_pca_explained_variance_on_ratio_train_data_plot.png'.format(path)
plt.savefig(file_name)
#plt.show()
plt.close()

print("------ Principal Component Analysis n_components={} of train data".format(component_counter))
# ------ Principal Component Analysis n_components=2 of train data
pca = PCA(n_components=component_counter)
print(pca.get_params(True))
principalComponents = pca.fit_transform(x)
print(pca.score(x)) # Debug only
print(pca.explained_variance_ratio_)  # Debug only
pca_explained_variance = pca.explained_variance_
print(pca_explained_variance)

plt.figure(figsize=(15, 10))
plt.grid(which="major", alpha=0.6)
plt.grid(which="minor", alpha=0.6)
plt.bar(range(component_counter), pca_explained_variance, align='center', label='individual variance', color="b")
plt.legend(bbox_to_anchor=(1, 0.5), loc='center left', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.ylabel('Variance', fontsize=16)
plt.xlabel('Principal components', fontsize=16)
plt.title("Principal Component Analysis explained variance ratio dimension reduced train data", fontsize=18)
plt.tight_layout() 
file_name = '{}/Tested_pca_explained_variance_on_dimension_reduced_train_data_plot.png'.format(path)
plt.savefig(file_name)
plt.close()

conscientious_indeces = input_data.index[input_data['Conscientious'] == 0]
none_conscientious_indeces = input_data.index[input_data['Conscientious'] == 1]
plt.figure(figsize=(15,10))
plt.scatter(principalComponents[conscientious_indeces.tolist(),0], principalComponents[conscientious_indeces.tolist(),1], c="b")
plt.scatter(principalComponents[none_conscientious_indeces.tolist(),0], principalComponents[none_conscientious_indeces.tolist(),1], c="r")
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.title('Principal Component Analysis train data n_components={}'.format(principalComponents.shape[1]), fontsize=18)
plt.tight_layout() 
file_name = '{}/True_principal_components_train_data_plot.png'.format(path)
plt.savefig(file_name)
plt.close()

file_name = '{}/PCA_report.txt'.format(path)
write_report(file_name, content)