from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.cluster import MiniBatchKMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame
import numpy as np
import numpy.matlib
import os
import shutil
import sys

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

# read cvs test data
load_test_data = pd.read_csv("All_Participents_Condition-C_WaveSum_DataFrame.csv", sep=";", decimal=',')			# plan of sensors weighting:
if input_data_type == 1: 
	load_test_data = pd.read_csv("All_Participents_Condition-C_ECG_WaveSum_DataFrame.csv", sep=";", decimal=',') 		# weight with 2/10
if input_data_type == 2: 
	load_test_data = pd.read_csv("All_Participents_Condition-C_EDA_WaveSum_DataFrame.csv", sep=";", decimal=',') 		# weight with 1/10
if input_data_type == 3: 
	load_test_data = pd.read_csv("All_Participents_Condition-C_EEG_WaveSum_DataFrame.csv", sep=";", decimal=',') 		# weight with 1/10
if input_data_type == 4: 
	load_test_data = pd.read_csv("All_Participents_Condition-C_EYE_WaveSum_DataFrame.csv", sep=";", decimal=',') 		# weight with 2/10
if input_data_type == 5: 
	load_test_data = pd.read_csv("All_Participents_Condition-C_PAGES_WaveSum_DataFrame.csv", sep=";", decimal=',') 	# weight with 4/10

# ------- fitler columns of train data
train_data = input_data.drop(columns=['Conscientious', 'time', 'pId'])
# add validity score ['DegTimeLowQuality', 'EvaluatedGlobalTIMERSICalc'] as weight to features
#train_data.loc[train_data.colums[~train_data.columns.isin(['DegTimeLowQuality', 'EvaluatedGlobalTIMERSICalc'])]]
#train_data.loc[train_data['DegTimeLowQuality'] > 0 :train_data.columns[~train_data.columns.isin(['DegTimeLowQuality', 'EvaluatedGlobalTIMERSICalc'])]] *= 2.0
exc_cols = [col for col in train_data.columns if col not in ['DegTimeLowQuality', 'EvaluatedGlobalTIMERSICalc']]
#train_data[train_data['DegTimeLowQuality'] > 0 : exc_cols] *= 2.0
#train_data[train_data['EvaluatedGlobalTIMERSICalc'] >= 1 : exc_cols] *= 2.0
r_num = train_data.shape[0]
c_num = train_data.shape[1]
temp = train_data.copy()
#print(c_num)
#print(train_data.iloc[1, 122])
train_data.loc[train_data.DegTimeLowQuality > 0, exc_cols] *= 2.0
train_data.loc[train_data.EvaluatedGlobalTIMERSICalc >= 1, exc_cols] *= 2.0

# index_degtimelowquality = train_data.columns.get_loc('DegTimeLowQuality')
# index_evaluatedglobalTIMERSIcalc = train_data.columns.get_loc('EvaluatedGlobalTIMERSICalc')
# for r in range(r_num):
#     for c in exc_cols:
#         if train_data.iloc[r , index_degtimelowquality] > 0 :
#             train_data.iloc[r, train_data.columns.get_loc(c)] *= 2.0
#         if train_data.iloc[r , index_evaluatedglobalTIMERSIcalc] >= 1 :
#             train_data.iloc[r, train_data.columns.get_loc(c)] *= 2.0
    

print(train_data.head(1))

# -------  filter columns of test data 
test_data = load_test_data.drop(columns=['time', 'pId'])
r_num_test_data = test_data.shape[0]
test_x = test_data.iloc[:, :].values

# ------ Normalizing
# Separating out the features
x = train_data.loc[:, :].values
# Separating out the target
y = np.array(input_data[["Conscientious"]].values.flatten())
# Standardizing the features of train data
transformed_x = StandardScaler().fit_transform(x)
# Standardizing the features of Test data
transformed_test_x = StandardScaler().fit_transform(test_x)

print("------- Mini-Batch-K-Means Model")
# ------- Mini-Batch-K-Means Model
# create dir
mode = 0o666

path = "./output"
if not os.path.exists(path):
    os.mkdir(path, mode)

path = "./output/K-Means-Feature-To-Feature-Compare_Validity_Score_{}".format(input_data_type)
if os.path.exists(path):
    shutil.rmtree(path, ignore_errors=True)

if not os.path.exists(path):
    os.mkdir(path, mode)

path_p = "./output/K-Means-Feature-To-Feature-Compare_Validity_Score_{}/pId-Confidence".format(input_data_type)
if not os.path.exists(path_p):
    os.mkdir(path_p, mode)

path_p = "./output/K-Means-Feature-To-Feature-Compare_Validity_Score_{}/Conscientious-pId".format(input_data_type)
if not os.path.exists(path_p):
    os.mkdir(path_p, mode)

path_p = "./output/K-Means-Feature-To-Feature-Compare_Validity_Score_{}/Conscientious-pId-highest_confidet".format(input_data_type)
if not os.path.exists(path_p):
    os.mkdir(path_p, mode)

path_p = "./output/K-Means-Feature-To-Feature-Compare_Validity_Score_{}/ROC-curve".format(input_data_type)
if not os.path.exists(path_p):
    os.mkdir(path_p, mode)

mbkm_train_data = train_data.copy()
miniBatchKMeans = MiniBatchKMeans(init="k-means++", n_clusters=2)
for i in range(c_num):
    c1 = transformed_x[:,i]
    print('======================================================= ' + str(i))
    if i < (c_num - 1) :
        for j in range(i+1,c_num):
            print('=================================================================== ' + str(j))
            c2 = transformed_x[:,j]
            tmp_input_x = pd.DataFrame(c1)
            tmp_input_x[1] = c2
            miniBatchKMeans.fit(tmp_input_x)
            tmp_score = miniBatchKMeans.score(tmp_input_x)
            tmp_cluster_centers_ = miniBatchKMeans.cluster_centers_
            result_output_x = tmp_input_x.copy()
            result_output_x["Conscientious"] = miniBatchKMeans.predict(tmp_input_x)
            result_output_x["Conscientious"] = result_output_x["Conscientious"].astype("int")
            result_output_x["pId"] = input_data["pId"]
            df = DataFrame()
            for k in range(2):
                df['p' + str(k)] = 0
            df[['p0', 'p1']] = soft_clustering_weights(tmp_input_x, tmp_cluster_centers_)
            df['confidence'] = np.max(df[['p0', 'p1']].values, axis = 1)
            result_output_x["Confidence"] = df['confidence']
            generatet_file_name = "{}/Conscientious-pId/Features_Analyse_with_Features_Conscientious-pId_{}_{}{}".format(path, i, j,".png")
            print(generatet_file_name)
            colors = {0:'b', 1:'r'}
            plt.scatter(x=result_output_x['Conscientious'], y=result_output_x['pId'], alpha=0.5, c=result_output_x['Conscientious'].map(colors))
            plt.savefig(generatet_file_name)
            plt.close()
            generatet_file_name = "{}/pId-Confidence/Features_Analyse_with_Features_pId-Confidence_{}_{}{}".format(path, i, j,".png")
            ax2 = result_output_x.plot.scatter(x='pId',  y='Confidence', alpha=0.5, c=result_output_x['Conscientious'].map(colors))
            plt.savefig(generatet_file_name)
            plt.close()
            # ----------- miniBatchKMeans Cluster IDs plot with heighest confidence
            _ids = [ 1,2,3,4,5,6,7,10,13,14,15,16,17,18,19,20,31,34]
            for id in _ids:
                temp = result_output_x[result_output_x["pId"] == id].copy()
                max_confi = temp['Confidence'].max()
                highest_confidet_index = temp[temp.Confidence == max_confi].index.tolist()[0]
                highest_confidet = temp.at[highest_confidet_index, 'Conscientious']
                result_output_x.loc[result_output_x.pId == id, 'Conscientious'] = highest_confidet
                #temp.loc['Conscientious'] = temp.at[highest_confidet_index, 'Conscientious']
                #result_output_x.Conscientious.loc[result_output_x.pId == id] = temp['Conscientious'].values[0]
            
            generatet_file_name = "{}/Conscientious-pId-highest_confidet/Features_Analyse_with_Features_Conscientious-pId-highest_confidet_{}_{}{}".format(path, i, j,".png")
            ax2 = result_output_x.plot.scatter(x='Conscientious',  y='pId', c=result_output_x['Conscientious'].map(colors))
            plt.savefig(generatet_file_name)
            plt.close()
            # ------- display roc_auc curve
            model_roc_auc = roc_auc_score(input_data["Conscientious"], miniBatchKMeans.predict(tmp_input_x))
            fpr, tpr, thresholds = roc_curve(input_data["Conscientious"], result_output_x["Confidence"])
            plt.figure()
            plt.plot(fpr, tpr, label='Mini-Batch-K-Means-Model (area = %0.2f)' % model_roc_auc)
            plt.plot([0, 1], [0, 1],'r--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver operating characteristic')
            plt.legend(loc="lower right")
            generatet_file_name = "{}/ROC-curve/Features_Analyse_with_Features_ROC-curve_{}_{}{}".format(path, i, j,".png")
            plt.savefig(generatet_file_name)
            plt.close()
            #result_df = pd.concat([result_df, pd.DataFrame({'column_index': i, 'compare_column_index' : j, 'score' : tmp_score, 'cluster_centers':  list(tmp_cluster_centers_)})])


