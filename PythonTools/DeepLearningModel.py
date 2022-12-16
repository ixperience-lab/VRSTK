from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import precision_recall_curve, accuracy_score, f1_score, roc_auc_score, roc_curve, confusion_matrix, classification_report
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.model_selection import train_test_split
from pytorch_tabnet.tab_model import TabNetClassifier
from pytorch_tabnet.augmentations import ClassificationSMOTE
import torch
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame
import numpy as np
import numpy.matlib
import os
from os.path import exists

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
    plt.title(title, fontsize=18)
    plt.tight_layout() 
    if save:
        plt.savefig(file_name)
    if show:
        plt.show()
    plt.close()

def write_matrix_and_report_to_file(file_name, content):
    if exists(file_name):
        os.remove(file_name)
    file = open(file_name, "w")
    file.write(content)
    file.close()

# 0 = all (with pages); 1= ecg, eda, eeg, eye; 2 = ecg, eda, eeg; 3 = ecg, eda; 4 = ecg 
input_data_type = 4     
intervals = 3 
# 0=scaled; 1=scaled+pca; 2=scaled+tSNE
dimensions = [0, 1, 2]               
p_thresholds = [0.3, 0.5, 0.7]
# 3=Selection_over_5_ratio; 56=Kaiser_Rule;
n_components=56 

# read csv train data as pandas data frame
input_data = pd.read_csv("All_Participents_Clusterd_WaveSum_DataFrame.csv", sep=";", decimal=',')		
if input_data_type == 1:
    intervals = 2 
    dimensions = [0, 1, 2] #0=scaled; 1=scaled+pca; 2=scaled+tSNE
    p_thresholds = [0.3, 0.5, 0.7]
    n_components=3 # 3=Selection_over_5_ratio; 56=Kaiser_Rule;
    input_data = pd.read_csv("All_Participents_Clusterd_WaveSum_DataFrame_01.csv", sep=";", decimal=',')
if input_data_type == 2:
    intervals = 2 
    dimensions = [0, 1, 2] #0=scaled; 1=scaled+pca; 2=scaled+tSNE
    p_thresholds = [0.3, 0.5, 0.7]
    n_components=3 # 3=Selection_over_5_ratio; 56=Kaiser_Rule;
    input_data = pd.read_csv("All_Participents_Clusterd_WaveSum_DataFrame_02.csv", sep=";", decimal=',')
if input_data_type == 3:
    intervals = 2 
    dimensions = [0, 1, 2] #0=scaled; 1=scaled+pca; 2=scaled+tSNE
    p_thresholds = [0.3, 0.5, 0.7]
    n_components=3 # 3=Selection_over_5_ratio; 56=Kaiser_Rule;
    input_data = pd.read_csv("All_Participents_Clusterd_WaveSum_DataFrame_03.csv", sep=";", decimal=',')
if input_data_type == 4:
    intervals = 2 
    dimensions = [0, 1, 2] #0=scaled; 1=scaled+pca; 2=scaled+tSNE
    p_thresholds = [0.3, 0.5, 0.7]
    n_components=3 # 3=Selection_over_5_ratio; 56=Kaiser_Rule;
    input_data = pd.read_csv("All_Participents_Clusterd_WaveSum_DataFrame_04.csv", sep=";", decimal=',')

# ------- fitler columns of train data
train_data = input_data.drop(columns=['Conscientious', 'time', 'pId'])

# count rows and columns
c_num = train_data.shape[1]
print(c_num)

# ------ Normalizing
# Separating out the features
x_train = train_data.loc[:, :].values
# Separating out the target
y_result_output = np.array(input_data[["Conscientious"]].values.flatten())
print(y_result_output)

transformed_train_x = StandardScaler().fit_transform(x_train)

print("Create output directory")
# --- create dir
mode = 0o666
if not os.path.exists("./output"):
    os.mkdir("./output", mode)
if not os.path.exists("./output/DeepLearning-Model_{}".format(input_data_type)):
    os.mkdir("./output/DeepLearning-Model_{}".format(input_data_type), mode)

transformed_train_x_temp = transformed_train_x

for interval in range(1, intervals):
    for dimension in dimensions:
        for p_threshold in p_thresholds:
            
            path = "./output/DeepLearning-Model_{}/DeepLearning-Model_{}_{}_{}_{}_{}".format(input_data_type, input_data_type, n_components, interval, dimension, int(p_threshold*10))
            if not exists(path):
                os.mkdir(path, mode)
            
            propability_threshold = p_threshold
            transformed_train_x = transformed_train_x_temp

            print("------ Transformed (True) train data")
            # ------ Transformed (True) train data
            conscientious_indeces = input_data.index[input_data['Conscientious'] == 0]
            none_conscientious_indeces = input_data.index[input_data['Conscientious'] == 1]
            file_name = '{}/Transformed_train_data_plot.png'.format(path)
            plot_data_cluster(transformed_train_x, conscientious_indeces.tolist(), none_conscientious_indeces.tolist(), 
                            'Transformed (True) train data  plot', file_name, show=False, save=True)

            if dimension == 1:
                print("------ T-Distributed Stochastic Neighbor Embedding n_components=2 of (True) train data ")
                # ------ T-Distributed Stochastic Neighbor Embedding n_components=2 of train data
                tsne_model = TSNE(n_components=3, learning_rate=500.0 , init='pca', perplexity=30.0)
                transformed_train_x = tsne_model.fit_transform(transformed_train_x)
                print(transformed_train_x.shape)
                conscientious_indeces = input_data.index[input_data['Conscientious'] == 0]
                none_conscientious_indeces = input_data.index[input_data['Conscientious'] == 1]
                file_name = '{}/tsne_True_train_data_plot.png'.format(path)
                plot_data_cluster(transformed_train_x, conscientious_indeces.tolist(), none_conscientious_indeces.tolist(), 
                                'T-Distributed Stochastic Neighbor Embedding train data n_components=3', file_name, show=False, save=True)

            if dimension == 2:
                print("------ Principal Component Analysis n_components={} of train data".format(n_components))
                # ------ Principal Component Analysis n_components=3 of train data
                pca = PCA(n_components=n_components)
                transformed_train_x = pca.fit_transform(transformed_train_x)
                print(pca.explained_variance_ratio_)  # Debug only

                conscientious_indeces = input_data.index[input_data['Conscientious'] == 0]
                none_conscientious_indeces = input_data.index[input_data['Conscientious'] == 1]
                plt.figure(figsize=(15,10))
                plt.xticks(fontsize=14)
                plt.yticks(fontsize=14)
                plt.scatter(transformed_train_x[conscientious_indeces.tolist(),0], transformed_train_x[conscientious_indeces.tolist(),1], c="b")
                plt.scatter(transformed_train_x[none_conscientious_indeces.tolist(),0], transformed_train_x[none_conscientious_indeces.tolist(),1], c="r")
                plt.title('Principal Component Analysis train data n_components={}'.format(n_components), fontsize=18)
                plt.tight_layout() 
                file_name = '{}/True_principal_components_train_data_plot.png'.format(path)
                plt.savefig(file_name)
                plt.close()

            print("------- -Model")
            # ------- -Model

            # ---- hyper parameters
            iterration = 2
            learning_rate = 0.001
            step_size = 10
            gamma = 0.9
            batch_size = 512
            virtual_batch_size = 64
            num_epochs = 100
            patience = 20
            num_classes = 1
            propability_threshold = 0.7 #0.3 #0.5 # 0.7

            aug = ClassificationSMOTE(p=0.2)
            # ---- tabnet
            #tab_net_model = TabNetClassifier(optimizer_fn=torch.optim.Adam, optimizer_params=dict(lr=learning_rate), scheduler_params={"step_size":step_size, "gamma":gamma},
            #                                 scheduler_fn=torch.optim.lr_scheduler.StepLR, mask_type='entmax')
            temp_path = path
            for iter in range(iterration):
                path = "{}/iterration_{}".format(temp_path, iter)
                if not os.path.exists(path):
                    os.mkdir(path, mode)
                
                tab_net_model = TabNetClassifier(optimizer_fn=torch.optim.Adam, optimizer_params=dict(lr=learning_rate), scheduler_params={"step_size":step_size, "gamma":gamma},
                                            scheduler_fn=torch.optim.lr_scheduler.StepLR, mask_type='entmax')
                
                train_dataframe, validation_dataframe, y_train_true_output, y_validation_true_output = train_test_split(transformed_train_x, 
                                                                                                                        np.array(input_data["Conscientious"].values.flatten()), test_size=0.5,  shuffle=True, 
                                                                                                                        stratify=np.array(input_data["Conscientious"].values.flatten()))

                test_dataframe, validation_dataframe, y_test_true_output, y_validation_true_output = train_test_split(validation_dataframe, 
                                                                                                                    y_validation_true_output, test_size=0.5,  shuffle=True, 
                                                                                                                    stratify=y_validation_true_output)

                print(y_train_true_output)
                print(y_validation_true_output)
                print("Using %d samples for training and %d for validation"  % (len(train_dataframe), len(validation_dataframe)))

                X = train_dataframe.astype('float32')
                print(X.shape)
                Y = y_train_true_output
                print(Y.shape)

                v_X = validation_dataframe.astype('float32')
                print(v_X.shape)
                v_Y = y_validation_true_output
                print(v_Y.shape)
            
                t_X = test_dataframe.astype('float32')
                t_Y = y_test_true_output 
                print(t_X.shape)
                print(t_Y.shape)

                tab_net_model.fit(X_train=X, y_train=Y, eval_set=[(X,Y),(v_X, v_Y)], eval_name=['train', 'valid'], max_epochs=num_epochs , patience=patience, augmentations=aug,
                            batch_size=batch_size, virtual_batch_size=virtual_batch_size, eval_metric=['auc','accuracy'], drop_last=False)

                predictions = tab_net_model.predict_proba(t_X)[:,1]
                print(predictions)

                predictions_transformed = []
                for i, predicted in enumerate(predictions):
                    if predicted > propability_threshold:
                        predictions_transformed.append(1)
                    else:
                        predictions_transformed.append(0)

                # summarize history for loss
                plt.figure(figsize=(15,10))
                plt.plot(tab_net_model.history['loss'])
                plt.title('TabNetClassifier Loss', fontsize=18)
                plt.ylabel('Loss', fontsize=16)
                plt.xlabel('Epoch', fontsize=16)
                plt.legend(['Train'], title_fontsize=18, bbox_to_anchor=(1, 0.5), loc='center left', fontsize=16)
                plt.xticks(fontsize=14)
                plt.yticks(fontsize=14)
                plt.grid(which="major", alpha=0.6)
                plt.grid(which="minor", alpha=0.6)
                plt.tight_layout() 
                file_name = '{}/tabnet_history_loss_plot.png'.format(path)
                plt.savefig(file_name)
                plt.close()

                # plot auc
                plt.figure(figsize=(15,10))
                plt.plot(tab_net_model.history['train_auc'])
                plt.plot(tab_net_model.history['valid_auc'])
                plt.title('TabNetClassifier AUC', fontsize=18)
                plt.ylabel('AUC', fontsize=16)
                plt.xlabel('Epoch', fontsize=16)
                plt.legend(['Train', 'Valid'], title_fontsize=18, bbox_to_anchor=(1, 0.5), loc='center left', fontsize=16)
                plt.xticks(fontsize=14)
                plt.yticks(fontsize=14)
                plt.grid(which="major", alpha=0.6)
                plt.grid(which="minor", alpha=0.6)
                plt.tight_layout() 
                file_name = '{}/tabnet_history_auc_plot.png'.format(path)
                plt.savefig(file_name)
                plt.close()

                # plot learning rates
                plt.figure(figsize=(15,10))
                plt.plot(tab_net_model.history['lr'])
                plt.title('TabNetClassifier Learning Rate', fontsize=18)
                plt.ylabel('Learning Rate', fontsize=16)
                plt.xlabel('Epoch', fontsize=16)
                plt.legend(['Train'], title_fontsize=18, bbox_to_anchor=(1, 0.5), loc='center left', fontsize=16)
                plt.xticks(fontsize=14)
                plt.yticks(fontsize=14)
                plt.grid(which="major", alpha=0.6)
                plt.grid(which="minor", alpha=0.6)
                plt.tight_layout() 
                file_name = '{}/tabnet_history_lr_plot.png'.format(path)
                plt.savefig(file_name)
                plt.close()

                # summarize history for accuracy
                plt.figure(figsize=(15,10))
                plt.plot(tab_net_model.history['train_accuracy'])
                plt.plot(tab_net_model.history['valid_accuracy'])
                plt.title('TabNetClassifier Accuracy', fontsize=18)
                plt.ylabel('Accuracy', fontsize=16)
                plt.xlabel('Epoch', fontsize=16)
                plt.legend(['Train', 'Validition'], title_fontsize=18, bbox_to_anchor=(1, 0.5), loc='center left', fontsize=16)
                plt.xticks(fontsize=14)
                plt.yticks(fontsize=14)
                plt.grid(which="major", alpha=0.6)
                plt.grid(which="minor", alpha=0.6)
                plt.tight_layout() 
                file_name = '{}/tabnet_history_accuracy_plot.png'.format(path)
                plt.savefig(file_name)
                plt.close()

                # conscientious_indeces = t_X.index[predictions_transformed == 0]
                # none_conscientious_indeces = t_X.index[predictions_transformed == 1]
                # file_name = '{}/TabNetClassifier_Predicted_test_data_plot.png'.format(path)
                # plot_data_cluster(t_X, conscientious_indeces.tolist(), none_conscientious_indeces.tolist(), 
                #              'TabNetClassifier predicted test data plot', file_name, show=False, save=True)

                # ------- display roc_auc curve
                roc_auc = roc_auc_score(t_Y, predictions_transformed)
                fpr, tpr, thresholds = roc_curve(t_Y, predictions)
                file_name = '{}/TabNetClassifier_DL-Model_test-data_ROC-curve.png'.format(path)
                plot_roc_curve(true_positive_rate = tpr, false_positive_rate = fpr, legend_label = 'AUC (area = %0.2f)' % roc_auc, 
                            title = 'TabNetClassifier-Model test data', file_name = file_name, show=False, save=True)

                precision, recall, thresholds = precision_recall_curve(t_Y, predictions)
                print(precision)
                print(recall)
                print(thresholds)

                matrix = confusion_matrix(t_Y, predictions_transformed)
                print(matrix)
                file_name = '{}/TabNetClassifier_DL-Model_test_data_confusion_Matrix.txt'.format(path)
                write_matrix_and_report_to_file(file_name, np.array2string(matrix))

                report = classification_report(t_Y, predictions_transformed)
                print(report)
                file_name = '{}/TabNetClassifier_DL-Model_test_deta_report.txt'.format(path)
                write_matrix_and_report_to_file(file_name, report)

                f1_score_value = f1_score(t_Y, predictions_transformed, average=None)
                print(f1_score_value)

                test_acc = accuracy_score(y_pred=t_Y, y_true=predictions_transformed)
                print(test_acc)

                print(f"BEST VALID SCORE: {tab_net_model.best_cost}")
                
                display = PrecisionRecallDisplay.from_predictions(t_Y, predictions_transformed, name="DL-Model")
                _ = display.ax_.set_title("Precision-Recall curve")
                file_name = '{}/TabNetClassifier_DL-Model_test-data_Precision-Recall-curve.png'.format(path)
                plt.savefig(file_name)
                plt.close()

                # --- save tabnet model
                file_name = '{}/trained_tabnet_model'.format(path)
                saved_filepath = tab_net_model.save_model(file_name)
