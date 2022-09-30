# gaussian mixture clustering
# weblink: https://machinelearningmastery.com/clustering-algorithms-with-python/
from numpy import unique
from numpy import where
from sklearn.datasets import make_classification
from sklearn.mixture import GaussianMixture
from matplotlib import pyplot

import pandas as pd

# read csv input file
input_data = pd.read_csv("All_Participents_Clusterd_WaveSum_DataFrame.csv", sep=";", decimal=',')

load_test_data = pd.read_csv("All_Participents_Condition-C_WaveSum_DataFrame.csv", sep=";", decimal=',')

print(input_data.head(1))
print(load_test_data.head(1))

test_data = load_test_data
lda_x_data = input_data.drop(columns=['Conscientious', ])
copy_x_data = lda_x_data

gaussianMixture = GaussianMixture(n_components=2).fit(lda_x_data)

print(gaussianMixture.score_samples(lda_x_data))
print(gaussianMixture.score(lda_x_data))
lda_x_data["Conscientious"] = gaussianMixture.predict(lda_x_data)
print(lda_x_data["Conscientious"]) 
lda_x_data["Conscientious"] = lda_x_data["Conscientious"].astype("int")



ax2 = lda_x_data.plot.scatter(x='Conscientious', y='pId', c='Conscientious', colormap='viridis')
# show the plot
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