import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt
import pandas as pd

input_data = pd.read_csv("All_Participents_Clusterd_WaveSum_DataFrame.csv", sep=";", decimal=',')

test_data = pd.read_csv("All_Participents_Condition-C_WaveSum_DataFrame.csv", sep=";", decimal=',')

test_x = test_data.iloc[:, :].values

#print(input_data.subtract(test_data))

lda_x_data = input_data.drop(columns=['Conscientious'])

print(lda_x_data.head(1))
print(test_data.head(1))

for col in test_data.columns:
    found = False
    for col1 in input_data.columns:
        if col == col1:
            found = True
            break
    if not found:
        print(col)

#------ Normalizing
# Separating out the features
x = lda_x_data.iloc[:, :].values
#print(x)

#test_value = lda_x_data.iloc[[0], :].values
#test_value = lda_x_data.iloc[[7000], :].values
#print(test_value)

# Separating out the target
y = np.array(input_data[["Conscientious"]].values.flatten()) #input_data[["Conscientious"]].values.to_numpy() #input_data.iloc[:,154].values#y = input_data.iloc[:,['Conscientious']].values

lda = LinearDiscriminantAnalysis()
lda.fit(x, y)

#result_array = lda.predict(x)
result_array = lda.predict(test_x)
print(result_array)

print(lda.score(test_x, result_array))
print(lda.predict_proba(test_x))
print(lda.predict_log_proba(test_x))
#lda_x_data['Conscientious'] = pd.Series(result_array)
#ax2 = lda_x_data.plot.scatter(x='Conscientious', y='pId', c='Conscientious', colormap='viridis')

test_data['Conscientious'] = pd.Series(result_array)
ax2 = test_data.plot.scatter(x='Conscientious', y='pId', c='Conscientious', colormap='viridis')

# show the plot
plt.show()

#print(lda.predict(test_value))

#print(lda.predict([[-0.8, -1]]))