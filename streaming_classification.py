# python 3.7
# Scikit-learn ver. 0.23.2
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import sklearn.metrics as metrics
from sklearn.metrics import plot_confusion_matrix, confusion_matrix
import numpy
# matplotlib 3.3.1
from matplotlib import pyplot

data = pd.read_csv('/Users/kosi/documents/ahcompsci/SKLearn/tv_shows.csv')
print(data.head())
# Set variables for the targets and features
Y = data.IMDb
data_features = ['Year', 'Age', 'RottenTomatoes', 'Netflix', 'Hulu', 'Prime Video', 'Disney+']
X = data[data_features]
# print(X.describe())
# print(X.head())
X = pd.DataFrame(X).to_numpy()
Y = pd.DataFrame(Y).to_numpy()

# Split the data into training and validation sets
train_X, val_X, train_Y, val_Y = train_test_split(X, Y, random_state=7)

# Create the classifier
data_model = RandomForestRegressor()
# Fit model
data_model.fit(train_X, train_Y)

# Predict classes given the validation features
pred_Y = data_model.predict(val_X)
pred_Y = list(pred_Y)
for i, y in enumerate(pred_Y):
    pred_Y[i] = [y]

# Calculate the accuracy as our performance metric
accuracy = metrics.pairwise.paired_distances(val_Y, pred_Y)
accuracyAvg = sum(accuracy) / len(accuracy)
percErr = round(accuracyAvg * 10, 2)
val_Y = val_Y.tolist()

indiv_result = list(zip(val_Y, pred_Y, accuracy))

for j, entry in enumerate(indiv_result):
    l = []
    for i, val in enumerate(entry):
        if isinstance(val, list):
            l.append(round(val[0], 2))
        else:
            l.append(round(val, 2))
    indiv_result[j] = l
print("\n                Rating            Prediction           Diff. (+/-)")
fp = open('/Users/kosi/documents/ahcompsci/SKLearn/tv_shows.csv', 'r')
p_list = []
val_list = []
for row in indiv_result[:100]:
    # print(named_list)
    # return(named_list)
    print("{: >20} {: >20} {: >20}".format(* row))
print("Percent of Error: %f " % percErr)
print("Accuracy: %f" % (100-percErr))

confusion_matrix(y_test_classes, y_pred_classes)
# plot_confusion_matrix(data_model, pred_Y, val_Y)
# pyplot.show()