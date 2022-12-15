import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
import sklearn.metrics as metrics
from sklearn import preprocessing                           # solution to error because  values for response variable are continuous.
from sklearn import utils
# matplotlib 3.3.1
from matplotlib import pyplot


def irrelevant(data):
    # drop irrelevant data
    data.drop(["L"], axis = 1, inplace = True)
    data.drop(["ID"], axis = 1, inplace = True)
    data.drop(["Title"], axis = 1, inplace = True)
    data.drop(["Year"], axis = 1, inplace = True)
    data.drop(["Age"], axis = 1, inplace = True)
    data.drop(["Netflix"], axis = 1, inplace = True)
    data.drop(["Hulu"], axis = 1, inplace = True)
    data.drop(["Prime Video"], axis = 1, inplace = True)
    data.drop(["Disney+"], axis = 1, inplace = True)
    data.drop(["Type"], axis = 1, inplace = True)
    print(data.head())


def distrib(data):
    # show IMDb x RottenTomatoes relationship distribution
    pyplot.scatter(data.RottenTomatoes, data.IMDb, marker = "+", color = "red")
    pyplot.show()

def prep(data):
    # Prep Data
    dataY = data["IMDb"].values
    dataX = data.drop(labels = ["IMDb"], axis = 1)
    print(utils.multiclass.type_of_target(dataX))
    print(utils.multiclass.type_of_target(dataY))
    lab = preprocessing.LabelEncoder()
    transformedY = lab.fit_transform(dataY)
    print(utils.multiclass.type_of_target(transformedY))
    # print(dataX.head())
    return dataX, transformedY

def regression(data):
    dataX, transformedY = prep(data)
    # Split Data
    trainX, testX, trainY, testY = train_test_split(
        dataX, transformedY, test_size = 0.3, shuffle = True
        )

    # Define Model
    model = LogisticRegression(max_iter = 10000)
    model.fit(trainX, trainY)

    # Test Model
    prediction_test = model.predict(testY.reshape(-1,1))
    correct = 0
    incorrect = 0
    for pred, gt in zip(prediction_test, testY):
        if pred == gt: correct += 1
        else: incorrect += 1
    print(f"\nCorrect: {correct}, Incorrect: {incorrect}, % Correct: {correct/(correct + incorrect): 5.2}")

    plot_confusion_matrix(model, testX, testY)
    pyplot.show()



def main():
    # Original Data
    data = pd.read_csv('/Users/kosi/documents/ahcompsci/SKLearn/tv_shows.csv')
    print(data.head())
    irrelevant(data)
    distrib(data)
    regression(data)

    """
    # Test Top 25% of Rotten Tomatoes
    data = pd.read_csv('/Users/kosi/documents/ahcompsci/SKLearn/top25.csv')
    irrelevant(data)
    regression(data)
    """
main()