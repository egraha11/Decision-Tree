import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder


def import_data():

    data = [["Yes", "Yes", 7, "No"], ["Yes", "No", 12, "No"], ["No", "Yes", 18, "Yes"], ["No", "Yes", 35, "Yes"],
            ["Yes", "Yes", 38, "Yes"], ["Yes", "No", 50, "No"], ["No", "No", 83, "No"], ["Yes", "Yes", 10, "Yes"], ["No", "Yes", 16, "No"], ["Yes", "No", 22, "No"]]

    df = pd.DataFrame(
        data, columns=["Loves Popcorn", "Loves Soda", "Age", "Loves \'Cool as Ice\'"])

    return df


def splitdata(data):

    cols = ["Loves Popcorn", "Loves Soda", "Loves \'Cool as Ice\'"]


    #for y in data.columns:

        #if data[y].dtype != 'int64':
            #data[y] = LabelEncoder().fit_transform(data[y])


    data[cols] = data[cols].apply(LabelEncoder().fit_transform)

    X = data.values[:, :-1]
    Y = data.values[:, -1]

    x_train, x_test, y_train, y_test = train_test_split(
        X, Y, random_state=10, test_size=.3)

    return X, Y, x_train, x_test, y_train, y_test


def gini_index(data):

    gini = DecisionTreeClassifier(criterion="gini")

    gini.fit(data[2], data[4])

    return gini


def predictions(model, x_test):

    predictions = model.predict(x_test)
    df = pd.DataFrame(x_test)
    df[0] = df[0].replace([1, 0], ["Yes", "No"])
    df[1] = df[1].replace([1, 0], ["Yes", "No"])
    df[3] = predictions
    df[3] = df[3].replace([1, 0], ["Yes", "No"])
    df.columns = columns = ["Loves Popcorn",
                            "Loves Soda", "Age", "Loves \'Cool as Ice\'"]

    print("Predictions : \n" + str(df))
    return predictions


def accuracy(preds, test):

    print("Accuracy: ", round(accuracy_score(test, preds)*100, 2), "%")
    print("Confusion Matrix ", confusion_matrix(test, preds))


def main():

    df = import_data()
    data = splitdata(df)
    model = gini_index(data)
    preds = predictions(model, data[3])
    accuracy(preds, data[5])


if __name__ == "__main__":
    main()