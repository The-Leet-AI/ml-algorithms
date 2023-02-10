import numpy as np
import pickle
from sklearn.neighbors import KNeighborsClassifier

def test_model(test_data):
    # Load the model
    with open("k-nearest-neighbors-model.pkl", "rb") as f:
        clf = pickle.load(f)

    features_test = test_data[:, :54]
    classes_test = test_data[:, 57]
    score = clf.score(features_test, classes_test)
    print("Accuracy on test data: %.3f%%" % (score * 100))

if __name__ == '__main__':
    # Load test data
    # I just copied some of the training data as a test data
    filename = "data/spambase/test_data.data"
    file = open(filename, "r")
    dataset_test = np.loadtxt(file, delimiter=",")

    test_model(dataset_test)
