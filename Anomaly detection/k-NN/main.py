from time import time
import numpy as np
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

""" Runs the K-Nearest Neighbors Classifier """
def run(data, k_nearest, save_model=False, test_data=None):
    print(">>>>> K-NEAREST NEIGHBORS (k=" + str(k_nearest) +") <<<<<")
    start = time()
    np.random.shuffle(data)

    classes = data[:, 57] # Classification for each email in the dataset
    features = data[:, :54] # Features for each email in the dataset

    clf = KNeighborsClassifier(n_neighbors=k_nearest, p=2, metric="euclidean")
    clf.fit(features, classes) # Fit the classifier to the training data
    results = cross_val_score(clf, features, classes, cv=10, n_jobs=-1)
    print("--------------------------")
    print("Accuracy (minimum): %.3f%%" % (results.min() * 100))
    print("Accuracy (maximum): %.3f%%" % (results.max() * 100))
    print("Accuracy (mean): %.3f%%" % (results.mean() * 100))
    print("Variance: " + str(results.var()))

    if save_model:
        with open('k-nearest-neighbors-model.pkl', 'wb') as f:
            pickle.dump(clf, f)
        print("Model saved as 'k-nearest-neighbors-model.pkl'")
    
    if test_data is not None:
        features_test = test_data[:, :54]
        classes_test = test_data[:, 57]
        score = clf.score(features_test, classes_test)
        print("Test data accuracy: %.3f%%" % (score * 100))

    end = time()
    print("\nTime elapsed: {}".format(end - start))
    print("--------------------------\n")


def main ():
    # Load dataset
    filename = "data/spambase/spambase.data"
    file = open(filename, "r")
    dataset = np.loadtxt(file, delimiter = ",")
    k_nearest = 5
    save_model = True
    
    # Test data
    filename_test = "data/spambase/test_data.data"
    file_test = open(filename_test, "r")
    dataset_test = np.loadtxt(file_test, delimiter = ",")

    run(dataset, k_nearest, save_model, dataset_test)

if __name__ == '__main__':
    main()
