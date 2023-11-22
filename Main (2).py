import numpy as np
import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

TEST_SIZE = 0.3
K = 3


def load_data(filename):
    """
    Load spam data from a CSV file `filename` and convert it into a list of
    feature vectors and a list of target labels. Return a tuple (features, labels).

    Feature vectors should be a list of lists, where each inner list contains the
    57 features.

    Labels should be the corresponding list of labels, where each label
    is 1 if spam, and 0 otherwise.
    """
    features = []
    labels = []

    with open(filename, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            # Extract features (first 57 numbers) and label (last number) from each row
            feature_vector = [float(value) for value in row[:-1]]
            label = int(row[-1])

            features.append(feature_vector)
            labels.append(label)

    return features, labels


def preprocess(features):
    """
    Normalize each feature by subtracting the mean value in each
    feature and dividing by the standard deviation.
    """
    features = np.array(features)
    means = np.mean(features, axis=0)
    stds = np.std(features, axis=0)

    # Avoid division by zero by setting small stds to a non-zero value
    stds[stds < 1e-8] = 1.0

    normalized_features = (features - means) / stds
    return normalized_features.tolist()


# def train_knn_model(filename):
#     # Load data from spreadsheet
#     features, labels = load_data(filename)

#     # Preprocess features
#     features = preprocess(features)

#     # Split the data into train and test sets
#     X_train, X_test, y_train, y_test = train_test_split(
#         features, labels, test_size=TEST_SIZE)

#     # Create and train k-NN classifier
#     knn = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
#     knn.fit(X_train, y_train)

#     return knn

class NN:
    def __init__(self, trainingFeatures, trainingLabels) -> None:
        self.trainingFeatures = trainingFeatures
        self.trainingLabels = trainingLabels

    def predict(self, features, k):
        """
        Given a list of features vectors of testing examples,
        return the predicted class labels (list of either 0s or 1s)
        using the k nearest neighbors.
        """
        predictions = []
        for feature_vector in features:
            distances = []
            for train_feature, train_label in zip(self.trainingFeatures, self.trainingLabels):
                distance = np.linalg.norm(np.array(train_feature) - np.array(feature_vector))
                distances.append((distance, train_label))

            distances.sort(key=lambda x: x[0])
            nearest_neighbors = distances[:k]
            labels = [neighbor[1] for neighbor in nearest_neighbors]
            predicted_label = max(set(labels), key=labels.count)
            predictions.append(predicted_label)

        return predictions
 
def train_mlp_model(features, labels):
    """
    Given a list of features lists and a list of labels, return a
    fitted MLP model trained on the data using scikit-learn.
    """
    mlp = MLPClassifier(hidden_layer_sizes=(10, 5), activation='logistic', max_iter=1000)
    mlp.fit(features, labels)
    return mlp

def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return (accuracy, precision, recall, f1).

    Assume each label is either a 1 (positive) or 0 (negative).
    """
    true_positives = sum(1 for true_label, predicted_label in zip(labels, predictions) if true_label == 1 and predicted_label == 1)
    false_positives = sum(1 for true_label, predicted_label in zip(labels, predictions) if true_label == 0 and predicted_label == 1)
    false_negatives = sum(1 for true_label, predicted_label in zip(labels, predictions) if true_label == 1 and predicted_label == 0)
    true_negatives = sum(1 for true_label, predicted_label in zip(labels, predictions) if true_label == 0 and predicted_label == 0)

    accuracy = (true_positives + true_negatives) / len(labels)

    if true_positives + false_positives == 0:
        precision = 0
    else:
        precision = true_positives / (true_positives + false_positives)

    if true_positives + false_negatives == 0:
        recall = 0
    else:
        recall = true_positives / (true_positives + false_negatives)

    if precision + recall == 0:
        f1 = 0
    else:
        f1 = (2 * precision * recall) / (precision + recall)


    # print("True Positives: ", true_positives)
    # print("True Negatives: ", true_negatives)
    # print("False Positives: ", false_positives)
    # print("False Negatives: ", false_negatives)

    return accuracy, precision, recall, f1

def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python template.py ./spambase.csv")

    # Load data from spreadsheet and split into train and test sets
    features, labels = load_data(sys.argv[1])
    features = preprocess(features)
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=TEST_SIZE)

    # Train a k-NN model and make predictions
    model_nn = NN(X_train, y_train)
    predictions_nn = model_nn.predict(X_test, K)
    accuracy_nn, precision_nn, recall_nn, f1_nn = evaluate(y_test, predictions_nn)

    # Print results for k-NN
    print("**** k-Nearest Neighbors Results ****")
    print("Accuracy: ", accuracy_nn)
    print("Precision: ", precision_nn)
    print("Recall: ", recall_nn)
    print("F1: ", f1_nn)
    
    # Train an MLP model and make predictions
    model_mlp = train_mlp_model(X_train, y_train)
    predictions_mlp = model_mlp.predict(X_test)
    accuracy_mlp, precision_mlp, recall_mlp, f1_mlp = evaluate(y_test, predictions_mlp)

    # Print results for MLP
    print("**** MLP Results ****")
    print("Accuracy: ", accuracy_mlp)
    print("Precision: ", precision_mlp)
    print("Recall: ", recall_mlp)
    print("F1: ", f1_mlp)


if __name__ == "__main__":
    main()
