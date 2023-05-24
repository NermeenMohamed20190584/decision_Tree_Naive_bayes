import csv
import random
from math import log2
from collections import Counter
import numpy as np
from DecisionTree import DecisionTree
from Naive import NaiveBayes


def read_csv(file_path):
    with open(file_path, 'r') as file:
        data = [row for row in csv.reader(file, delimiter=';')]
    return data


# Preprocess data
def preprocess_data(data, columns):
    processed_data = []
    for row in data:
        processed_row = {}
        for i, value in enumerate(row):
            if i in columns:
                processed_row[i] = value.strip('"')
        processed_data.append(processed_row)
    return processed_data


# Split data into training and testing sets
def split_data(data, train_ratio):
    random.shuffle(data)
    train_size = int(len(data) * train_ratio)
    train_set = data[:train_size]
    test_set = data[train_size:]
    return train_set, test_set


# Convert labels to binary values
def convert_labels(labels):
    return [1 if label == 'yes' else 0 for label in labels]


# Calculate accuracy of predictions
def calculate_accuracy(predictions, labels):
    correct = sum([1 if p == l else 0 for p, l in zip(predictions, labels)])
    accuracy = correct / len(predictions) * 100
    return accuracy


# Set seed for reproducibility
random.seed(42)

# Define input and output columns
input_columns = [1, 2, 3, 6]  # Index of columns:job, marital, education, housing
output_columns = [16]  # Index of column: y

# Read data from CSV file
file_path = "Bank_dataset.csv"
data = read_csv(file_path)

# Extract header from data
header = data[0]
data = data[1:]

# Interface to select the percentage of data to read
percentage_of_records= float(input("Enter the percentage of records which you want to read (0-100):"))
num_records = int(len(data) * (percentage_of_records / 100))

# Preprocess data
data = preprocess_data(data[:num_records], input_columns + output_columns)

# Split data into training and testing sets
train_set, test_set = split_data(data, train_ratio=0.7)

# Extract input features and labels from the training set
X_train = np.array([{column: row[column] for column in input_columns} for row in train_set])
y_train = np.array(convert_labels([row[output_columns[0]] for row in train_set]))

# Extract input features and labels from the testing set
X_test = [{column: row[column] for column in input_columns} for row in test_set]
y_test = convert_labels([row[output_columns[0]] for row in test_set])

# Train the decision tree classifier
dt_classifier = DecisionTree(max_depth=5)
dt_classifier.fit(X_train, y_train)

# Make predictions using the decision tree classifier
dt_predictions = dt_classifier.predict(X_test)

# Calculate accuracy of the decision tree classifier
dt_accuracy = calculate_accuracy(dt_predictions, y_test)
print("Decision Tree classification accuracy: {:.2f}%".format(dt_accuracy))

# Train the Naive Bayes classifier
nb_classifier = NaiveBayes()
nb_classifier.fit(X_train, y_train)

# Make predictions using the Naive Bayes classifier
nb_predictions = nb_classifier.predict(X_test)

# Calculate accuracy of the Naive Bayes classifier
nb_accuracy = calculate_accuracy(nb_predictions, y_test)
print("Naive Bayes classification accuracy: {:.2f}%".format(nb_accuracy))

if dt_accuracy > nb_accuracy:
    print("Decision Tree classification is better than Naive Bayes classification for this data")
elif dt_accuracy < nb_accuracy:
    print("Naive Bayes classification is better than Decision Tree classification for this data")
else:
    print("Both classifications have the same accuracy")
