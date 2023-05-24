from collections import Counter
import numpy as np


class NaiveBayes:
    def __init__(self):
        self.class_probabilities = {}
        self.feature_probabilities = {}
        self.classes = []
    
    def fit(self, X_train, y_train):
        self.classes = np.unique(y_train)
        self.class_probabilities = self.calculate_class_probabilities(y_train)
        self.feature_probabilities = self.calculate_feature_probabilities(X_train, y_train)
    
    def predict(self, X_test):
        predictions = []
        for instance in X_test:
            probabilities = []
            for class_label in self.classes:
                 # set probability to the probability of y
                probability = self.class_probabilities[class_label]
                 # each key represents a feature and its corresponding value.
                for feature, value in instance.items():
                    if (class_label, feature, value) in self.feature_probabilities:
                        # seen during training
                        probability *= self.feature_probabilities[(class_label, feature, value)]
                    else:
                        probability *= 0  # Laplace smoothing for unseen feature values
                probabilities.append(probability)
            predicted_class = self.classes[np.argmax(probabilities)]
            predictions.append(predicted_class)
        return predictions
    
    def calculate_class_probabilities(self, y_train):
        class_counts = {}
        for class_label in y_train:
            if class_label in class_counts:
                class_counts[class_label] += 1
            else:
                class_counts[class_label] = 1
        total_instances = len(y_train)
        class_probabilities = {class_label: count / total_instances for class_label, count in class_counts.items()}
        return class_probabilities
    
    def calculate_feature_probabilities(self, X_train, y_train):
        feature_counts = {}
        for instance, class_label in zip(X_train, y_train):
            for feature, value in instance.items():
                if (class_label, feature, value) in feature_counts:
                    feature_counts[(class_label, feature, value)] += 1
                else:
                    feature_counts[(class_label, feature, value)] = 1
        feature_probabilities = {}
        class_counts = Counter(y_train)
        for (class_label, feature, value), count in feature_counts.items():
            feature_probabilities[(class_label, feature, value)] = count / class_counts[class_label]
        return feature_probabilities
