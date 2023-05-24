from math import log2
from collections import Counter
import numpy as np

class Node:
    def __init__(self, attribute=None, value=None, class_label=None):
        self.attribute = attribute
        self.value = value
        self.class_label = class_label
        self.children = {}


class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.root = None

    def fit(self, X, y):
        attributes = list(X[0].keys())
        self.root = self._build_tree(X, y, attributes)

    def _build_tree(self, X, y, attributes, depth=0):
        # Check for stopping conditions
        if depth == self.max_depth or len(set(y)) == 1:
            return Node(class_label=self._most_common_label(y))

        if not attributes:
            return Node(class_label=self._most_common_label(y))

        best_attribute = self._choose_best_attribute(X, y, attributes)
        node = Node(attribute=best_attribute)
        # Value Iteration:
        for value in set([x[best_attribute] for x in X]):
            filtered_X = [x for x in X if x[best_attribute] == value]
            filtered_y = [y[i] for i in range(len(X)) if X[i][best_attribute] == value]
              # Child Node Creation:
            if not filtered_X or not filtered_y:
                node.children[value] = Node(class_label=self._most_common_label(y))
            else:
                new_attributes = attributes.copy()
                new_attributes.remove(best_attribute)
                node.children[value] = self._build_tree(filtered_X, filtered_y, new_attributes, depth+1)

        return node

    def predict(self, X):
        return [self._traverse_tree(x, self.root) for x in X]

    def _traverse_tree(self, x, node):
        if node.class_label is not None:
            return node.class_label

        attribute_value = x[node.attribute]
        if attribute_value in node.children:
            child_node = node.children[attribute_value]
        else:
            child_node = list(node.children.values())[0]

        return self._traverse_tree(x, child_node)

    def _choose_best_attribute(self, X, y, attributes):
        max_gain = -float('inf')
        best_attribute = None

        for attribute in attributes:
            gain = self._information_gain(X, y, attribute)
            if gain > max_gain:
                max_gain = gain
                best_attribute = attribute

        return best_attribute

    def _information_gain(self, X, y, attribute):
        entropy = self._entropy(y)
        attribute_values = set([x[attribute] for x in X])
        weighted_entropy = 0

        for value in attribute_values:
            filtered_y = [y[i] for i in range(len(X)) if X[i][attribute] == value]
            weighted_entropy += (len(filtered_y) / len(y)) * self._entropy(filtered_y)

        information_gain = entropy - weighted_entropy
        return information_gain

    def _entropy(self, y):
        counter = Counter(y)
        probabilities = [count / len(y) for count in counter.values()]
        entropy = sum([-p * log2(p) for p in probabilities])
        return entropy


    def _most_common_label(self, y):
        counter = Counter(y)
        most_common_label = counter.most_common(1)[0][0]
        return most_common_label

