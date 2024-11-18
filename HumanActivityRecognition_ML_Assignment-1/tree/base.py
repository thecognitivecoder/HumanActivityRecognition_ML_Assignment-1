"""
The current code given is for the Assignment 1.
You will be expected to use this to make trees for:
> discrete input, discrete output
> real input, real output
> real input, discrete output
> discrete input, real output
"""

from dataclasses import dataclass
from typing import Literal, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tree.utils import entropy, gini_index, mse, information_gain, opt_split_attribute, split_data, one_hot_encoding, check_ifreal

@dataclass
class DecisionTree:
    criterion: Literal["information_gain", "gini_index", "mse"]
    max_depth: int = 5

    def __init__(self, criterion: Literal["information_gain", "gini_index", "mse"] = "information_gain", max_depth: int = 5):
        self.criterion = criterion
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        if not check_ifreal(y):
            y = y.astype("category")
        self.tree = self._fit(X, y, depth=0)

    def _fit(self, X: pd.DataFrame, y: pd.Series, depth: int) -> dict:
        if len(y.unique()) == 1:
            return {'prediction': y.mode()[0]}

        if depth >= self.max_depth:
            return {'prediction': y.mode()[0]}

        if X.empty:
            return {'prediction': y.mode()[0]}

        feature, split = opt_split_attribute(X, y, self.criterion, X.columns)

        if feature is None:
            return {'prediction': y.mode()[0]}

        tree = {'feature': feature, 'split': split}
        left_X, left_y, right_X, right_y = split_data(X, y, feature, split)

        if not left_X.empty and not right_X.empty:
            tree['left'] = self._fit(left_X, left_y, depth + 1)
            tree['right'] = self._fit(right_X, right_y, depth + 1)
        else:
            tree['prediction'] = y.mode()[0]

        return tree

    def predict(self, X: pd.DataFrame) -> pd.Series:
        return X.apply(self._predict_row, axis=1)

    def _predict_row(self, row: pd.Series) -> pd.Series:
        node = self.tree
        while 'feature' in node:
            feature_value = row[node['feature']]
            if (check_ifreal(pd.Series([feature_value])) and feature_value <= node['split']) or (not check_ifreal(pd.Series([feature_value])) and feature_value == node['split']):
                node = node.get('left', {'prediction': None})
            else:
                node = node.get('right', {'prediction': None})

        return node.get('prediction', None)

    def plot(self) -> None:
        def plot_tree(node, depth=0):
            if 'prediction' in node:
                print(" " * depth * 4, "Predict:", node['prediction'])
            else:
                print(" " * depth * 4, f"Feature {node['feature']} <= {node['split']}")
                print(" " * (depth + 1) * 4, "Left:")
                plot_tree(node['left'], depth + 1)
                print(" " * (depth + 1) * 4, "Right:")
                plot_tree(node['right'], depth + 1)

        if self.tree is not None:
            plot_tree(self.tree)
        else:
            print("Tree is not trained yet.")
             """
        Function to plot the tree

        Output Example:
        ?(X1 > 4)
            Y: ?(X2 > 7)
                Y: Class A
                N: Class B
            N: Class C
        Where Y => Yes and N => No
        """
