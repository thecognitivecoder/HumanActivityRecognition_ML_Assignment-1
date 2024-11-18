"""
You can add your own functions here according to your decision tree implementation.
There is no restriction on following the below template, these fucntions are here to simply help you.
"""
import numpy as np
import pandas as pd


def one_hot_encoding(X: pd.DataFrame) -> pd.DataFrame:
    return pd.get_dummies(X)

def check_ifreal(y: pd.Series) -> bool:
    return y.dtype.name!='category'


def entropy(Y: pd.Series) -> float:
    probs= Y.value_counts(normalize=True)
    return-sum(probs*np.log2(probs))


def gini_index(Y: pd.Series) -> float:
    probs=Y.value_counts(normalize=True)
    return 1-sum(probs**2)


def mse(Y:pd.Series) -> float:
    if Y.dtype.name == 'category' or Y.dtype == object:
        # Calculate the gini index for categorical data to be used in MSE calculation
        return 1 - (Y.value_counts(normalize=True)**2).sum()
    else:
        return np.mean((Y - np.mean(Y))**2)


def information_gain(Y: pd.Series, attr: pd.Series, criterion: str) -> float:
    if criterion=="entropy":
        initial_criterion= entropy(Y)
    elif criterion=="gini_index":
        initial_criterion=gini_index(Y)
    else:
        initial_criterion=mse(Y)
    
    values,counts=np.unique(attr,return_counts=True)
    weighted_avg_criterion=0
    total_count=sum(counts)
    for i,v in enumerate(values):
        proportion=counts[i]/total_count
        if criterion=="entropy":
            criterion_value=entropy(Y[attr==v])
        elif criterion=="gini_index":
            criterion_value=gini_index(Y[attr==v])
        else:
            criterion_value=mse(Y[attr==v])
        weighted_avg_criterion += proportion*criterion_value
    


def opt_split_attribute(X: pd.DataFrame, y: pd.Series, criterion, features: pd.Series):
    best_feature=None
    best_gain=-np.inf
    best_split=None
    for feature in X.columns:
        gain=information_gain(y,X[feature],criterion)
        if gain>best_gain:
            best_gain=gain
            best_feature=feature
            best_split=X[feature].mean() if check_ifreal(X[feature]) else None
    return best_feature,best_split
    


def split_data(X: pd.DataFrame, y: pd.Series, attribute, value):
    if check_ifreal(X[attribute]):
        mask=X[attribute]<=value
    else:
        mask=X[attribute]==value
    left_X=X[mask]
    right_x=X[~mask]
    left_y=y[mask]
    right_y=y[~mask]

    return left_x,left_y,right_X,right_y

    # Split the data based on a particular value of a particular attribute. You may use masking as a tool to split the data.

    
