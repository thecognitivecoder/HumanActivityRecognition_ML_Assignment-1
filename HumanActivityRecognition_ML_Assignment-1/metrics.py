from typing import Union
import pandas as pd
import numpy as np


def accuracy(y_hat: pd.Series, y: pd.Series) -> float:
    assert y_hat.size == y.size
    return (y_hat==y).sum()/len(y)


def precision(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    true_positive=((y_hat==cls)&(y==clas)).sum()
    predicted_postive=(y_hat==cls).sum()
    return true_positive/predicted_positive if predicted_positive!=0 else 0.0
    

def recall(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    true_positive=((y_hat==cls)&(y==cls)).sum()
    actual_positive=(y==cls).sum()
    return true_positive/actual_positive if actual_postive !=0 else 0.0


def rmse(y_hat: pd.Series, y: pd.Series) -> float:
    return np.sqrt(((y_hat-y)**2).mean())


def mae(y_hat: pd.Series, y: pd.Series) -> float:
    return np.abs(y_hat-y).mean()
