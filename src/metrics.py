import numpy as np
import matplotlib.pyplot as plt


def confusion_matrix(y_true, y_pred):
    assert len(y_true) == len(y_pred)
    
    TP = np.sum((y_pred == 1) & (y_true == 1))
    TN = np.sum((y_pred == 0) & (y_true == 0))
    FP = np.sum((y_pred == 1) & (y_true == 0))
    FN = np.sum((y_pred == 0) & (y_true == 1))
    
    return np.array([[TN, FP],
                     [FN, TP]])
    
def accuracy(y_true, y_pred):
    return np.mean((y_pred > 0.5) == (y_true > 0.5))