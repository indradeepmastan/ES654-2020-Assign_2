"""
The current code given is for the Assignment 2.
> Classification
> Regression
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from metrics import *

from models.bagging import BaggingClassifier
from models.bagging import BaggingRegressor

########### RandomForestClassifier ###################

N = 30
P = 5
X = pd.DataFrame(np.random.randn(N, P))
y = pd.Series(np.random.randint(P, size = N), dtype="category")

for criteria in ['information_gain', 'gini_index']:
    Classifier_BG = BaggingClassifier(base_estimator='decision tree')
    Classifier_BG.fit(X, y)
    y_hat = Classifier_BG.predict(X)
    Classifier_BG.plot()
    print('Criteria :', criteria)
    print('Accuracy: ', accuracy(y_hat, y))
    for cls in y.unique():
        print('Precision: ', precision(y_hat, y, cls))
        print('Recall: ', recall(y_hat, y, cls))

########### RandomForestRegressor ###################

N = 30
P = 5
X = pd.DataFrame(np.random.randn(N, P))
y = pd.Series(np.random.randn(N))

for criteria in ['information_gain', 'gini_index']:
    Regressor_BG = BaggingRegressor(base_estimator='decision tree')
    Regressor_BG.fit(X, y)
    y_hat = Regressor_BG.predict(X)
    Regressor_BG.plot()
    print('Criteria :', criteria)
    print('RMSE: ', rmse(y_hat, y))
    print('MAE: ', mae(y_hat, y))