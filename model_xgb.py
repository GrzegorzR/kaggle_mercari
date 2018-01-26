import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
import xgboost as xgb

from prepare_data import get_data

import math
#from ml_metrics import rmsle


# A function to calculate Root Mean Squared Logarithmic Error (RMSLE)
def rmsle(y, y_pred):
    assert len(y) == len(y_pred)
    terms_to_sum = [(math.log(y_pred[i] + 1) - math.log(y[i] + 1)) ** 2.0 for i, pred in enumerate(y_pred)]
    return (sum(terms_to_sum) * (1.0 / len(y))) ** 0.5

def rmsle_xgb(preds, dtrain):
    labels = dtrain.get_label()
    gini_score = rmsle(labels, preds)
    return [('gini', gini_score)]

def prepare_xgb_data(x_train, y_train, x_test):
    # Take a random 20% of the dataset as validation data
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=4242)
    # Take a random 20% of the dataset as validation data
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=4242)
    print('Train samples: {} Validation samples: {}'.format(len(x_train), len(x_valid)))

    # Convert our data into XGBoost format
    d_train = xgb.DMatrix(x_train, y_train)
    d_valid = xgb.DMatrix(x_valid, y_valid)
    d_test = xgb.DMatrix(x_test)
    return d_train, d_valid, d_test


def get_xgb_model(params, d_train, d_valid):
    watchlist = [(d_train, 'train'), (d_valid, 'valid')]

    mdl = xgb.train(params, d_train, 10000, watchlist, early_stopping_rounds=100, verbose_eval=10)

    return mdl
