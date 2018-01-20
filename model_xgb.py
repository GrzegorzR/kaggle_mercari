import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
import xgboost as xgb


def get_xgb_model(params, d_train, d_valid):

    watchlist = [(d_train, 'train'), (d_valid, 'valid')]

    mdl = xgb.train(params, d_train, 10000, watchlist, early_stopping_rounds=100, verbose_eval=10)

    return mdl