from sklearn import svm
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Ridge, BayesianRidge

import numpy as np

import lightgbm as lgb
from xgboost.sklearn import XGBRegressor
from prepare_data import get_data
import math
import xgboost as xgb



def rmsle(y, y_pred):
    y_pred = [i if i> 0 else 0 for i in y_pred]
    assert len(y) == len(y_pred)
    terms_to_sum = [(math.log(y_pred[i] + 1) - math.log(y[i] + 1)) ** 2.0 for i, pred in enumerate(y_pred)]
    return (sum(terms_to_sum) * (1.0 / len(y))) ** 0.5


def test_model(clf, x_train, y_train, x_valid, y_valid):
    clf.fit(x_train, y_train)
    result = clf.predict(x_valid)
    print ("{} result: {}".format(str(clf), rmsle( y_valid, result)))


def test_lgbm(x_train, y_train, x_valid, y_valid):
    params = {
        'boosting_type': 'gbdt',
        'objective': 'regression_l1',
        'max_depth': 4,
        'num_leaves': 100,
        'metric': 'rmsle',
        'learning_rate': 0.15,
        'verbose': 0}

    params2 = {
        'learning_rate': 0.75,
        'application': 'regression',
        'max_depth': 3,
        'num_leaves': 100,
        'verbosity': -1,
        'metric': 'RMSE',
    }


    n_estimators = 1000
    d_valid = lgb.Dataset(x_valid, label=y_valid)
    watchlist = [d_valid]


    d_train = lgb.Dataset(x_train, label=y_train)
    model = lgb.train(params, d_train, n_estimators, watchlist, verbose_eval=1)

    result = model.predict(x_valid)
    print ("{} result: {}".format("lgbm", rmsle(y_valid, result)))
    return result

def test_xgb(x_train, y_train, x_valid, y_valid):
    params = {}
    params['objective'] = 'reg:linear'
    params['eta'] = 0.1
    params['silent'] = True
    params['max_depth'] = 12
    params['subsample'] = 0.7
    params['colsample_bytree'] = 0.7
    params['n_estimators'] = 1500
    params['min_child_weight'] = 5
    params['seed'] = 1337
    params['eval_metric'] = 'rmse'

    d_train = xgb.DMatrix(x_train, y_train)
    d_valid = xgb.DMatrix(x_valid, y_valid)


    watchlist = [(d_train, 'train'), (d_valid, 'valid')]
    d_test = xgb.DMatrix(x_valid)

    model = xgb.train(params, d_train, 10000, watchlist, early_stopping_rounds=50, verbose_eval=10)


    result = model.predict(d_test)
    print ("{} result: {}".format("lgbm", rmsle(y_valid, result)))
    return result


def main():
    x_train, y_train, x_test, id_test = get_data()
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=4242)
    print('Train samples: {} Validation samples: {}'.format(len(x_train), len(x_valid)))

    models = [
        BayesianRidge()
        ,
              RandomForestRegressor(verbose=True, n_estimators=20),
              Ridge(solver="sag", fit_intercept=True)
              ]
    for model in models:
        test_model(model, np.array(x_train), np.array(y_train),
                   np.array(x_valid), np.array(y_valid))


    r1 = test_lgbm(np.array(x_train.values), np.array(y_train),
              np.array(x_valid.values), np.array(y_valid))

    r2 = test_xgb(np.array(x_train.values), np.array(y_train.values),
             np.array(x_valid.values), np.array(y_valid.values))

    print "mean: {}".format(rmsle(np.array(y_valid), np.array(r1)*0.6 + np.array(r2)*0.4))


if __name__ == '__main__':


    main()