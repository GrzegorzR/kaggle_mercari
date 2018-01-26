from collections import Counter
import pandas as pd

from model_xgb import prepare_xgb_data, get_xgb_model
from prepare_data import get_data


def main():
    params = {}
    params['objective'] = 'reg:linear'
    params['eta'] = 0.3
    params['silent'] = True
    params['max_depth'] = 12
    params['subsample'] = 0.7
    params['colsample_bytree'] = 0.7
    params['n_estimators'] = 1500
    params['min_child_weight'] = 5
    params['seed'] = 1337
    params['eval_metric'] = 'rmse'

    x_train, y_train, x_test, id_test = get_data()

    d_train, d_valid, d_test = prepare_xgb_data(x_train, y_train, x_test)

    model = get_xgb_model(params, d_train, d_valid)

    asd = model.predict(d_test)

if __name__ == '__main__':
    main()