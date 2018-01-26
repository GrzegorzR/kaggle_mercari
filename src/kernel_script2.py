# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

import lightgbm as lgb
from xgboost.sklearn import XGBRegressor
import math
import xgboost as xgb
from collections import Counter

from sklearn.cross_validation import train_test_split


def split_categories(df):
    all_categories = []
    for row in df["category_name"]:
        try:
            all_categories.append(row.split("/"))
        except Exception:
            all_categories.append(["None", "None", "None"])
    for i in range(3):
        df['cat{}'.format(str(i))] = [j[i] for j in all_categories]


def add_mean_cat_column(test, train, column):
    cat_mean_dic = get_category_mean_dict(train, column)
    col_test, col_train = test[column], train[column]
    res_test, res_train = [], []
    for val in col_test:
        if val in cat_mean_dic.keys():
            res_test.append(cat_mean_dic[val])
        else:
            res_test.append(cat_mean_dic["None"])
    for val in col_train:
        if val in cat_mean_dic.keys():
            res_train.append(cat_mean_dic[val])
    test['{}_mean'.format(column)] = res_test
    train['{}_mean'.format(column)] = res_train


def get_category_mean_dict(df, column):
    result = {}
    categories = list(set(df[column]))
    l = len(categories)
    for i, cat in enumerate(categories):
        print ("{} {}/{}".format(column, i, l))
        result[cat] = df[df[column] == cat]['price'].mean()
    return result


def item_description_to_length(df):
    res = []
    for description in df["item_description"]:
        res.append(len(str(description)))
    df["def_len"] = res
    return df


def categorical_to_numerical(train, test, categorical):
    for column_name in categorical:
        categories = list(set(train[column_name]))
        categories_dic = dict(zip(categories, range(len(categories))))
        test_num_cat, train_num_cat = [], []
        for cat in train[column_name]:
            train_num_cat.append(categories_dic[cat])
        for cat in test[column_name]:
            try:
                test_num_cat.append(categories_dic[cat])
            except Exception:
                test_num_cat.append(categories_dic["None"])
        test[column_name] = test_num_cat
        train[column_name] = train_num_cat


def transform_to_dummies(test, train, cat, top_n):
    ######brands

    top_brands = Counter(train[cat]).most_common(top_n)
    top_brands = [i[0] for i in top_brands]
    train[cat] = [i if i in top_brands else top_n + 1 for i in list(train[cat])]
    train = pd.concat([pd.get_dummies(train[cat]), train], axis=1)

    test[cat] = [i if i in top_brands else top_n + 1 for i in list(test[cat])]
    test = pd.concat([pd.get_dummies(test[cat]), test], axis=1)
    return test, train


def get_data():
    train_df = pd.read_table("../input/train.tsv")
    test_df = pd.read_table("../input/test.tsv")

    #categorical_columns = ['brand_name', 'cat0', 'cat1', 'cat2']
    # categorical_columns = ['cat0']
    # categorical_columns = ['cat0', 'cat1', 'cat2']
    categorical_columns = []

    test_df["brand_name"] = test_df["brand_name"].fillna("None")
    train_df["brand_name"] = train_df["brand_name"].fillna("None")

    split_categories(test_df)
    split_categories(train_df)

    for column in categorical_columns:
        add_mean_cat_column(test_df, train_df, column)

    categorical_to_numerical(train_df, test_df, ["brand_name", "cat0", "cat1", "cat2"])

    train_df = item_description_to_length(train_df)
    test_df = item_description_to_length(test_df)

    y_train = train_df["price"]
    id_test = test_df['test_id']
    print (len(test_df))
    print (len(test_df))
    train_df.drop(['price', 'train_id', 'name', 'item_description', 'category_name'
                   ], axis=1,
                  inplace=True)

    test_df.drop(['test_id', 'name', 'item_description', 'category_name'
                  ], axis=1, inplace=True)
    #train_df['aadd'] = train_df['brand_name_mean'] * train_df['cat2_mean']
    #test_df['aadd'] = test_df['brand_name_mean'] * test_df['cat2_mean']
    test_df, train_df = transform_to_dummies(test_df, train_df, "brand_name", 20)
    test_df, train_df = transform_to_dummies(test_df, train_df, "cat2", 20)
    test_df, train_df = transform_to_dummies(test_df, train_df, "cat1", 20)
    x_train = train_df
    x_test = test_df
    print (len(x_test))
    print (len(id_test))

    return x_train, y_train, x_test, id_test


def train_lgbm(x_train, y_train, x_valid, y_valid):
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

    return model


def train_xgb(x_train, y_train, x_valid, y_valid):
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
    return model


def main():
    x_train, y_train, x_test, id_test = get_data()
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=4242)
    x_test_xgb = xgb.DMatrix(x_test)


    model1 = train_lgbm(np.array(x_train.values), np.array(y_train),
                        np.array(x_valid.values), np.array(y_valid))

    model2 = train_xgb(np.array(x_train.values), np.array(y_train.values),
                       np.array(x_valid.values), np.array(y_valid.values))

    x_test_xgb = xgb.DMatrix(np.array(x_test))
    r1 = model1.predict(x_test)
    r2 = model2.predict(x_test_xgb)

    price1 = np.array(r1) * 0.6 + np.array(r2) * 0.4

    price1 = [i if i > 0 else 0 for i in price1]
    print(len(price1))
    print(len(id_test))
    sub = pd.DataFrame()
    sub['test_id'] = id_test
    # sub['price'] = (np.array(price1) * 0.5) + (np.array(price2) * 0.5)
    sub['price'] = price1
    sub.to_csv('sub3.csv', index=False)


main()