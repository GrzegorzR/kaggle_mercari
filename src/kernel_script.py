import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output



from sklearn.cross_validation import train_test_split
import xgboost as xgb


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


def get_data():
    train_df = pd.read_table("../input/train.tsv")
    test_df = pd.read_table("../input/test.tsv")

    # categorical_columns = ['brand_name', 'cat0', 'cat1', 'cat2']
    # categorical_columns = ['cat0']
    # categorical_columns = ['cat0', 'cat1', 'cat2']
    categorical_columns = []

    test_df["brand_name"] = test_df["brand_name"].fillna("None")
    train_df["brand_name"] = train_df["brand_name"].fillna("None")

    split_categories(test_df)
    split_categories(train_df)

    for column in categorical_columns:
        add_mean_cat_column(test_df, train_df, column)

    # categorical_to_numerical(train_df, test_df, ["brand_name", "cat0", "cat1", "cat2"])


    train_df = item_description_to_length(train_df)
    test_df = item_description_to_length(test_df)

    y_train = train_df["price"]
    id_test = test_df['test_id']
    train_df.drop(['price', 'train_id', 'name', 'brand_name', 'item_description', 'category_name',
                   'cat0', 'cat1', 'cat2'], axis=1,
                  inplace=True)

    test_df.drop(['test_id', 'name', 'brand_name', 'item_description', 'category_name',
                  'cat0', 'cat1', 'cat2'], axis=1, inplace=True)
    # train_df['aadd'] = train_df['brand_name_mean'] * train_df['cat2_mean']
    # test_df['aadd'] = test_df['brand_name_mean'] * test_df['cat2_mean']
    x_train = train_df
    x_test = test_df

    return x_train, y_train, x_test, id_test


def prepare_xgb_data(x_train, y_train, x_test):
    # Take a random 20% of the dataset as validation data
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.15, random_state=4242)
    print('Train samples: {} Validation samples: {}'.format(len(x_train), len(x_valid)))

    # Convert our data into XGBoost format
    d_train = xgb.DMatrix(x_train, y_train)
    d_valid = xgb.DMatrix(x_valid, y_valid)
    d_test = xgb.DMatrix(x_test)
    return d_train, d_valid, d_test


def get_xgb_model(params, d_train, d_valid):
    watchlist = [(d_train, 'train'), (d_valid, 'valid')]

    mdl = xgb.train(params, d_train, 10, watchlist, early_stopping_rounds=50, verbose_eval=10)

    return mdl


def main():
    params = {}
    params['objective'] = 'reg:linear'
    params['eta'] = 0.2
    params['silent'] = True
    params['max_depth'] = 6
    params['subsample'] = 0.7
    params['colsample_bytree'] = 0.7
    params['n_estimators'] = 1500
    params['min_child_weight'] = 5
    params['seed'] = 1337
    params['eval_metric'] = 'rmse'

    x_train, y_train, x_test, id_test = get_data()

    d_train, d_valid, d_test = prepare_xgb_data(x_train, y_train, x_test)

    model1 = get_xgb_model(params, d_train, d_valid)

    params = {}
    params['objective'] = 'reg:linear'
    params['eta'] = 0.3
    params['silent'] = True
    params['max_depth'] = 5
    params['subsample'] = 0.7
    params['colsample_bytree'] = 0.7
    params['n_estimators'] = 1500
    params['min_child_weight'] = 10
    params['seed'] = 13437
    params['eval_metric'] = 'rmse'

    model2 = get_xgb_model(params, d_train, d_valid)

    price1 = model1.predict(d_test)
    price1 = [i if i > 0 else 0 for i in price1]
    # price1 = pd.DataFrame()

    price2 = model2.predict(d_test)
    price2 = [i if i > 0 else 0 for i in price2]
    # price2 = pd.DataFrame()

    sub = pd.DataFrame()
    sub['test_id'] = id_test
    sub['price'] = (np.array(price1) * 0.5) + (np.array(price2) * 0.5)
    sub.to_csv('sub3.csv', index=False)


main()