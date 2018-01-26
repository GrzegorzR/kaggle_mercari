from collections import Counter
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def transform_to_dummies(test, train, cat, top_n):
    ######brands

    top_brands = Counter(train[cat]).most_common(top_n)
    top_brands = [i[0] for i in top_brands]
    train[cat] = [i if i in top_brands else top_n + 1 for i in list(train[cat])]
    train = pd.concat([pd.get_dummies(train[cat]), train], axis=1)

    test[cat] = [i if i in top_brands else top_n + 1 for i in list(test[cat])]
    test = pd.concat([pd.get_dummies(test[cat]), test], axis=1)

    return test, train



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
        print "{} {}/{}".format(column, i, l)
        result[cat] = df[df[column] == cat]['price'].mean()
    return result


def split_categories(df):
    all_categories = []
    for row in df["category_name"]:
        try:
            all_categories.append(row.split("/"))
        except Exception:
            all_categories.append(["None", "None", "None"])
    for i in range(3):
        df['cat{}'.format(str(i))] = [j[i] for j in all_categories]



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

    test.to_csv('data/test3.csv')
    train.to_csv('data/train3.csv')


def item_description_to_length(df):
    res =[]
    for description in df["item_description"]:
        res.append(len(str(description)))
    df["def_len"] = res
    return df

def main():
    test_data_path = "data/test.tsv"
    train_data_path = "data/train.tsv"

    categorical_columns = ['brand_name', 'cat0', 'cat1', 'cat2']
    # categorical_columns = ['cat0']

    test_df = pd.read_csv(test_data_path, sep='\t')
    train_df = pd.read_csv(train_data_path, sep='\t')

    test_df["brand_name"] = test_df["brand_name"].fillna("None")
    train_df["brand_name"] = train_df["brand_name"].fillna("None")

    split_categories(test_df)
    split_categories(train_df)

    for column in categorical_columns:
        add_mean_cat_column(test_df, train_df, column)

    test_df.to_csv('data/test2.csv')
    train_df.to_csv('data/train2.csv')


    train_df = pd.read_csv("data/train2.csv")
    test_df = pd.read_csv("data/test2.csv")
    categorical_to_numerical(train_df, test_df, ["brand_name", "cat0", "cat1", "cat2"])

    train_df = pd.read_csv("data/train3.csv")
    test_df = pd.read_csv("data/test3.csv")
    train_df = item_description_to_length(train_df)
    test_df = item_description_to_length(test_df)
    train_df.to_csv("data/train4.csv")
    test_df.to_csv("data/test4.csv")



def get_data():
    train_df = pd.read_csv("data/train4.csv")
    test_df = pd.read_csv("data/test4.csv")
    y_train = train_df["price"]
    id_test = test_df['test_id']

    """
    MAX_FEATURES_ITEM_DESCRIPTION= 20
    nrow_train = train_df.shape[0]
    
    all_descriptions = pd.concat([train_df["item_description"], test_df["item_description"]])
    tv = TfidfVectorizer(max_features=MAX_FEATURES_ITEM_DESCRIPTION,
                         ngram_range=(1, 3),
                         stop_words='english')
    all_descriptions = pd.DataFrame(all_descriptions)
    all_descriptions.fillna(value='missing', inplace=True)
    X_description = tv.fit_transform(all_descriptions["item_description"])
    
    test_df = pd.concat(test_df, X_description[:nrow_train])
    train_df = pd.concat(test_df, X_description[nrow_train:])
    """

    test_df, train_df = transform_to_dummies(test_df, train_df, "brand_name", 20)
    test_df, train_df = transform_to_dummies(test_df, train_df, "cat2", 20)
    test_df, train_df = transform_to_dummies(test_df, train_df, "cat1", 20)

    train_df.drop(['price', 'train_id', 'name',  'item_description', 'category_name'], axis=1,
                  inplace=True)

    test_df.drop(['test_id', 'name',  'item_description', 'category_name'], axis=1, inplace=True)
    train_df['aadd'] = train_df['brand_name_mean'] * train_df['cat2_mean']
    test_df['aadd'] = test_df['brand_name_mean'] * test_df['cat2_mean']
    x_train = train_df
    x_test = test_df

    return x_train, y_train, x_test, id_test







if __name__ == '__main__':
    #main()
    #train_df = pd.read_csv("data/train2.csv")
    #test_df = pd.read_csv("data/test2.csv")
    #categorical_to_numerical(train_df, test_df, ["brand_name", "cat0", "cat1", "cat2"])
    """
    train_df = pd.read_csv("data/train3.csv")
    test_df = pd.read_csv("data/test3.csv")
    train_df = item_description_to_length(train_df)
    test_df = item_description_to_length(test_df)
    train_df.to_csv("data/train4.csv")
    test_df.to_csv("data/test4.csv")
    """
    import missingno as msno

    #train_df = pd.read_table("data/train.tsv")
    test_df = pd.read_table("data/test.tsv")
    msno.matrix(test_df)