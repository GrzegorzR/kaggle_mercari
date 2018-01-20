
from collections import Counter
import pandas as pd



train_ = pd.DataFrame()


def transform_to_dummies(test, train, cat, top_n):
    ######brands
    top_brands = Counter(train_df["brand_name"]).most_common(top_n)
    top_brands = [i[0] for i in top_brands]
    train_["brand_name"] = [i if i in top_brands else "other" for i in list(test_df["brand_name"])]
    # train_ = pd.concat([pd.get_dummies(train_["brand_name"]), train_], axis=1)

    all_categories = [i.split("/") for i in train_df["category_name"] if type(i) != float]

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


def main():
    test_data_path = "data/test.tsv"
    train_data_path = "data/train.tsv"


    categorical_columns = ['brand_name', 'cat0', 'cat1', 'cat2']
    #categorical_columns = ['cat0']

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

if __name__ == '__main__':
    main()