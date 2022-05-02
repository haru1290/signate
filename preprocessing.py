import numpy as np
import pandas as pd
from argparse import ArgumentParser
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestNeighbors


def knn(train, df):
    nn = NearestNeighbors(metric='cosine')
    nn.fit(np.array([train['lat'], train['lon']]).T)
    knn_ids = nn.kneighbors(np.array([df['lat'], df['lon']]).T, n_neighbors=3, return_distance=False)
    result = []
    for knn_id in knn_ids:
        result.append((train['pm25_mid'][knn_id[1]] + train['pm25_mid'][knn_id[2]])/2)
    df['pm25_mean'] = result

    return df


def preprocessing(train, test, categorical_features):
    for category in categorical_features:
        le = LabelEncoder().fit(list(
            set(train[category].unique()).union(
            set(test[category].unique()))
        ))
        train[category] = le.transform(train[category])
        test[category] = le.transform(test[category])

    train = knn(train, train)
    test = knn(train, test)

    return train, test


def main(args):
    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)

    categorical_features = ['Country', 'City']
    train, test = preprocessing(train, test, categorical_features)

    train.to_csv('./data/train_new.csv', index=False)
    test.to_csv('./data/test_new.csv', index=False)


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--train', default='./data/train.csv')
    parser.add_argument('--test', default='./data/test.csv')
    parser.add_argument('--submit', default='.submit.csv')

    args = parser.parse_args()

    main(args)