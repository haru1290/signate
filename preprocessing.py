import numpy as np
import pandas as pd
import category_encoders as ce
from sklearn.neighbors import NearestNeighbors
from argparse import ArgumentParser


def preprocessing(train, test, categorical_features):
    for category in categorical_features:
        count_encoder = ce.CountEncoder(cols=[category])
        train[category] = count_encoder.fit_transform(train[category])
        test[category] = count_encoder.fit_transform(test[category])

    train_coord = np.array([train['lat'], train['lon']]).T
    test_coord = np.array([train['lat'], train['lon']]).T
    nbrs = NearestNeighbors(n_neighbors=3, algorithm='kd_tree', metric='euclidean').fit(train_coord)
    nbrs_indices = nbrs.kneighbors(test_coord, return_distance=False)
    print(nbrs_indices)

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
    parser.add_argument('--submit', default='./submit.csv')

    args = parser.parse_args()

    main(args)