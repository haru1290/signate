import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error


FOLD = 5
NUM_ROUND = 1000
VERBOSE_EVAL = 1


def make_submit_file(pred):
    submit = pd.read_csv('./data/submit_sample.csv', sep=',', header=None)
    submit.loc[:,1] = pred
    submit.to_csv("./submit.csv", index=None, header=None)


def get_category_id(category):
    le = LabelEncoder()
    category_id = le.fit_transform(category)

    return category_id


def lgb_model(params, X, y):
    valid_scores = []
    models = []
    kf = KFold(n_splits=FOLD, shuffle=True, random_state=123)

    for fold, (train_indices, valid_indices) in enumerate(kf.split(X)):
        X_train, X_valid = X[train_indices], X[valid_indices]
        y_train, y_valid = y[train_indices], y[valid_indices]
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_eval = lgb.Dataset(X_valid, y_valid)

        model = lgb.train(
            params,
            lgb_train,
            valid_sets=lgb_eval,
            num_boost_round=NUM_ROUND,
            verbose_eval=VERBOSE_EVAL
        )

        y_valid_pred = model.predict(X_valid)
        score = mean_squared_error(y_valid, y_valid_pred, squared=False)
        print(f"fold: {fold} | RMSE: {score:.3f}")
        valid_scores.append(score)

        models.append(model)

    return models


def main(): 
    train = pd.read_csv('./data/train.csv')
    test = pd.read_csv('./data/test.csv')

    categorical_features = ['Country', 'City']
    for category in categorical_features:
        train[category] = get_category_id(train[category].values)
        test[category] = get_category_id(test[category].values)

    X = train.drop(['id', 'pm25_mid'], axis=1).values
    y = train['pm25_mid'].values
    X_test = test.drop(['id'], axis=1).values
    
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
    )

    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'verbose': -1
    }

    models = lgb_model(params, X_train, y_train)

    # pred = model.predict(test_)
    # make_submit_file(pred)


if __name__ == '__main__':
    main()