import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn import datasets
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score


def make_submit_file(pred):
    submit = pd.read_csv('./data/submit_sample.csv', sep=',', header=None)
    submit.loc[:,1] = pred
    submit.to_csv("./submit.csv", index=None, header=None)


def train_model(X_train, X_valid, y_train, y_valid):
    lgb_train = lgb.Dataset(X_train, y_train, free_raw_data=False)
    lgb_valid = lgb.Dataset(X_valid, y_valid, free_raw_data=False)

    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': 'rmse',
        'learning_rate': 0.1
    }

    evaluation_results = {}
    model = lgb.train(
        params,
        lgb_train,
        num_boost_round=2000,
        valid_names=['train', 'valid'],
        valid_sets=[lgb_train, lgb_valid],
        evals_result=evaluation_results,
        early_stopping_rounds=10
    )

    return model


def main(): 
    train = pd.read_csv('./data/train.csv')
    test = pd.read_csv('./data/test.csv')

    train_ = train.drop(['id', 'pm25_mid'], axis=1)
    test_ = test.drop(['id'], axis=1)
    y = train['pm25_mid']
    
    X_train, X_valid, y_train, y_valid = train_test_split(
            train_, y,
            test_size=0.2,
            random_state=123
    )

    K_fold = KFold(n_splits=5, shuffle=True,  random_state=42)
    for train_cv_no, eval_cv_no in K_fold.split(list(range(len(y_train))), y_train):
        pass
    
    model = train_model(X_train, X_valid, y_train, y_valid)
    pred = model.predict(test_)
    make_submit_file(pred)


if __name__ == '__main__':
    main()