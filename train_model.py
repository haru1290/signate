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


def train_model(X_train_cv, X_valid_cv, y_train_cv, y_valid_cv):
    lgb_train = lgb.Dataset(X_train_cv, y_train_cv, free_raw_data=False)
    lgb_valid = lgb.Dataset(X_valid_cv, y_valid_cv, reference=lgb_train, free_raw_data=False)

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
    y_ = train['pm25_mid']
    
    X_train, X_test, y_train, y_test = train_test_split(
        train_, y_,
        test_size=0.2,
        random_state=123
    )

    # 5分割交差検証
    models = []
    round_num = 0
    K_fold = KFold(n_splits=5, shuffle=True,  random_state=123)
    for train_cv_idx, valid_cv_idx in K_fold.split(X_train, y_train):
        X_train_cv = X_train.iloc[train_cv_idx]
        y_train_cv = y_train.iloc[train_cv_idx]
        X_valid_cv = X_train.iloc[valid_cv_idx]
        y_valid_cv = y_train.iloc[valid_cv_idx]

        model = train_model(X_train_cv, X_valid_cv, y_train_cv, y_valid_cv)
        models.appned(model)

        y_train_pred = model.predict(X_train_cv, num_iteration=model.best_iteration)
        y_valid_pred = model.predict(X_valid_cv, num_iteration=model.best_iteration)
        y_test_pred = model.predict(X_test, num_iteration=model.best_iteration)

        train_RMSE_score = mean_squared_error(y_train_cv, y_train_pred, squared=False)
        valid_RMSE_score = mean_squared_error(y_valid_cv, y_valid_pred, squared=False)
        test_RMSE_score = mean_squared_error(y_test, y_test_pred, squared=False)

        print(f"train_RMSE:{train_RMSE_score:.3f} | valid_RMSE:{valid_RMSE_score:.3f} | test_RMSE:{test_RMSE_score:.3f}")

        round_num += 1
    
    # model = train_model(X_train, X_valid, y_train, y_valid)
    # pred = model.predict(test_)
    # make_submit_file(pred)


if __name__ == '__main__':
    main()