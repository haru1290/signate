import numpy as np
import pandas as pd
import optuna
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error


def tmp():
    train = pd.read_csv('./data/train.csv')
    test = pd.read_csv('./data/test.csv')

    train, test = preprocessing(train, test, ['Country', 'City'])

    print(train.isnull().sum())
    print(test.isnull().sum())



class Objective:
    def __init__(self, X_train, X_valid, y_train, y_valid):
        self.X_train = X_train
        self.X_valid = X_valid
        self.y_train = y_train
        self.y_valid = y_valid

    def __call__(self, trial):
        lgb_train = lgb.Dataset(self.X_train, self.y_train)
        lgb_valid = lgb.Dataset(self.X_valid, self.y_valid)

        params = {
            'task': 'train',
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'metric': 'rmse',
            'learning_rate': trial.suggest_uniform('learning_rate', 0.01, 0.1),
            'num_leaves': trial.suggest_int("num_leaves", 10, 300),
            'tree_learner': trial.suggest_categorical('tree_learner', ["serial", "feature", "data", "voting"]),
            'seed': 0,
            'verbose': 0
        }

        model = lgb.train(
            params=params,
            train_set=lgb_train,
            valid_sets=[lgb_train, lgb_valid],
            valid_names=['train', 'valid'],
            num_boost_round=100000,
            early_stopping_rounds=100,
            verbose_eval=100
        )

        y_valid_pred = model.predict(self.X_valid, num_iteration=model.best_iteration)
        rmse = mean_squared_error(self.y_valid, y_valid_pred, squared=False)

        return rmse


def preprocessing(train, test, categorical_features):
    for category in categorical_features:
        le = LabelEncoder().fit(list(
            set(train[category].unique()).union(
            set(test[category].unique()))
        ))
        train[category] = le.transform(train[category])
        test[category] = le.transform(test[category])

    # train['tmp'] = (train['co_cnt'] + train['co_min'] + train['co_mid'] + train['no2_mid'] + train['co_max'] + train['co_var'])
    # test['tmp'] = (test['co_cnt'] + test['co_min'] + test['co_mid'] + test['no2_mid'] + test['co_max'] + test['co_var'])

    train['tmp1'] = (train['co_mid'] * train['o3_mid'] * train['so2_mid'] * train['no2_mid'] * train['temperature_mid'] * train['humidity_mid'] * train['pressure_mid'] * train['ws_mid'] * train['dew_mid'])
    test['tmp1'] = (test['co_mid'] * test['o3_mid'] * test['so2_mid'] * test['no2_mid'] * test['temperature_mid'] * test['humidity_mid'] * test['pressure_mid'] * test['ws_mid'] * test['dew_mid'])

    # train['tmp2'] = (train['co_mid'] + train['o3_mid'] + train['so2_mid'] + train['no2_mid'] + train['temperature_mid'] + train['humidity_mid'] + train['pressure_mid'] + train['ws_mid'] + train['dew_mid'])
    # test['tmp2'] = (test['co_mid'] + test['o3_mid'] + test['so2_mid'] + test['no2_mid'] + test['temperature_mid'] + test['humidity_mid'] + test['pressure_mid'] + test['ws_mid'] + test['dew_mid'])

    # train['tmp3'] = (train['co_mid'] - train['o3_mid'] - train['so2_mid'] - train['no2_mid'] - train['temperature_mid'] - train['humidity_mid'] - train['pressure_mid'] - train['ws_mid'] - train['dew_mid'])
    # test['tmp3'] = (test['co_mid'] - test['o3_mid'] - test['so2_mid'] - test['no2_mid'] - test['temperature_mid'] - test['humidity_mid'] - test['pressure_mid'] - test['ws_mid'] - test['dew_mid'])

    # train['tmp4'] = (train['co_mid'] / train['o3_mid'] / train['so2_mid'] / train['no2_mid'] / train['temperature_mid'] / train['humidity_mid'] / train['pressure_mid'] / train['ws_mid'] / train['dew_mid'])
    # test['tmp4'] = (test['co_mid'] / test['o3_mid'] / test['so2_mid'] / test['no2_mid'] / test['temperature_mid'] / test['humidity_mid'] / test['pressure_mid'] / test['ws_mid'] / test['dew_mid'])

    return train, test


def train_valid_step(X, y):
    cv_valid_scores = []
    models = []
    kf = KFold(n_splits=8, shuffle=True, random_state=0)

    for fold, (train_indices, valid_indices) in enumerate(kf.split(X)):
        X_train, X_valid = X[train_indices], X[valid_indices]
        y_train, y_valid = y[train_indices], y[valid_indices]
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_valid = lgb.Dataset(X_valid, y_valid)

        objective = Objective(X_train, X_valid, y_train, y_valid)
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, timeout=100)

        add_params = {
            'task': 'train',
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'metric': 'rmse',
            'seed': 0,
            'verbose': 0
        }

        add_params.update(study.best_params)

        model = lgb.train(
            params=add_params,
            train_set=lgb_train,
            valid_sets=[lgb_train, lgb_valid],
            valid_names=['train', 'valid'],
            num_boost_round=100000,
            early_stopping_rounds=100,
            verbose_eval=100
        )

        y_valid_pred = model.predict(X_valid, num_iteration=model.best_iteration)
        score = mean_squared_error(y_valid, y_valid_pred, squared=False)
        print(f"fold: {fold} | RMSE: {score:.3f}")
        cv_valid_scores.append(score)

        models.append(model)
  
    print(f'CV : {np.mean(cv_valid_scores)}')

    return models


def test_step(models, X_test):
    y_test_preds = pd.DataFrame()
    
    for fold, model in enumerate(models):
        y_test_pred = model.predict(X_test, num_iteration=model.best_iteration)
        y_test_preds[fold] = y_test_pred

    return y_test_preds.mean(axis=1)


def main():
    train = pd.read_csv('./data/train.csv')
    test = pd.read_csv('./data/test.csv')

    train, test = preprocessing(train, test, ['Country', 'City'])

    X_train = train.drop(['id', 'pm25_mid'], axis=1).values
    y_train = train['pm25_mid'].values
    X_test = test.drop(['id'], axis=1).values

    models = train_valid_step(X_train, y_train)
    y_test_preds = test_step(models, X_test)

    submission = pd.DataFrame({'id': test['id'], 'pm25_mid': y_test_preds})
    submission.to_csv('./submit.csv', header=False, index=False)

    importance = pd.DataFrame(models[0].feature_importance(), index=train.drop(['id', 'pm25_mid'], axis=1).columns, columns=['importance'])
    print(importance.head(60))


if __name__ == '__main__':
    # tmp()
    main()