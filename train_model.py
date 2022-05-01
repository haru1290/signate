import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error


class ModelXGBoost:
    def __init__(self):
        self.model = None

    def fit(self, X_train, X_valid, y_train, y_valid):
        xgb_train = xgb.DMatrix(X_train, label=y_train)
        xgb_valid = xgb.DMatrix(X_valid, label=y_valid)

        params = {
            'objective': 'reg:squarederror',
            'random_state': 0,
        }

        self.model = xgb.train(
            params,
            xgb_train,
            evals=[(xgb_train, 'train'), (xgb_valid, 'valid')],
            num_boost_round=100000,
            early_stopping_rounds=100,
            verbose_eval=100
        )

    def predict(self, X):
        pred = self.model.predict(xgb.DMatrix(X))

        return pred


class ModelLightGBM:
    def __init__(self):
        self.model = None

    def fit(self, X_train, X_valid, y_train, y_valid):
        lgb_train = lgb.Dataset(X_train, label=y_train)
        lgb_valid = lgb.Dataset(X_valid, label=y_valid, reference=lgb_train)

        params = {
            'objective': 'regression',
            'random_state': 0,
            'metric': 'rmse',
        }

        self.model = lgb.train(
            params,
            lgb_train,
            valid_sets=lgb_valid,
            num_boost_round=100000,
            early_stopping_rounds=100,
            verbose_eval=100
        )

    def predict(self, X):
        pred = self.model.predict(X, num_iteration=self.model.best_iteration)

        return pred


class ModelCatBoost:
    def __init__(self):
        self.model = None

    def fit(self, X_train, X_valid, y_train, y_valid):
        params = {
            'iterations': 100000,
            'use_best_model': True,
            'random_seed': 0,
            'l2_leaf_reg': 3,
            'depth': 6,
            'loss_function': 'RMSE',
            'task_type': 'GPU',
        }

        model_ = CatBoostRegressor(**params)
        self.model = model_.fit(X_train, y_train, eval_set=(X_valid, y_valid), early_stopping_rounds=100, verbose=100)

    def predict(self, X):
        pred = self.model.predict(X)

        return pred


class ModelLinearSVR:
    def __init__(self):
        self.model = None

    def fit(self, X_train, X_valid, y_train, y_valid):
        self.scaler = StandardScaler()
        self.scaler.fit(X_train)
        X_train = self.scaler.transform(X_train)
        self.model = LinearSVR(
            max_iter=1000,
            C=1.0,
            random_state=0,
            epsilon=5.0,
            verbose=100,
        )
        self.model.fit(X_train, y_train)

    def predict(self, X):
        X = self.scaler.transform(X)
        pred = self.model.predict(X)

        return pred


class ModelRandomForest:
    def __init__(self):
        self.model = None

    def fit(self, X_train, X_valid, y_train, y_valid):
        self.scaler = StandardScaler()
        self.scaler.fit(X_train)
        X_train = self.scaler.transform(X_train)
        self.model = RandomForestRegressor(
            max_depth=5,
            n_estimators=100,
            random_state=0,
            verbose=1
        )
        self.model.fit(X_train, y_train)

    def predict(self, X):
        X = self.scaler.transform(X)
        pred = self.model.predict(X)

        return pred


class Model2Linear:
    def __init__(self):
        self.model = None
        self.scaler = None

    def fit(self, X_train, X_valid, y_train, y_valid):
        self.scaler = StandardScaler()
        self.scaler.fit(X_train)
        X_train = self.scaler.transform(X_train)
        self.model = LinearRegression()
        self.model.fit(X_train, y_train)

    def predict(self, X):
        X = self.scaler.transform(X)
        pred = self.model.predict(X)

        return pred


def preprocessing(train, test, categorical_features):
    for category in categorical_features:
        le = LabelEncoder().fit(list(
            set(train[category].unique()).union(
            set(test[category].unique()))
        ))
        train[category] = le.transform(train[category])
        test[category] = le.transform(test[category])

    return train, test


def predict_cv(model, X_train, y_train, X_test):
    preds = []
    preds_test = []
    valid_indices = []
    
    kf = KFold(n_splits=6, shuffle=True, random_state=0)
    for fold, (train_index, valid_index) in enumerate(kf.split(X_train)):
        X_tr, X_va = X_train.iloc[train_index], X_train.iloc[valid_index]
        y_tr, y_va = y_train.iloc[train_index], y_train.iloc[valid_index]

        model.fit(X_tr, X_va, y_tr, y_va)
        pred = model.predict(X_va)
        preds.append(pred)
        pred_test = model.predict(X_test)
        preds_test.append(pred_test)
        valid_indices.append(valid_index)
    
    valid_indices = np.concatenate(valid_indices)
    preds = np.concatenate(preds, axis=0)
    order = np.argsort(valid_indices)
    pred_train = preds[order]
  
    preds_test = np.mean(preds_test, axis=0)

    return pred_train, preds_test


def main():
    train = pd.read_csv('./data/train.csv')
    test = pd.read_csv('./data/test.csv')

    train, test = preprocessing(train, test, ['Country', 'City'])

    X_train_valid = train.drop(['id', 'pm25_mid'], axis=1)
    y_train_valid = train['pm25_mid']
    X_test = test.drop(['id'], axis=1)

    model_1a = ModelXGBoost()
    pred_train_1a, pred_test_1a = predict_cv(model_1a, X_train_valid, y_train_valid, X_test)

    model_1b = ModelLightGBM()
    pred_train_1b, pred_test_1b = predict_cv(model_1b, X_train_valid, y_train_valid, X_test)

    model_1c = ModelCatBoost()
    pred_train_1c, pred_test_1c = predict_cv(model_1c, X_train_valid, y_train_valid, X_test)

    model_1d = ModelLinearSVR()
    pred_train_1d, pred_test_1d = predict_cv(model_1d, X_train_valid, y_train_valid, X_test)

    model_1e = ModelRandomForest()
    pred_train_1e, pred_test_1e = predict_cv(model_1e, X_train_valid, y_train_valid, X_test)

    print(f'a XGBoost RMSE: {mean_squared_error(y_train_valid, pred_train_1a, squared=False):.4f}')    
    print(f'b LightGBM RMSE: {mean_squared_error(y_train_valid, pred_train_1b, squared=False):.4f}')
    print(f'c CatBoost RMSE: {mean_squared_error(y_train_valid, pred_train_1c, squared=False):.4f}')
    print(f'd LinearSVR RMSE: {mean_squared_error(y_train_valid, pred_train_1d, squared=False):.4f}')
    print(f'e RandomForest RMSE: {mean_squared_error(y_train_valid, pred_train_1e, squared=False):.4f}')

    X_train_2 = pd.DataFrame({
        'pred_1a': pred_train_1a,
        'pred_1b': pred_train_1b,
        'pred_1c': pred_train_1c,
        'pred_1d': pred_train_1d,
        'pred_1e': pred_train_1e
    })
    X_test_2 = pd.DataFrame({
        'pred_1a': pred_test_1a,
        'pred_1b': pred_test_1b,
        'pred_1c': pred_test_1c,
        'pred_1d': pred_test_1d,
        'pred_1e': pred_test_1e
    })
    X_train_2 = pd.concat([X_train_valid, X_train_2], axis=1)
    X_test_2 = pd.concat([X_test, X_test_2], axis=1)

    model_2b = Model2Linear()
    pred_train_2, pred_test_2 = predict_cv(model_2b, X_train_2, y_train_valid, X_test_2)

    print(f'RMSE: {mean_squared_error(y_train_valid, pred_train_2, squared=False):.4f}')

    submission = pd.DataFrame({'id': test['id'], 'pm25_mid': pred_test_2})
    submission.to_csv('./submit.csv', header=False, index=False)


if __name__ == '__main__':
    main()