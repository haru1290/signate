import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from argparse import ArgumentParser


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


def main(args):
    train = pd.read_csv(args.X_train_valid)
    test = pd.read_csv(args.y_train_valid)

    X_train_valid = train.drop(['id', 'pm25_mid'], axis=1)
    y_train_valid = train['pm25_mid']
    X_test = test.drop(['id'], axis=1)

    model_1b = ModelLightGBM()
    pred_train_1b, pred_test_1b = predict_cv(model_1b, X_train_valid, y_train_valid, X_test)

    print(f'RMSE: {mean_squared_error(y_train_valid, pred_train_1b, squared=False):.4f}')

    submission = pd.DataFrame({'id': X_test['id'], 'pm25_mid': pred_test_1b})
    submission.to_csv(args.submit, header=False, index=False)


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--train', default='./data/train_new.csv')
    parser.add_argument('--test', default='./data/test_new.csv')
    parser.add_argument('--submit', default='.submit.csv')

    args = parser.parse_args()

    main(args)
