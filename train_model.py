import numpy as np
import pandas as pd
import optuna
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error
from argparse import ArgumentParser


class Objective:
    def __init__(self, X_train, X_valid, y_train, y_valid):
        self.X_train = X_train
        self.X_valid = X_valid
        self.y_train = y_train
        self.y_valid = y_valid

    def __call__(self, trial):
        lgb_train = lgb.Dataset(self.X_train, label=self.y_train)
        lgb_valid = lgb.Dataset(self.X_valid, label=self.y_valid, reference=lgb_train)

        params = {
            'task': 'train',
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'random_state': 0,
            'metric': {'rmse'},
            'learning_rate': trial.suggest_uniform('learning_rate', 0.01, 0.05),
            'num_leaves': trial.suggest_int("num_leaves", 5, 50),
            'tree_learner': trial.suggest_categorical('tree_learner', ["serial", "feature", "data", "voting"]),
            'seed': 0
        }

        self.model = lgb.train(
            params,
            lgb_train,
            valid_sets=lgb_valid,
            valid_names=['Valid'],
            num_boost_round=100000,
            early_stopping_rounds=100,
            verbose_eval=100
        )

        y_valid_pred = self.model.predict(self.X_valid)

        return mean_squared_error(self.y_valid, y_valid_pred, squared=False)


class ModelLightGBM:
    def __init__(self):
        self.model = None

    def fit(self, X_train, X_valid, y_train, y_valid):
        lgb_train = lgb.Dataset(X_train, label=y_train)
        lgb_valid = lgb.Dataset(X_valid, label=y_valid, reference=lgb_train)

        # objective = Objective(X_train, X_valid, y_train, y_valid,)
        # study = optuna.create_study(direction='minimize')
        # study.optimize(objective, timeout=60)

        add_params = {
            'task': 'train',
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'random_state': 0,
            'metric': {'rmse'},
            'seed': 0
        }

        # add_params.update(study.best_params)

        self.model = lgb.train(
            add_params,
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
    
    gkf = GroupKFold(n_splits=5)
    for fold, (train_index, valid_index) in enumerate(gkf.split(X_train, y_train, X_train['City'])):
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
    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)

    X_train_valid = train.drop(['id', 'day', 'humidity_var', 'dew_var', 'pm25_mid'], axis=1)
    y_train_valid = train['pm25_mid']
    X_test = test.drop(['id', 'day', 'humidity_var', 'dew_var'], axis=1)

    model_1b = ModelLightGBM()
    pred_train_1b, pred_test_1b = predict_cv(model_1b, X_train_valid, y_train_valid, X_test)

    print(f'CV RMSE: {mean_squared_error(y_train_valid, pred_train_1b, squared=False):.4f}')

    submission = pd.DataFrame({'id': test['id'], 'pm25_mid': pred_test_1b})
    submission.to_csv(args.submit, header=False, index=False)

    importance = pd.DataFrame(model_1b.model.feature_importance(importance_type='gain'), index=X_train_valid.columns, columns=['importance'])
    print(importance.head(60))


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--train', default='./data/train_new.csv')
    parser.add_argument('--test', default='./data/test_new.csv')
    parser.add_argument('--submit', default='./submit.csv')

    args = parser.parse_args()

    main(args)
