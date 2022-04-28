from tabnanny import verbose
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import make_scorer, mean_squared_error


def output_file(pred_):
    submit = pd.read_csv('./submit_sample.csv', sep=',', header=None)
    submit.loc[:,1] = pred_
    submit.to_csv("./submit.csv", index=None, header=None)


def train_model(X_train, y_train, params, scorer):
    xgb_reg = xgb.XGBRegressor()
    grid_xgb_reg = GridSearchCV(
        xgb_reg,
        param_grid=params,
        scoring=scorer,
        cv=5,
        n_jobs=-1,
        verbose=3
    )
    grid_xgb_reg.fit(X_train, y_train)

    return grid_xgb_reg


def main(): 
    train = pd.read_csv('./data/train.csv')
    test = pd.read_csv('./data/test.csv')

    train_ = train.drop(['id','Country', 'City', 'pm25_mid'], axis=1)
    test_ = test.drop(['id','Country', 'City'], axis=1)
    y = train['pm25_mid']
    
    X_train, X_test, y_train, y_test = train_test_split(
            train_, y,
            test_size=.2,
    )

    params=[{
        'max_depth':[3, 4, 5, 6],
        'n_estimators': [100, 300, 500]
    }]

    scorer = make_scorer(mean_squared_error, greater_is_better=False)

    model = train_model(X_train, y_train, params, scorer)
    pred = model.predict(X_test)
    print(f"RMSE: {mean_squared_error(pred, y_test, squared=False):.3f}")
    # pred = model.predict(test_)
    # output_file(pred)


if __name__ == '__main__':
    main()