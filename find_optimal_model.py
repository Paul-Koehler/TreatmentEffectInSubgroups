from typing import List

import numpy as np
from lightgbm import *
from sklearn.linear_model import *
from sklearn.ensemble import *
from sklearn.naive_bayes import *
from sklearn.neural_network import *
from sklearn.model_selection import GridSearchCV
from sklearn.tree import *
import pandas as pd
from main import load_data

regression_learners = [
    LinearRegression(),
    Ridge(),
    Lasso(),
    ElasticNet(),
    RandomForestRegressor(),
    GradientBoostingRegressor(),
    MLPRegressor(),
    DecisionTreeRegressor(),
    ExtraTreesRegressor(),
    AdaBoostRegressor(),
    LGBMRegressor(),
    HistGradientBoostingRegressor(),
]

classification_learners = [
    LogisticRegression(),
    RandomForestClassifier(),
    GradientBoostingClassifier(),
    MLPClassifier(),
    BernoulliNB(),
    MultinomialNB(),
    DecisionTreeClassifier(),
    ExtraTreesClassifier(),
    AdaBoostClassifier(),
    LGBMClassifier(),
    HistGradientBoostingClassifier(),
]

arguments = {
    "LinearRegression": {
        "fit_intercept": [True, False],
        "positive": [True, False],
        "copy_X": [True, False],
        "n_jobs": [-1],
    },
    "Ridge": {
        "alpha": [0.1, 1.0, 10.0],
        "fit_intercept": [True, False],
        "positive": [True, False],
        "copy_X": [True, False],
        "max_iter": [None, 1000],
        "tol": [0.001, 0.0001],
        "solver": ["auto", "sag", "saga"],
    },
    "Lasso": {
        "alpha": [0.1, 1.0, 10.0],
        "fit_intercept": [True, False],
        "positive": [True, False],
        "copy_X": [True, False],
        "max_iter": [None, 1000],
        "tol": [0.001, 0.0001],
        "selection": ["cyclic", "random"],
    },
    "ElasticNet": {
        "alpha": [0.1, 1.0, 10.0],
        "l1_ratio": [0.1, 0.5, 0.9],
        "fit_intercept": [True, False],
        "positive": [True, False],
        "copy_X": [True, False],
        "max_iter": [None, 1000],
        "tol": [0.001, 0.0001],
        "selection": ["cyclic", "random"],
    },
    "LogisticRegression": {
        "penalty": ["l1", "l2"],
        "C": [0.1, 1.0, 10.0],
        "fit_intercept": [True, False],
        "max_iter": [100, 200],
        "solver": ["liblinear", "saga"],
    },
    "RandomForestClassifier": {
        "n_estimators": [100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
        "max_features": ["auto", "sqrt"],
    },
    "RandomForestRegressor": {
        "n_estimators": [100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
        "max_features": ["auto", "sqrt"],
    },
    "GradientBoostingClassifier": {
        "n_estimators": [100, 200],
        "learning_rate": [0.01, 0.1],
        "max_depth": [3, 5],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
    },
    "GradientBoostingRegressor": {
        "n_estimators": [100, 200],
        "learning_rate": [0.01, 0.1],
        "max_depth": [3, 5],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
    },
    "MLPClassifier": {
        "hidden_layer_sizes": [(100,), (50, 50)],
        "activation": ["relu", "tanh"],
        "solver": ["adam", "sgd"],
        "alpha": [0.0001, 0.001],
        "learning_rate": ["constant", "adaptive"],
    },
    "MLPRegressor": {
        "hidden_layer_sizes": [(100,), (50, 50)],
        "activation": ["relu", "tanh"],
        "solver": ["adam", "sgd"],
        "alpha": [0.0001, 0.001],
        "learning_rate": ["constant", "adaptive"],
    },
    "BernoulliNB": {
        "alpha": [1.0, 2.0],
        "fit_prior": [True, False],
    },
    "MultinomialNB": {
        "alpha": [1.0, 2.0],
        "fit_prior": [True, False],
    },
    "DecisionTreeClassifier": {
        "criterion": ["gini", "entropy"],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
    },
    "DecisionTreeRegressor": {
        "criterion": ["squared_error", "friedman_mse"],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
    },
    "ExtraTreesClassifier": {
        "n_estimators": [100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
        "max_features": ["auto", "sqrt"],
    },
    "ExtraTreesRegressor": {
        "n_estimators": [100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
        "max_features": ["auto", "sqrt"],
    },
    "AdaBoostClassifier": {
        "n_estimators": [50, 100],
        "learning_rate": [1.0, 0.1],
    },
    "AdaBoostRegressor": {
        "n_estimators": [50, 100],
        "learning_rate": [1.0, 0.1],
    },
    "LGBMClassifier": {
        "n_estimators": [100, 200],
        "learning_rate": [0.01, 0.1],
        "max_depth": [-1, 10, 20],
        "num_leaves": [31, 63],
    },
    "LGBMRegressor": {
        "n_estimators": [100, 200],
        "learning_rate": [0.01, 0.1],
        "max_depth": [-1, 10, 20],
        "num_leaves": [31, 63],
    },
    "HistGradientBoostingClassifier": {
        "max_iter": [100, 200],
        "learning_rate": [0.01, 0.1],
        "max_depth": [3, 5],
        "min_samples_leaf": [1, 2],
        "l2_regularization": [0.0, 0.1],

    },
    "HistGradientBoostingRegressor": {
        "max_iter": [100, 200],
        "learning_rate": [0.01, 0.1],
        "max_depth": [3, 5],
        "min_samples_leaf": [1, 2],
        "l2_regularization": [0.0, 0.1],
    }
}

score_func = "neg_mean_squared_error"


def find_models_and_params(models, args: dict[str, dict], scoring: str, x, y):
    grid_searchs = []
    for model in models:
        param = args[model.__class__.__name__]
        grid_search = GridSearchCV(model, param, cv=8, n_jobs=-1, scoring=scoring)
        grid_search.fit(x, y)
        print(f"Best params for {model.__class__.__name__}: {grid_search.best_params_}")
        print(f"Best score for {model.__class__.__name__}: {grid_search.best_score_}")
        grid_searchs.append(grid_search)
    grid_searchs.sort(key=lambda obj: obj.best_score_, reverse=True)
    print(f"Best model: {grid_searchs[0].best_estimator_.__class__.__name__}")
    print(f"Best score: {grid_searchs[0].best_score_}")
    return grid_searchs

if __name__ == "__main__":
    data, X_data, Y, T, W, X_data_train, X_data_test, Y_train, Y_test, T_train, T_test, W_train, W_test = load_data()

    models = find_models_and_params(regression_learners, arguments, score_func, np.concatenate((X_data, W, T.reshape(-1, 1)), axis=1), Y)

    models2 = find_models_and_params(classification_learners, arguments, score_func, np.concatenate((X_data, W), axis=1), T)
    T_est = models2[0].predict(np.concatenate((X_data, W), axis=1))
    Y_est = models[0].predict(np.concatenate((X_data, W, T_est.reshape(-1, 1)), axis=1))
    print(T-T_est)
    print(Y-Y_est)
    model_partial_out_X = find_models_and_params(regression_learners, arguments, score_func, np.reshape(T-T_est, (-1, 1)), np.reshape(Y-Y_est, (-1, 1)))
    print("finished")