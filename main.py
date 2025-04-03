from typing import List
import gc

import matplotlib.pyplot as plt
import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
from econml._cate_estimator import BaseCateEstimator
from econml.cate_interpreter import SingleTreeCateInterpreter
from econml.dml import CausalForestDML, LinearDML, NonParamDML, DML
from econml.inference._inference import InferenceResults
from econml.orf import DMLOrthoForest
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, AdaBoostRegressor
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.preprocessing import LabelEncoder
from numpy.typing import NDArray
from training_basic_eval import *

print("Importing complete")

def show_overview_data(data):
    print(data.describe())
    print(data.head(5))
    corr_matrix = data.drop(columns=["Adj"]).corr()
    plt.figure(figsize=(10, 8))
    plt.imshow(corr_matrix, cmap='RdYlGn', interpolation='nearest')
    plt.colorbar()
    plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=90)
    plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
    plt.title('Correlation Matrix')
    plt.show()

def plot_gini_curve(element_loss):
    element_loss = np.sort(element_loss)
    n = len(element_loss)

    lorenz_curve = np.cumsum(element_loss) / np.sum(element_loss)
    lorenz_curve = np.insert(lorenz_curve, 0, 0)

    plt.figure(figsize=(8, 6))
    plt.plot(np.linspace(0, 1, n + 1), lorenz_curve, label='Lorenz Curve')
    plt.plot([0, 1], [0, 1], label='Line of Equality', linestyle='--')
    plt.xlabel('Proportion of Samples')
    plt.ylabel('Proportion of Loss')
    plt.title('Gini Curve')
    plt.legend()
    plt.grid()
    plt.show()

def prepare_data(data, y_col, t_col, w_cols, drop_cols):
    t = data[t_col]
    y = data[y_col]
    w = data[w_cols]
    x_data = data.drop(columns=[t_col, y_col] + w_cols + drop_cols)
    if not w_cols:
        w = None
    return x_data, y, t, w

def total_model_evaluation_and_training(model_class, model_params, x_train, y_train, t_train, w_train, x_test, y_test, t_test, w_test, t_encoder, feature_names, interval_available=False, tree_explainer_available=False, shap_available=False, score_available=False):
    model = model_class(**model_params)
    print(f"{model.__class__.__name__} evaluating")
    print(f"with params: {model_params['model_y'].__class__.__name__} and {model_params['model_t'].__class__.__name__}")

    if tree_explainer_available:
        train_and_interpret(model, y_train, t_train, x_train, w_train, feature_names=feature_names)
    else:
        model.fit(Y_train, T_train, X=X_data_train, W=W_train)
        print(f"Model with {model.__class__.__name__} scored {model.score(y_test, t_test, X=x_test, W=w_test)}")
    if interval_available:
        build_biomarker_effects_with_intervals(model, t_encoder, x_test, columns_to_analyze)
    else:
        build_biomarker_effects(model, t_encoder, x_test, columns_to_analyze)
    if shap_available:
        show_shap_plot(model, x_train, Y_col)

    return model

def run_permutation_explainer(fun, x, feature_names):
    explainer = shap.PermutationExplainer(fun, x, feature_names=feature_names)
    return explainer(x)

def try_k_fold(data):
    k_fold = KFold(random_state=RANDOM_STATE, n_splits=CV_value, shuffle=True)
    split_generator = k_fold.split(data) # inlining this caused AccessViolation
    model_collection = []
    for data_train_idxes, data_test_idxes in split_generator:
        data_iter = data.copy()
        x_train, y_train, t_train, w_train = prepare_data(data_iter.iloc[data_train_idxes], Y_col, t_col, w_cols, drop_cols)
        x_test, y_test, t_test, w_test = prepare_data(data_iter.iloc[data_test_idxes], Y_col, t_col, w_cols, drop_cols)

        t_test = label_encoder.transform(t_test)
        t_train = label_encoder.transform(t_train)

        total_model_evaluation_and_training(
            CausalForestDML, {"model_y": MLPRegressor(), "model_t": MLPClassifier(), "discrete_treatment": True,
                              "random_state": RANDOM_STATE, "cv": CV_value, "verbose": 5},
            x_train, y_train, t_train, w_train, x_test, y_test, t_test, w_test, label_encoder, x_train.columns,
            interval_available=True, tree_explainer_available=True, shap_available=True, score_available=True)

    lambda_val = lambda val: np.mean([model.effect(val) for model in model_collection])
    eval_x, _, _, _ = prepare_data(data, Y_col, t_col, w_cols, drop_cols)
    shap_values = run_permutation_explainer(lambda_val, eval_x , data.columns)
    ax = shap.plots.beeswarm(shap_values[Y_col]["gefitinib"], show=False)
    plt.title(f"{Y_col} with avg Model")
    plt.tight_layout()
    plt.show()
    print("Finished Model1 with CV")

def load_data(encoder):
    data = pd.read_csv("data/individualPatients.csv")
    x_data, y, t, w = prepare_data(data, Y_col, t_col, w_cols, drop_cols)
    y = y.values
    t = encoder.fit_transform(t)
    x_data_train, x_data_test = x_data, x_data
    y_train, y_test = y, y
    t_train, t_test = t, t
    w_train, w_test = w, w
    if w is None:
        w_train, w_test = train_test_split(w, test_size=0.2, random_state=RANDOM_STATE)
    else:
        x_data_train, x_data_test, y_train, y_test, t_train, t_test = train_test_split(x_data, y, t, test_size=0.2,
                                                                                       random_state=RANDOM_STATE)
        w_train, w_test = None, None
    return data, x_data, y, t, w, x_data_train, x_data_test, y_train, y_test, t_train, t_test, w_train, w_test

RANDOM_STATE = 15
CV_value= 6
Y_col = "DFS"
t_col = "Adj"
w_cols = ["Age", "Sex", "Smoking_history", "Clinical_stage", "N_stage"]
drop_cols = ["PatientID", "OS", "DFS", "OS_status", "DFS_status"]
columns_to_analyze = ['EGFR_subtype', 'NKX2_1_Gain', 'CDKN2A_Loss', 'PIK3CA', 'TERT_Gain', 'CDK4_Gain', 'STK11_Loss', 'RB1', 'None']



if __name__ == "__main__":

    label_encoder = LabelEncoder()
    dataframe, X_data, Y, T, W, X_data_train, X_data_test, Y_train, Y_test, T_train, T_test, W_train, W_test = load_data(label_encoder)
    show_overview_data(dataframe)
    try_k_fold(dataframe)

    total_model_evaluation_and_training(
        CausalForestDML, {"model_y": lgb.LGBMRegressor(), "model_t": lgb.LGBMClassifier(), "discrete_treatment": True, "random_state": RANDOM_STATE, "cv": CV_value, "verbose": 1},
        X_data_train, Y_train, T_train, W_train, X_data_test, Y_test, T_test, W_test, label_encoder,
        interval_available=True, tree_explainer_available=True, shap_available=True, score_available=True)
    print("Finished Model1")

    total_model_evaluation_and_training(
        LinearDML, {"model_y": lgb.LGBMRegressor(), "model_t": lgb.LGBMClassifier(), "discrete_treatment": True, "random_state": RANDOM_STATE, "cv": CV_value},
        X_data_train, Y_train, T_train, W_train, X_data_test, Y_test, T_test, W_test, label_encoder,
        interval_available=True, tree_explainer_available=True, shap_available=True, score_available=True)
    print("Finished Model2")

    total_model_evaluation_and_training(
        LinearDML, {"model_y": "automl", "model_t": "automl", "discrete_treatment": True, "random_state": RANDOM_STATE, "cv": CV_value},
        X_data_train, Y_train, T_train, W_train, X_data_test, Y_test, T_test, W_test, label_encoder,
        interval_available=True, tree_explainer_available=True, shap_available=True, score_available=True)
    print("Finished Model3")

    total_model_evaluation_and_training(
        CausalForestDML, {"model_y": MLPRegressor(), "model_t": MLPClassifier(), "discrete_treatment": True, "random_state": RANDOM_STATE, "cv": CV_value},
        X_data_train, Y_train, T_train, W_train, X_data_test, Y_test, T_test, W_test, label_encoder,
        interval_available=True, tree_explainer_available=True, shap_available=True, score_available=True)
    print("Finished Model4")

    #High loss
    # No model5 because no interval
    total_model_evaluation_and_training(
        NonParamDML, {"model_y": RandomForestRegressor(), "model_t": RandomForestClassifier(), "model_final": AdaBoostRegressor(), "discrete_treatment": True, "random_state": RANDOM_STATE, "cv": CV_value},
        X_data_train, Y_train, T_train, W_train, X_data_test, Y_test, T_test, W_test, label_encoder,
        interval_available=False, tree_explainer_available=False, shap_available=True, score_available=True)
    print("Finished Model5")

    #show_shap_plot(model8, X_data) takes too long, different explainer+
    total_model_evaluation_and_training(
        DMLOrthoForest, {"model_T": BernoulliNB(), "model_Y": AdaBoostRegressor(), "model_T_final": AdaBoostRegressor(), "model_Y_final": AdaBoostRegressor(), "random_state": RANDOM_STATE},
        X_data_train, Y_train, T_train, W_train, X_data_test, Y_test, T_test, W_test, label_encoder,
        interval_available=True, tree_explainer_available=True, shap_available=False, score_available=False)
    print("Finished Model8")

    total_model_evaluation_and_training(
        DML, {"model_t": BernoulliNB(), "model_y": AdaBoostRegressor(), "model_final": ElasticNet(), "random_state": RANDOM_STATE, "discrete_treatment": True},
        X_data_train, Y_train, T_train, W_train, X_data_test, Y_test, T_test, W_test, label_encoder,
        interval_available=False, tree_explainer_available=False, shap_available=True, score_available=True)
    print("Finished Model9")



