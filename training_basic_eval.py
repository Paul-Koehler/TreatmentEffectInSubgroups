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

def test_model(model, Y_test, T_test, X_data_test, W_test):
    print(model.score(Y_test, T_test, X=X_data_test, W=W_test))

def build_biomarker_effects(model, t_encoder, x_test, columns_to_analyze):
    effects = []
    effect_stds = []
    for column in columns_to_analyze:
        # Create test dataframe with modified column
        X_modified = x_test.copy()
        if column != 'None':
            X_modified[column] = 1

        # Calculate effect between treatment 1 and treatment 0 for all samples
        effect = model.effect(X_modified, T0=0, T1=1)
        effects.append(np.mean(effect))
        effect_stds.append(np.std(effect))
    # Create forest plot
    plt.figure(figsize=(10, 6))
    y_pos = np.arange(len(columns_to_analyze))
    plt.errorbar(effects, y_pos, xerr=effect_stds, fmt='o', capsize=5)
    plt.yticks(y_pos, columns_to_analyze)
    plt.xlabel(f'Difference in Treatment Effect (more means {t_encoder.inverse_transform([1])} is better)')
    plt.title(f'Forest Plot of Treatment Effects for {model.__class__.__name__}')
    plt.axvline(x=0, color='gray', linestyle='--')
    plt.grid(True, axis='x')
    plt.show()

def train_encoder_for_treatment(encoder, T_train):
    encoder.fit(T_train)
    return encoder

def build_biomarker_effects_with_intervals(model: BaseCateEstimator, t_encoder, x_test, columns_to_analyze):
    effects = []
    effect_lb = []
    effect_ub = []
    for column in columns_to_analyze:
        # Create test dataframe with modified column
        X_modified = x_test.copy()
        if column != 'None':
            X_modified[column] = 1
        effect: InferenceResults = model.effect_inference(X_modified)
        point_est = np.mean(effect.point_estimate)
        lb = np.mean(effect.conf_int(0.05)[0])
        ub = np.mean(effect.conf_int(0.05)[1])
        effects.append(point_est)
        effect_lb.append(lb)
        effect_ub.append(ub)

    effect_lb = np.array(effect_lb)
    effect_ub = np.array(effect_ub)
    effects = np.array(effects)
    plt.figure(figsize=(10, 6))
    y_pos = np.arange(len(columns_to_analyze))
    plt.errorbar(effects, y_pos, xerr=[effects - effect_lb, effect_ub - effects], fmt='o', capsize=5)
    plt.yticks(y_pos, columns_to_analyze)
    plt.xlabel(f'Difference in Treatment Effect (more means {t_encoder.inverse_transform([1])} is better)')
    plt.title(f'Forest Plot of Treatment Effects for {model.__class__.__name__}')
    plt.axvline(x=0, color='gray', linestyle='--')
    plt.grid(True, axis='x')
    plt.show()

def train_and_interpret(model, y_train, t_train, x_train, w_train, *, feature_names=None):
    model.fit(y_train, t_train, X=x_train, W=w_train)

    intrp = SingleTreeCateInterpreter(include_model_uncertainty=True, max_depth=3, min_samples_leaf=10)
    intrp.interpret(model, x_train)
    plt.figure(figsize=(12, 8), dpi=400)
    intrp.plot(feature_names=feature_names)
    plt.title(model.__class__.__name__)
    plt.show()

def show_shap_plot(model, x_data, y_col):
    shap_values = model.shap_values(x_data[:], feature_names=x_data.columns, treatment_names=["gefitinib"], output_names=[y_col])
    ax = shap.plots.beeswarm(shap_values[y_col]["gefitinib"], show=False)
    plt.title(f"{y_col} with {model.__class__.__name__}")
    plt.tight_layout()
    plt.show()