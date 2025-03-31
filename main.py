import numpy as np
import pandas as pd

# Generic ML imports
import lightgbm as lgb
import sklearn.preprocessing
from econml.grf import CausalForest, RegressionForest
from econml.dr import ForestDRLearner
from econml.orf import  DMLOrthoForest
from econml._cate_estimator import BaseCateEstimator
from econml.cate_interpreter import SingleTreeCateInterpreter
from econml.inference._inference import InferenceResults
from econml.validate import BLPEvaluationResults, EvaluationResults, DRTester
from econml.iv.nnet import DeepIV
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.preprocessing import PolynomialFeatures, LabelEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV
from sklift.metrics import metrics
import shap

# EconML imports
from econml.dml import CausalForestDML, LinearDML, NonParamDML, DML, SparseLinearDML

import matplotlib.pyplot as plt


def train_and_interpret(model, Y_train, T_train, X_data_train, W_train):
    model.fit(Y_train, T_train, X=X_data_train, W=W_train)
    print("Done")

    intrp = SingleTreeCateInterpreter(include_model_uncertainty=True, max_depth=3, min_samples_leaf=10)
    intrp.interpret(model, X_data_train)
    plt.figure(figsize=(12, 8), dpi=400)
    intrp.plot(feature_names=X_data.columns)
    plt.title(model.__class__.__name__)
    plt.show()

def test_model(model, Y_test, T_test, X_data_test, W_test):
    print(model.score(Y_test, T_test, X=X_data_test, W=W_test))

def build_biomarker_effects(model, columns_to_analyze):
    effects = []
    effect_stds = []
    for column in columns_to_analyze:
        # Create test data with modified column
        X_modified = X_data_test.copy()
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
    plt.xlabel(f'Difference in Treatment Effect (more means {label_encoder.inverse_transform([1])} is better)')
    plt.title(f'Forest Plot of Treatment Effects for {model.__class__.__name__}')
    plt.axvline(x=0, color='gray', linestyle='--')
    plt.grid(True, axis='x')
    plt.show()

def build_biomarker_effects_with_intervals(model: BaseCateEstimator, columns_to_analyze):
    effects = []
    effect_lb = []
    effect_ub = []
    for column in columns_to_analyze:
        # Create test data with modified column
        X_modified = X_data_test.copy()
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
    plt.xlabel(f'Difference in Treatment Effect (more means {label_encoder.inverse_transform([1])} is better)')
    plt.title(f'Forest Plot of Treatment Effects for {model.__class__.__name__}')
    plt.axvline(x=0, color='gray', linestyle='--')
    plt.grid(True, axis='x')
    plt.show()

def show_shap_plot(model, X_data):
    shap_values = model.shap_values(X_data[:], feature_names=X_data.columns, treatment_names=["gefitinib"], output_names=[Y_col])
    ax = shap.plots.beeswarm(shap_values[Y_col]["gefitinib"], show=False)
    plt.title(f"{Y_col} with {model.__class__.__name__}")
    plt.tight_layout()
    plt.show()

def prepare_data(data, y_col, t_col, w_cols, drop_cols):
    T = data[t_col]
    Y = data[y_col]
    W = data[w_cols]
    X_data = data.drop(columns=[t_col, y_col] + w_cols + drop_cols)
    if not w_cols:
        W = None
    return X_data, Y, T, W

print("Imports successful")

RANDOM_STATE = 15
CV_value= 6
Y_col = "DFS"


def load_data():
    global data, X_data, label_encoder, X_data_train, X_data_test, Y_train, Y_test, T_train, T_test, W_train, W_test
    data = pd.read_csv("data/individualPatients.csv")
    X_data, Y, T, W = prepare_data(data, Y_col, "Adj", ["Age", "Sex", "Smoking_history", "Clinical_stage", "N_stage"],
                                   ["PatientID", "OS", "DFS", "OS_status", "DFS_status"])
    Y = Y.values
    label_encoder = LabelEncoder()
    T = label_encoder.fit_transform(T)
    X_data_train, X_data_test = X_data, X_data
    Y_train, Y_test = Y, Y
    T_train, T_test = T, T
    W_train, W_test = W, W
    if W is not None:
        X_data_train, X_data_test, Y_train, Y_test, T_train, T_test, W_train, W_test = train_test_split(X_data, Y, T, W,
                                                                                                        test_size=0.2,
                                                                                                        random_state=RANDOM_STATE)
    else:
        X_data_train, X_data_test, Y_train, Y_test, T_train, T_test = train_test_split(X_data, Y, T, test_size=0.2,
                                                                                       random_state=RANDOM_STATE)
        W_train, W_test = None, None
    data["Adj_encoded"] = T
    return data, X_data, Y, T, W, X_data_train, X_data_test, Y_train, Y_test, T_train, T_test, W_train, W_test

data, X_data, Y, T, W, X_data_train, X_data_test, Y_train, Y_test, T_train, T_test, W_train, W_test = load_data()

if __name__ == "__main__":
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


    # Fit Causal Forest
    # TODO Implement CV to find the best parameters
    model = CausalForestDML(model_y=lgb.LGBMRegressor(), model_t=lgb.LGBMClassifier(), discrete_treatment=True, random_state=RANDOM_STATE, cv=CV_value)
    train_and_interpret(model, Y_train, T_train, X_data_train, W_train)
    test_model(model, Y_test, T_test, X_data_test, W_test)
    print("Finished Model1")
    model2 = LinearDML(model_y=lgb.LGBMRegressor(), model_t=lgb.LGBMClassifier(), discrete_treatment=True, random_state=RANDOM_STATE, cv=CV_value)
    train_and_interpret(model2, Y_train, T_train, X_data_train, W_train)
    test_model(model2, Y_test, T_test, X_data_test, W_test)
    print("Finished Model2")

    model3 = LinearDML(model_y="automl", model_t="automl", discrete_treatment=True, random_state=RANDOM_STATE, cv=CV_value)
    train_and_interpret(model3, Y_train, T_train, X_data_train, W_train)
    test_model(model3, Y_test, T_test, X_data_test, W_test)
    print("Finished Model3")

    model4 = CausalForestDML(model_y=MLPRegressor(), model_t=MLPClassifier(), discrete_treatment=True, random_state=RANDOM_STATE, cv=CV_value)
    train_and_interpret(model4, Y_train, T_train, X_data_train, W_train)
    test_model(model4, Y_test, T_test, X_data_test, W_test)
    print("Finished Model4")

    model5 = NonParamDML(model_y=RandomForestRegressor(), model_t=RandomForestClassifier(), model_final=AdaBoostRegressor(), discrete_treatment=True, random_state=RANDOM_STATE, cv=CV_value)
    model5.fit(Y_train, T_train, X=X_data_train, W=W_train)
    print(model5.score(Y_test, T_test, X=X_data_test, W=W_test))
    print("Finished Model5")

    model6 = SparseLinearDML(model_y=AdaBoostRegressor(), model_t=BernoulliNB(), discrete_treatment=True, random_state=RANDOM_STATE, cv=CV_value)
    model6.fit(Y_train, T_train, X=X_data_train, W=W_train)
    test_model(model6, Y_test, T_test, X_data_test, W_test)
    print("Finished Model6")

    model7 = ForestDRLearner(cv=CV_value, random_state=RANDOM_STATE)
    train_and_interpret(model7, Y_train, T_train, X_data_train, W_train)
    test_model(model7, Y_test, T_test, X_data_test, W_test)
    print("Finished Model7")

    model8 = DMLOrthoForest(model_T=BernoulliNB(), model_Y=AdaBoostRegressor(), model_T_final=AdaBoostRegressor(), model_Y_final=AdaBoostRegressor(), random_state=RANDOM_STATE)
    train_and_interpret(model8, Y_train, T_train, X_data_train, W_train)

    # List of columns to analyze
    columns_to_analyze = ['EGFR_subtype', 'NKX2_1_Gain', 'CDKN2A_Loss', 'PIK3CA', 'TERT_Gain', 'CDK4_Gain', 'STK11_Loss', 'RB1', 'None']

    build_biomarker_effects(model, columns_to_analyze)
    build_biomarker_effects(model2, columns_to_analyze)
    build_biomarker_effects(model3, columns_to_analyze)
    build_biomarker_effects(model4, columns_to_analyze)
    build_biomarker_effects(model5, columns_to_analyze)
    build_biomarker_effects(model6, columns_to_analyze)

    build_biomarker_effects_with_intervals(model, columns_to_analyze)
    build_biomarker_effects_with_intervals(model2, columns_to_analyze)
    build_biomarker_effects_with_intervals(model3, columns_to_analyze)
    build_biomarker_effects_with_intervals(model4, columns_to_analyze)
    # No model5 because no interval
    build_biomarker_effects_with_intervals(model6, columns_to_analyze)

    show_shap_plot(model, X_data)
    show_shap_plot(model2, X_data)
    show_shap_plot(model3, X_data)
    show_shap_plot(model4, X_data)
    show_shap_plot(model5, X_data)
    show_shap_plot(model6, X_data)

# TODO Try qini curve for evaluation of the models
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
