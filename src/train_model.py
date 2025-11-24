import os, sys, urllib.request
import numpy as np
import pprint as pp
import pandas as pd
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, max_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold, cross_val_score

from sklearn.dummy import DummyClassifier, DummyRegressor

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LassoCV

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.linear_model import LassoCV
from sklearn.linear_model import RidgeCV, Ridge, ElasticNetCV, ElasticNet
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import train_test_split

import umap
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform, loguniform
from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge

from sklearn.decomposition import PCA
from umap import UMAP
import shap

from io import StringIO
from io import BytesIO

trainDFLoc = os.path.join("data", "df_train.csv")
testDFLoc = os.path.join("data", "df_test.csv")
cleanedDFLoc = os.path.join("data", "cleaned_df.csv")

# DF = pd.read_csv(cleanedDFLoc)
DF_TRAIN = pd.read_csv(trainDFLoc)
DF_TEST = pd.read_csv(testDFLoc)

RANDOM_SEED = 1147
np.random.seed(RANDOM_SEED)

#Defining a function to get regression metrics of a model
def get_regression_metrics(model, X, y_true, random_seed=1147, n_splits=5):

    kf = KFold(n_splits=5, shuffle=True, random_state=random_seed)

    #Training scores
    model.fit(X, y_true)
    y_pred = model.predict(X)
    mae_train = mean_absolute_error(y_true, y_pred)
    mse_train = mean_squared_error(y_true, y_pred)
    max_error_train = max_error(y_true, y_pred)
    mape_train = mean_absolute_percentage_error(y_true, y_pred)
    r2_score_train = r2_score(y_true, y_pred)

    train_dict = {
        'mae_train': float(mae_train),
        'mse_train': float(mse_train),
        'max_error_train': float(max_error_train),
        'mape_train': float(mape_train),
        'r2_train': float(r2_score_train)
    }

    mae_cv = cross_val_score(model, X, y_true, cv=kf, scoring='neg_mean_absolute_error')
    mse_cv = cross_val_score(model, X, y_true, cv=kf, scoring='neg_mean_squared_error')
    maximum_error_cv = cross_val_score(model, X, y_true, cv=kf, scoring='neg_max_error')
    mape_cv =  cross_val_score(model, X, y_true, cv=kf, scoring='neg_mean_absolute_percentage_error')
    r2_score_cv = cross_val_score(model, X, y_true, cv=kf, scoring='r2')

    cv_dict = {
        'mae_cv': float(mae_cv.mean()),
        'mse_cv': float(mse_cv.mean()),
        'max_error_cv': float(maximum_error_cv.mean()),
        'mape_cv': float(mape_cv.mean()),
        'r2_cv': float(r2_score_cv.mean())
    }

    return train_dict, cv_dict

#Defining the features and target
TARGET = 'CO2 Capacity (mmol/g)'
FEATURES = DF_TRAIN.drop(columns=[TARGET]).columns

#Creating baseline models using dummy regressors
dummy_mean = DummyRegressor(strategy='mean')
dummy_median = DummyRegressor(strategy='median')

#Getting scores for the dummy models
mean_metrics_train, mean_metrics_cv = get_regression_metrics(dummy_mean, DF_TRAIN[FEATURES], DF_TRAIN[TARGET], random_seed=RANDOM_SEED)
median_metrics_train, median_metrics_cv = get_regression_metrics(dummy_median, DF_TRAIN[FEATURES], DF_TRAIN[TARGET], random_seed=RANDOM_SEED)

#Creating baseline model summary dataframe
baseline_df = pd.DataFrame()
for key, value in mean_metrics_train.items():
    baseline_df.loc['Dummy Mean', f'{key}'] = abs(value)
for key, value in median_metrics_train.items():
    baseline_df.loc['Dummy Median', f'{key}'] = abs(value)
for key, value in mean_metrics_cv.items():
    baseline_df.loc['Dummy Mean', f'{key}'] = abs(value)
for key, value in median_metrics_cv.items():
    baseline_df.loc['Dummy Median', f'{key}'] = abs(value)
baseline_df

#Creating more dummy models, this time untuned linear regression
linear_dummy = LinearRegression()
#Getting scores for the baseline model
linear_metrics_train, linear_metrics_cv = get_regression_metrics(linear_dummy, DF_TRAIN[FEATURES], DF_TRAIN[TARGET], random_seed=RANDOM_SEED)

#Getting insight into the linear regression baseline model
linear_params_df = pd.DataFrame(linear_dummy.coef_, index=FEATURES, columns=['Coefficient'])
linear_params_df.sort_values(by='Coefficient', ascending=False)

#Scaling the data to prepare for knn
scaler = StandardScaler()
DF_TRAIN_SCALED = scaler.fit_transform(DF_TRAIN)

#Creating more dummy models, this time k nearest neighbors
k_list = np.arange(1, 11)
r2_baseline_knn = []
for k in k_list:
    knn_dummy = KNeighborsRegressor(n_neighbors=k)
    knn_metrics_train, knn_metrics_cv = get_regression_metrics(knn_dummy, DF_TRAIN[FEATURES], DF_TRAIN[TARGET], random_seed=RANDOM_SEED)
    r2_baseline_knn.append(knn_metrics_cv['r2_cv'])
best_k_baseline = k_list[np.argmax(r2_baseline_knn)]

knn_metrics_train, knn_metrics_cv = get_regression_metrics(KNeighborsRegressor(n_neighbors=best_k_baseline), DF_TRAIN[FEATURES], DF_TRAIN[TARGET], random_seed=RANDOM_SEED)

#Adding the additional baseline models to the data
for key, value in linear_metrics_train.items():
    baseline_df.loc['Linear Regression', f'{key}'] = abs(value)
for key, value in linear_metrics_cv.items():
    baseline_df.loc['Linear Regression', f'{key}'] = abs(value)
for key, value in knn_metrics_train.items():
    baseline_df.loc['KNN', f'{key}'] = abs(value)
for key, value in knn_metrics_cv.items():
    baseline_df.loc['KNN', f'{key}'] = abs(value)
baseline_df

DF_TRAIN.info()

#Defining types of features
NUMERICAL_FEATURES = DF_TRAIN.columns[:17].drop(['CO2 Capacity (mmol/g)'])
CATEGORICAL_FEATURES = DF_TRAIN.columns[17:]

DF_TRAIN[NUMERICAL_FEATURES]

#We are going to try our first model, linear regression with L1 Regularization
def l1_mixed_data_pipeline(
    numeric_features,
    categorical_features,
    scaler="standard",       # "standard", "minmax", or None
    alphas=None,
    cv=5,
    max_iter=5000,
    random_state=42,
):
    # choose scaler
    if scaler == "standard":
        scaler_obj = StandardScaler()
    elif scaler == "minmax":
        scaler_obj = MinMaxScaler()
    elif scaler is None:
        scaler_obj = "passthrough"

    # preprocessing
    preproc = ColumnTransformer(
        transformers=[
            ("num", scaler_obj, numeric_features),
            ("cat", "passthrough", categorical_features)
        ]
    )

    # L1 model with cross-validation
    lasso = LassoCV(
        alphas=alphas,
        cv=cv,
        max_iter=max_iter,
        random_state=random_state
    )

    # full pipeline
    pipe = Pipeline([
        ("preprocess", preproc),
        ("model", lasso)
    ])

    return pipe

lasso_alphas = np.logspace(-3, 1, 1000)
lasso_scalers = ["standard", "minmax"]
lasso_alphas_cv_scores = []

lasso_standard_pipe = l1_mixed_data_pipeline(
    numeric_features=NUMERICAL_FEATURES,
    categorical_features=CATEGORICAL_FEATURES,
    scaler=lasso_scalers[0],
    alphas=lasso_alphas,
    cv=10
  )

lasso_minmax_pipe = l1_mixed_data_pipeline(
    numeric_features=NUMERICAL_FEATURES,
    categorical_features=CATEGORICAL_FEATURES,
    scaler=lasso_scalers[1],
    alphas=lasso_alphas,
    cv=10
  )

lasso_standard_train_score, lasso_standard_cv_score = get_regression_metrics(lasso_standard_pipe, DF_TRAIN[FEATURES], DF_TRAIN[TARGET], random_seed=RANDOM_SEED, n_splits=10)
lasso_minmax_train_score, lasso_minmax_cv_score = get_regression_metrics(lasso_minmax_pipe, DF_TRAIN[FEATURES], DF_TRAIN[TARGET], random_seed=RANDOM_SEED, n_splits=10)

print(lasso_standard_cv_score)
print(lasso_minmax_cv_score)

#Saving regression results to a dataframe

model_results = pd.DataFrame()
for key, value in lasso_standard_train_score.items():
    model_results.loc['Lasso', f'{key}'] = abs(value)
for key, value in lasso_standard_cv_score.items():
    model_results.loc['Lasso', f'{key}'] = abs(value)
model_results.loc['Lasso', 'Number of Features'] = np.sum(lasso_standard_pipe.named_steps['model'].coef_!=0)
model_results.loc['Lasso', 'Description'] = f'best_alpha: {lasso_standard_pipe.named_steps["model"].alpha_:.3f}, scaler: {str(lasso_standard_pipe.named_steps["preprocess"]["num"])}'
model_results

#We are going to try our first model, linear regression with L1 Regularization
def l2_mixed_data_pipeline(
    numeric_features,
    categorical_features,
    scaler="standard",       # "standard", "minmax", or None
    alphas=None,
    cv=5,
    max_iter=5000,
    random_state=42,
):
    # choose scaler
    if scaler == "standard":
        scaler_obj = StandardScaler()
    elif scaler == "minmax":
        scaler_obj = MinMaxScaler()
    elif scaler is None:
        scaler_obj = "passthrough"

    # preprocessing
    preproc = ColumnTransformer(
        transformers=[
            ("num", scaler_obj, numeric_features),
            ("cat", "passthrough", categorical_features)
        ]
    )

    # L1 model with cross-validation
    ridge = RidgeCV(
        alphas=alphas,
        cv=cv
    )

    # full pipeline
    pipe = Pipeline([
        ("preprocess", preproc),
        ("model", ridge)
    ])

    return pipe

ridge_alphas = np.logspace(-3, 1, 100)
ridge_scalers = ["standard", "minmax"]

ridge_standard_pipe = l2_mixed_data_pipeline(
    numeric_features=NUMERICAL_FEATURES,
    categorical_features=CATEGORICAL_FEATURES,
    scaler=ridge_scalers[0],
    alphas=ridge_alphas,
    cv=10
  )

ridge_minmax_pipe = l2_mixed_data_pipeline(
    numeric_features=NUMERICAL_FEATURES,
    categorical_features=CATEGORICAL_FEATURES,
    scaler=ridge_scalers[1],
    alphas=ridge_alphas,
    cv=10
  )

ridge_standard_train_metrics, ridge_standard_cv_metrics = get_regression_metrics(ridge_standard_pipe, DF_TRAIN[FEATURES], DF_TRAIN[TARGET], random_seed=RANDOM_SEED)
ridge_minmax_train_metrics, ridge_minmax_cv_metrics = get_regression_metrics(ridge_minmax_pipe, DF_TRAIN[FEATURES], DF_TRAIN[TARGET], random_seed = RANDOM_SEED)

print(ridge_standard_cv_metrics)
print(ridge_minmax_cv_metrics)

#Adding ridge regression results to the dataframe
for key, value in ridge_standard_train_metrics.items():
    model_results.loc['Ridge', f'{key}'] = abs(value)
for key, value in ridge_standard_cv_metrics.items():
    model_results.loc['Ridge', f'{key}'] = abs(value)
model_results.loc['Ridge', 'Number of Features'] = np.sum(ridge_standard_pipe.named_steps['model'].coef_!=0)
model_results.loc['Ridge', 'Description'] = f'best_alpha: {ridge_standard_pipe.named_steps["model"].alpha_:.3f}, scaler: {str(ridge_standard_pipe.named_steps["preprocess"]["num"])}'

model_results

#Now we define a pipeline for elastic net regression
def elasticnet_mixed_data_pipeline(
    numeric_features,
    categorical_features,
    scaler="standard",      # "standard", "minmax", or None
    alphas=None,            # list of alphas; None lets ElasticNetCV choose
    l1_ratio=(0.1, 0.5, 0.9),  # can be scalar or iterable
    cv=5,
    max_iter=5000,
    random_state=42,
):
    # choose scaler
    if scaler == "standard":
        scaler_obj = StandardScaler()
    elif scaler == "minmax":
        scaler_obj = MinMaxScaler()
    elif scaler is None:
        scaler_obj = "passthrough"

    # preprocessing
    preproc = ColumnTransformer(
        transformers=[
            ("num", scaler_obj, numeric_features),
            ("cat", "passthrough", categorical_features),
        ]
    )

    # Elastic Net with cross-validation
    enet = ElasticNetCV(
        alphas=alphas,
        l1_ratio=l1_ratio,
        cv=cv,
        max_iter=max_iter,
        random_state=random_state
    )

    # full pipeline
    pipeline = Pipeline([
        ("preprocess", preproc),
        ("model", enet)
    ])

    return pipeline

enet_alphas = np.logspace(-3, 1, 100)
enet_l1_ratios = np.linspace(0.1, 0.9, 20)
enet_scalers = ["standard", "minmax"]

enet_standard_pipe = elasticnet_mixed_data_pipeline(
    numeric_features=NUMERICAL_FEATURES,
    categorical_features=CATEGORICAL_FEATURES,
    scaler=enet_scalers[0],
    alphas=enet_alphas,
    l1_ratio=enet_l1_ratios,
    cv=10,
    random_state=RANDOM_SEED
  )

enet_minmax_pipe = elasticnet_mixed_data_pipeline(
    numeric_features=NUMERICAL_FEATURES,
    categorical_features=CATEGORICAL_FEATURES,
    scaler=enet_scalers[1],
    alphas=enet_alphas,
    l1_ratio=enet_l1_ratios,
    cv=10,
    random_state=RANDOM_SEED
  )

enet_train_standard_metrics, enet_cv_standard_metrics = get_regression_metrics(enet_standard_pipe, DF_TRAIN[FEATURES], DF_TRAIN[TARGET], random_seed=RANDOM_SEED)
enet_train_minmax_metrics, enet_cv_minmax_metrics = get_regression_metrics(enet_minmax_pipe, DF_TRAIN[FEATURES], DF_TRAIN[TARGET], random_seed=RANDOM_SEED)

print(enet_cv_standard_metrics)
print(enet_cv_minmax_metrics)

#Adding regression results to the dataframe
for key, value in enet_train_standard_metrics.items():
    model_results.loc['Elastic Net', f'{key}'] = abs(value)
for key, value in enet_cv_standard_metrics.items():
    model_results.loc['Elastic Net', f'{key}'] = abs(value)
model_results.loc['Elastic Net', 'Number of Features'] = np.sum(enet_standard_pipe.named_steps["model"].coef_!=0)
model_results.loc['Elastic Net', 'Description'] = f'best_alpha: {enet_standard_pipe.named_steps["model"].alpha_:.3f}, best_l1_ratio: {enet_standard_pipe.named_steps["model"].l1_ratio_:.3f}, scaler: {str(enet_standard_pipe.named_steps["preprocess"]["num"])}'
model_results

#Now we will try using polynomial regression
def lasso_polynomial_pipeline(
    numeric_features,
    categorical_features=None,
    degree=2,               # degree of polynomial expansion
    interaction_only=False, # if True, only include cross-terms, no squared terms
    scaler="standard",      # "standard", "minmax", or None
    alphas=None,            # list of alphas; None lets LassoCV choose
    cv=5,
    max_iter=5000,
    random_state=42,
):
    # choose scaler
    if scaler == "standard":
        scaler_obj = StandardScaler()
    elif scaler == "minmax":
        scaler_obj = MinMaxScaler()
    elif scaler is None:
        scaler_obj = "passthrough"

    # numeric pipeline: scaling + polynomial expansion
    numeric_pipeline = Pipeline([
        ("scaler", scaler_obj),
        ("poly", PolynomialFeatures(degree=degree, interaction_only=interaction_only, include_bias=False))
    ])

    # preprocessing: numeric + categorical
    transformers = [("num", numeric_pipeline, numeric_features)]
    if len(categorical_features)>0:
        transformers.append(("cat", "passthrough", categorical_features))

    preproc = ColumnTransformer(transformers)

    # Lasso with cross-validation
    lasso = LassoCV(
        alphas=alphas,
        cv=cv,
        max_iter=max_iter,
        random_state=random_state
    )

    # full pipeline
    pipeline = Pipeline([
        ("preprocess", preproc),
        ("model", lasso)
    ])

    return pipeline

poly_alphas = np.logspace(-4, 1, 5)
poly_scaler = ['standard', 'minmax']

poly_standard_pipe = lasso_polynomial_pipeline(
    numeric_features=NUMERICAL_FEATURES,
    categorical_features=CATEGORICAL_FEATURES,
    degree=2,
    interaction_only=False,
    scaler=poly_scaler[0],
    alphas=poly_alphas,
    cv=10,
    max_iter=10000,
    random_state=RANDOM_SEED,
)

poly_minmax_pipe = lasso_polynomial_pipeline(
    numeric_features=NUMERICAL_FEATURES,
    categorical_features=CATEGORICAL_FEATURES,
    degree=2,
    interaction_only=False,
    scaler=poly_scaler[1],
    alphas=poly_alphas,
    cv=10,
    max_iter=10000,
    random_state=RANDOM_SEED,
)

poly_standard_train_metrics, poly_standard_cv_metrics = get_regression_metrics(poly_standard_pipe, DF_TRAIN[FEATURES], DF_TRAIN[TARGET], random_seed=1147, n_splits=5)
poly_minmax_train_metrics, poly_minmax_cv_metrics = get_regression_metrics(poly_minmax_pipe, DF_TRAIN[FEATURES], DF_TRAIN[TARGET], random_seed=1147, n_splits=5)

print(poly_standard_cv_metrics)
print(poly_minmax_cv_metrics)

#Adding regression results to the dataframe
pd.options.display.max_colwidth = None
for key, value in poly_minmax_train_metrics.items():
    model_results.loc['Polynomial', f'{key}'] = abs(value)
for key, value in poly_minmax_cv_metrics.items():
    model_results.loc['Polynomial', f'{key}'] = abs(value)
model_results.loc['Polynomial', 'Number of Features'] = np.sum(poly_minmax_pipe.named_steps['model'].coef_!=0)
model_results.loc['Polynomial','Description'] = f'best_alpha: {poly_minmax_pipe.named_steps["model"].alpha_:.4f}, scaler: {str(poly_minmax_pipe.named_steps["preprocess"]["num"][0])}, regularization: Lasso'
model_results

#Now we will do XGBoost
def make_xgb_pipeline():

    model = XGBRegressor(
        objective="reg:squarederror",
        tree_method="auto",
        random_state=1147
    )

    pipe = Pipeline([
        ("model", model)
    ])

    return pipe

def xgb_with_random_search(X, y, n_iter=80, cv=5):
    """
    Performs RandomizedSearchCV on XGBoost pipeline.
    n_iter controls how many random hyperparameter combinations are tested.
    """

    def make_xgb_pipeline():

        model = XGBRegressor(
            objective="reg:squarederror",
            tree_method="auto",
            random_state=1147
        )

        pipe = Pipeline([
            ("model", model)
        ])

        return pipe

    pipe = make_xgb_pipeline()

    param_dist = {
        "model__n_estimators": randint(300, 1500),
        "model__max_depth": randint(3, 10),
        "model__learning_rate": uniform(0.01, 0.2),
        "model__min_child_weight": randint(1, 10)
    }

    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        random_state=1147,
        verbose=2
    )

    search.fit(X, y)

    print("Best parameters:", search.best_params_)
    print("Best CV score:", search.best_score_)

    return search.best_estimator_

xgb_best_model = xgb_with_random_search(DF_TRAIN[FEATURES], DF_TRAIN[TARGET], n_iter=40, cv=10)

xg_boost_train_metrics, xg_boost_cv_metrics = get_regression_metrics(xgb_best_model, DF_TRAIN[FEATURES], DF_TRAIN[TARGET], random_seed=1147, n_splits=10)

for key, value in xg_boost_train_metrics.items():
    model_results.loc['XGBoost', f'{key}'] = abs(value)
for key, value in xg_boost_cv_metrics.items():
    model_results.loc['XGBoost', f'{key}'] = abs(value)
model_results.loc['XGBoost', 'Number of Features'] = len(DF_TRAIN[FEATURES].columns)
model_results.loc['XGBoost', 'Description'] = f'model__max_depth: {xgb_best_model.named_steps["model"].max_depth}, model__n_estimators: {xgb_best_model.named_steps["model"].n_estimators}, model_min_child_weight: {xgb_best_model.named_steps["model"].min_child_weight}'
model_results

def rf_with_random_search(X, y, n_iter=40, cv=5):
    """
    Performs RandomizedSearchCV on Random Forest pipeline.
    n_iter controls how many random hyperparameter combinations are tested.
    """

    def make_rf_pipeline():
        """Simple Random Forest pipeline when features are numeric / already one-hot encoded."""

        model = RandomForestRegressor(
            random_state=1147,
            n_jobs=-1
        )

        pipe = Pipeline([
            ("model", model)
        ])

        return pipe

    pipe = make_rf_pipeline()

    # Randomized hyperparameter distribution
    param_dist = {
        "model__n_estimators": randint(100, 400),
        "model__max_depth": randint(3, 12),
        "model__min_samples_split": randint(4, 40)
    }

    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        random_state=1147,
        verbose=2
    )

    search.fit(X, y)

    print("Best parameters:", search.best_params_)
    print("Best CV score:", search.best_score_)

    return search.best_estimator_

rf_best_model = rf_with_random_search(DF_TRAIN[FEATURES], DF_TRAIN[TARGET], n_iter=40, cv=10)

rf_train_metrics, rf_cv_metrics = get_regression_metrics(rf_best_model, DF_TRAIN[FEATURES], DF_TRAIN[TARGET], random_seed=1147, n_splits=10)

for key, value in rf_train_metrics.items():
    model_results.loc['Random Forest', f'{key}'] = abs(value)
for key, value in rf_cv_metrics.items():
    model_results.loc['Random Forest', f'{key}'] = abs(value)
model_results.loc['Random Forest', 'Number of Features'] = len(DF_TRAIN[FEATURES].columns)
model_results.loc['Random Forest', 'Description'] = 'max_depth: 18, min_samples_split: 2, n_estimators: 258'
model_results

#K nearest neighbors

def knn_feature_selected(X,y,corr_columns_sorted,scaler='standard'):
    X_loop = X.copy()

    if scaler == 'standard':
        scaler = StandardScaler()
    elif scaler == 'minmax':
        scaler = MinMaxScaler()
    else:
        scaler = None

    X_loop = pd.DataFrame(scaler.fit_transform(X_loop), columns=X_loop.columns)

    k_list = []
    mse_list = []
    cv_initial = -1000000000
    best_k_initial = 1

    #initial fit
    for k in range(1,60):
        kf = KFold(n_splits=10, shuffle=True, random_state=1147)
        knn = KNeighborsRegressor(n_neighbors=k)
        cv_initial_loop = cross_val_score(knn, X_loop, y, cv=kf, scoring='neg_mean_squared_error')

        if cv_initial_loop.mean() > cv_initial:
            cv_initial = cv_initial_loop.mean()
            best_k_initial = k

    k_list.append(best_k_initial)
    mse_list.append(cv_initial)


    for i in np.arange(len(corr_columns_sorted)-1)+1:
        X_loop = X_loop.drop(columns=corr_columns_sorted[-i])
        cv_drop = -1000000000
        best_k_drop = 1

        for k in range(1,60):
            kf = KFold(n_splits=10, shuffle=True, random_state=1147)
            knn = KNeighborsRegressor(n_neighbors=k)
            cv_loop = cross_val_score(knn, X_loop, y, cv=kf, scoring='neg_mean_squared_error')
            if cv_loop.mean() > cv_drop:
                cv_drop = cv_loop.mean()
                best_k_drop = k

        k_list.append(best_k_drop)
        mse_list.append(cv_drop)

    return k_list, mse_list

k_list_standard, mse_list_standard = knn_feature_selected(DF_TRAIN[FEATURES],DF_TRAIN[TARGET],abs(DF_TRAIN.corr()[TARGET]).sort_values(ascending=False)[1:].index,scaler='standard')

knn_df_standard = pd.DataFrame({'Features Removed':np.arange(0,28),'k':k_list_standard,'mse':mse_list_standard})
knn_df_standard.sort_values(by='mse', ascending=False).head(5)

k_list_minmax, mse_list_minmax = knn_feature_selected(DF_TRAIN[FEATURES],DF_TRAIN[TARGET],abs(DF_TRAIN.corr()[TARGET]).sort_values(ascending=False)[1:].index,scaler='minmax')

knn_df_minmax = pd.DataFrame({'Features Removed':np.arange(0,28),'k':k_list_minmax,'mse':mse_list_minmax})
knn_df_minmax.sort_values(by='mse', ascending=False).head(5)

scaler_knn = MinMaxScaler()
scaler_knn.fit(DF_TRAIN[FEATURES])
df_for_knn = scaler_knn.transform(DF_TRAIN[FEATURES].copy())
df_for_knn = pd.DataFrame(df_for_knn, columns=DF_TRAIN[FEATURES].columns)
df_for_knn = df_for_knn.drop(list(abs(DF_TRAIN.corr()[TARGET]).sort_values(ascending=False)[-9:].index), axis=1)


knn_final = KNeighborsRegressor(n_neighbors=3)

knn_train_metrics, knn_cv_metrics = get_regression_metrics(knn_final, df_for_knn, DF_TRAIN[TARGET], random_seed=1147, n_splits=10)

for key, value in knn_train_metrics.items():
    model_results.loc['KNN', f'{key}'] = abs(value)
for key, value in knn_cv_metrics.items():
    model_results.loc['KNN', f'{key}'] = abs(value)
model_results.loc['KNN', 'Number of Features'] = len(df_for_knn.columns)
model_results.loc['KNN', 'Description'] = 'scaler: MinMax, k: 3'
model_results

#PCA pretreatment before Lasso
def l1_pca_pipeline(
    n_components=None,       # number of PCA components or fraction of variance
    alphas=None,
    cv=5,
    max_iter=5000,
    random_state=42
):
    """
    Pipeline: StandardScaler → optional PCA → LassoCV.
    All features are standardized.
    """
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=n_components, random_state=random_state)),
        ("model", LassoCV(
            alphas=alphas,
            cv=cv,
            max_iter=max_iter,
            random_state=random_state
        ))
    ])

    return pipe

lasso_pca_alphas = np.logspace(-3, 1, 1000)

lasso_pca_pipe = l1_pca_pipeline(
    n_components=0.99,
    alphas=lasso_pca_alphas,
    cv=10,
    max_iter=5000,
    random_state=RANDOM_SEED
)

lasso_pca_train, lasso_pca_cv = get_regression_metrics(lasso_pca_pipe, DF_TRAIN[FEATURES], DF_TRAIN[TARGET], random_seed=1147, n_splits=10)

for key, value in lasso_pca_train.items():
    model_results.loc['Lasso_PCA', f'{key}'] = abs(value)
for key, value in lasso_pca_cv.items():
    model_results.loc['Lasso_PCA', f'{key}'] = abs(value)
model_results.loc['Lasso_PCA', 'Number of Features'] = lasso_pca_pipe.named_steps['pca'].n_components_
model_results.loc['Lasso_PCA', 'Description'] = f'scaler: Standard, n_components: 0.99'
model_results

def krr_with_random_search(X, y, n_iter=80, cv=5):
    """
    Performs RandomizedSearchCV on Kernel Ridge Regression pipeline.
    n_iter controls how many random hyperparameter combinations are tested.
    """

    def make_krr_pipeline():
        model = KernelRidge()
        pipe = Pipeline([
            ("model", model)
        ])
        return pipe

    pipe = make_krr_pipeline()

    # Hyperparameter search space
    param_dist = {
        "model__alpha": loguniform(1e-2, 1e1),       # regularization
        "model__kernel": ["linear", "rbf", "poly", "sigmoid"],
        "model__gamma": loguniform(1e-4, 1e1),       # for rbf/poly/sigmoid
        "model__degree": randint(2, 5)               # only used if kernel='poly'
    }

    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        random_state=1147,
        verbose=2
    )

    search.fit(X, y)

    print("Best parameters:", search.best_params_)
    print("Best CV score:", search.best_score_)

    return search.best_estimator_

krr_best_model = krr_with_random_search(DF_TRAIN[FEATURES], DF_TRAIN[TARGET], n_iter=40, cv=10)

krr_train_metrics, krr_cv_metrics = get_regression_metrics(krr_best_model, DF_TRAIN[FEATURES], DF_TRAIN[TARGET], random_seed=RANDOM_SEED, n_splits=10)

for key, value in krr_train_metrics.items():
    model_results.loc['KRR', f'{key}'] = abs(value)
for key, value in krr_cv_metrics.items():
    model_results.loc['KRR', f'{key}'] = abs(value)
model_results.loc['KRR', 'Number of Features'] = len(DF_TRAIN[FEATURES].columns)
model_results.loc['KRR', 'Description'] = f'model__alpha: {krr_best_model.named_steps["model"].alpha:.4f}, model__gamma: {krr_best_model.named_steps["model"].gamma:.4f}, model__kernel: {krr_best_model.named_steps["model"].kernel}'

print("\n\nFinal Model Results", model_results.to_string())