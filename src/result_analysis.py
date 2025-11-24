from xgboost import XGBRegressor
import pandas as pd
import shap
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from scipy.stats import randint, uniform
import os

trainDFLoc = os.path.join("data", "df_train.csv")
testDFLoc = os.path.join("data", "df_test.csv")
cleanedDFLoc = os.path.join("data", "cleaned_df.csv")

# DF = pd.read_csv(cleanedDFLoc)
DF_TRAIN = pd.read_csv(trainDFLoc)
DF_TEST = pd.read_csv(testDFLoc)

#Defining the features and target
TARGET = 'CO2 Capacity (mmol/g)'
FEATURES = DF_TRAIN.drop(columns=[TARGET]).columns

RANDOM_SEED = 1147

best_model = XGBRegressor(learning_rate = 0.17583, max_depth = 5, n_estimator = 492, min_child_weight = 4).fit(DF_TRAIN[FEATURES], DF_TRAIN[TARGET])

explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(DF_TRAIN[FEATURES])

mean_abs_shap = np.abs(shap_values).mean(axis=0)
feature_names = DF_TRAIN[FEATURES].columns
shap_df = pd.DataFrame({
    "feature": feature_names,
    "mean_abs_shap": mean_abs_shap
})

# Step 3: Pick top 5 features
top_features = shap_df.sort_values("mean_abs_shap", ascending=False).head(8)
top_feature_names = top_features["feature"].tolist()

# Step 4: Buzzplot (barplot showing mean absolute SHAP values)
plt.figure(figsize=(8, 5))
sns.barplot(
    data=top_features,
    x="mean_abs_shap",
    y="feature",
    palette="viridis"
)
plt.xlabel("Mean |SHAP value|")
plt.ylabel("Feature")
plt.title("Top 5 Important Features (SHAP)")
plt.tight_layout()
plt.show()

shap.summary_plot(
    shap_values[:, [DF_TRAIN[FEATURES].columns.get_loc(f) for f in top_feature_names]],
    DF_TRAIN[FEATURES][top_feature_names],
    plot_type="dot",  # "dot" is the classic beeswarm
    show=True
)

shap.plots.waterfall(explainer(DF_TRAIN[FEATURES])[np.argmax(DF_TRAIN[TARGET])])

shap.plots.waterfall(explainer(DF_TRAIN[FEATURES])[444])

#Uncertainty measurement
def train_bootstrap_ensemble(X_train, y_train, n_models=25, sample_frac=1.0, xgb_kwargs=None, random_state=None):
    """
    Train a bootstrap ensemble of XGBoost models.
    Returns a list of trained models.
    """
    rng = np.random.RandomState(random_state)
    models = []
    xgb_kwargs = {} if xgb_kwargs is None else xgb_kwargs

    for i in range(n_models):
        idx = rng.choice(len(X_train), size=int(sample_frac*len(X_train)), replace=True)
        X_s, y_s = X_train.iloc[idx], y_train.iloc[idx]
        model = XGBRegressor(**xgb_kwargs)
        model.fit(X_s, y_s)
        models.append(model)

    return models

def ensemble_predict(models, X):
    """
    Given a list of models, predict mean and std for each sample.
    Returns:
        y_mean: predicted mean
        y_std: predictive uncertainty (std across ensemble)
    """
    preds = np.column_stack([m.predict(X) for m in models])
    y_mean = preds.mean(axis=1)
    y_std = preds.std(axis=1, ddof=0)  # population std
    return y_mean, y_std

def plot_1d_uncertainty(X_test, y_mean, y_std, feature_idx=0):
    """
    X_test: 2D array, n_samples x n_features
    feature_idx: which feature to plot on x-axis
    """
    plt.figure(figsize=(8,5))
    sc = plt.scatter(X_test[:, feature_idx], y_mean, c=y_std, cmap='viridis', s=50)
    plt.colorbar(sc, label='Predictive uncertainty (std)')
    plt.xlabel(f'Feature {feature_idx}')
    plt.ylabel('Prediction')
    plt.title('Predictions colored by uncertainty')
    plt.grid(True)
    plt.show()

def plot_2d_uncertainty(X_test, y_mean, y_std, feature_indices=[0,1]):
    """
    X_test: 2D array, n_samples x n_features
    feature_indices: list of two indices to plot on x and y axis
    """
    plt.figure(figsize=(7,6))
    sc = plt.scatter(
        X_test[:, feature_indices[0]],
        X_test[:, feature_indices[1]],
        c=y_std,
        s=50,
        cmap='viridis'
    )
    plt.colorbar(sc, label='Predictive uncertainty (std)')
    plt.xlabel(f'Feature {feature_indices[0]}')
    plt.ylabel(f'Feature {feature_indices[1]}')
    plt.title('Predictive uncertainty across feature space')
    plt.grid(True)
    plt.show()

xgb_params = {'learning_rate':0.17583, 'max_depth':5, 'n_estimator':492,'min_child_weight':4, 'random_state':RANDOM_SEED}

# Train ensemble
models = train_bootstrap_ensemble(DF_TRAIN[FEATURES], DF_TRAIN[TARGET], n_models=25, xgb_kwargs=xgb_params, random_state=RANDOM_SEED)

# Predict and get uncertainty
y_mean, y_std = ensemble_predict(models, DF_TEST[FEATURES])

preds = best_model.predict(DF_TEST[FEATURES])

sns.scatterplot(x=preds, y=DF_TEST[TARGET], hue=y_std, palette='viridis')
plt.xlabel('Predicted Values')
plt.ylabel('True Values')

sort_idx = np.argsort(preds)
y_pred_sorted = preds[sort_idx]
y_true_sorted = DF_TEST[TARGET].iloc[sort_idx]
y_std_sorted = y_std[sort_idx]

# Step 2: Compute 95% confidence interval
ci_lower = y_pred_sorted - 1.96 * y_std_sorted
ci_upper = y_pred_sorted + 1.96 * y_std_sorted

# Step 3: Plot
plt.figure(figsize=(8,6))

# Confidence interval shaded area
plt.fill_between(y_pred_sorted, ci_lower, ci_upper, color='lightblue', alpha=0.5, label='95% CI')

# Scatter plot with color representing uncertainty
sc = plt.scatter(preds, DF_TEST[TARGET], c=y_std, cmap='magma', s=50, edgecolor='k', alpha=0.8)
plt.colorbar(sc, label='Predictive uncertainty (std)')

# Optional: y=x reference line
plt.plot([DF_TEST[TARGET].min(), DF_TEST[TARGET].max()], [DF_TEST[TARGET].min(), DF_TEST[TARGET].max()], color='red', linestyle='--', label='y=x')

plt.xlabel('Predicted y')
plt.ylabel('True y')
plt.title('Predictions vs True with Uncertainty')
plt.legend()
plt.grid(True)
plt.show()

r2_score(DF_TEST[TARGET], preds)

def ucb_acquisition(y_mean, y_std, kappa=2.0):
    return y_mean + kappa * y_std

def batch_select_diverse(X_pool, scores, k, sigma_dist=0.1):
    chosen = []
    scores = scores.copy()
    for _ in range(k):
        i = np.argmax(scores)
        chosen.append(i)
        dists = np.linalg.norm(X_pool - X_pool[i], axis=1)
        penalty = np.exp(-0.5 * (dists / sigma_dist)**2)
        scores *= (1 - penalty)
        scores[i] = -np.inf
    return chosen

def active_learning_bo(X_train, y_train, X_pool, y_true_pool=None, n_rounds=5, batch_size=5, n_models=25, kappa=2.0, xgb_kwargs=None, random_state=None):
    all_selected_idx = []
    rng = np.random.RandomState(random_state)

    for round_idx in range(n_rounds):
        # 1. Train ensemble
        models = train_bootstrap_ensemble(X_train, y_train, n_models=n_models, xgb_kwargs=xgb_kwargs, random_state=rng.randint(1e6))

        # 2. Predict on candidate pool
        y_mean, y_std = ensemble_predict(models, X_pool)

        # 3. Compute UCB acquisition
        scores = ucb_acquisition(y_mean, y_std, kappa=kappa)

        # 4. Select top-k candidates with diversity
        selected_idx = batch_select_diverse(X_pool, scores, batch_size, sigma_dist=0.1)
        all_selected_idx.extend(selected_idx)

        print(f"Round {round_idx+1}: selected indices -> {selected_idx}")

        # 5. Optionally, get ground truth for selected candidates
        if y_true_pool is not None:
            X_new = X_pool[selected_idx]
            y_new = y_true_pool[selected_idx]
            # 6. Add to training set
            X_train = np.vstack([X_train, X_new])
            y_train = np.hstack([y_train, y_new])

        # 7. Remove selected from pool
        X_pool = np.delete(X_pool, selected_idx, axis=0)
        if y_true_pool is not None:
            y_true_pool = np.delete(y_true_pool, selected_idx, axis=0)

    return all_selected_idx

def plot_predictions_with_uncertainty(X, y_true, y_pred, y_std):
    plt.figure(figsize=(8,6))
    # Sort for smooth CI
    sort_idx = np.argsort(y_pred)
    y_pred_sorted = y_pred[sort_idx]
    y_true_sorted = y_true[sort_idx]
    y_std_sorted = y_std[sort_idx]

    ci_lower = y_pred_sorted - 1.96 * y_std_sorted
    ci_upper = y_pred_sorted + 1.96 * y_std_sorted

    plt.fill_between(y_pred_sorted, ci_lower, ci_upper, color='lightblue', alpha=0.4, label='95% CI')
    sc = plt.scatter(y_pred, y_true, c=y_std, cmap='plasma', s=50, edgecolor='k', alpha=0.8)
    plt.colorbar(sc, label='Predictive uncertainty (std)')
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', label='y=x')
    plt.xlabel('Predicted y')
    plt.ylabel('True y')
    plt.title('Predictions vs True with Uncertainty')
    plt.legend()
    plt.grid(True)
    plt.show()

# def plot_learning_curve_two_axes(model, X, y, cv=5,
#                                  train_sizes=np.linspace(0.1, 1.0, 8)):
#     # ---- GET LEARNING CURVE DATA ----
#     train_sizes_abs, train_scores, valid_scores = learning_curve(
#         estimator=model,
#         X=X,
#         y=y,
#         cv=cv,
#         scoring="neg_mean_squared_error",
#         train_sizes=train_sizes,
#         shuffle=True,
#         random_state=42,
#         n_jobs=-1
#     )

#     train_mse = -train_scores.mean(axis=1)
#     valid_mse = -valid_scores.mean(axis=1)
#     train_std = train_scores.std(axis=1)
#     valid_std = valid_scores.std(axis=1)

#     # ---- CREATE FIGURE ----
#     fig, ax1 = plt.subplots(figsize=(12, 6))

#     # ---------------- LEFT AXIS (TRAIN MSE) ----------------
#     color1 = "tab:blue"
#     ax1.set_xlabel("Number of Training Samples", fontsize=12)
#     ax1.set_ylabel("Training MSE", color=color1, fontsize=12)

#     ax1.plot(train_sizes_abs, train_mse, color=color1, marker="o",
#              label="Training MSE", linewidth=2)
#     ax1.fill_between(train_sizes_abs,
#                      train_mse - train_std,
#                      train_mse + train_std,
#                      color=color1, alpha=0.2)

#     ax1.tick_params(axis="y", labelcolor=color1)
#     ax1.grid(True, alpha=0.3)

#     # ---------------- RIGHT AXIS (CV MSE) ----------------
#     ax2 = ax1.twinx()

#     color2 = "tab:red"
#     ax2.set_ylabel("CV MSE", color=color2, fontsize=12)

#     ax2.plot(train_sizes_abs, valid_mse, color=color2, marker="s",
#              label="CV MSE", linewidth=2)
#     ax2.fill_between(train_sizes_abs,
#                      valid_mse - valid_std,
#                      valid_mse + valid_std,
#                      color=color2, alpha=0.2)

#     ax2.tick_params(axis="y", labelcolor=color2)
#     ax2.set_yticks

#     # FORCE separation of right axis
#     ax2.spines["right"].set_position(("outward", 60))

#     ax1.set_yticks(np.linspace(train_mse.min(), train_mse.max(), 6))
#     ax2.set_yticks(np.linspace(valid_mse.min(), valid_mse.max(), 6))

#     # ---------------- TITLE ----------------
#     plt.title("Learning Curve with Two Y-Axes", fontsize=14)

#     # ---------------- COMBINED LEGEND ----------------
#     lines_1, labels_1 = ax1.get_legend_handles_labels()
#     lines_2, labels_2 = ax2.get_legend_handles_labels()
#     ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right")

#     plt.tight_layout()
#     plt.show()

def plot_learning_curve_two_axes(model, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 10)):
    """
    Plot a learning curve with two y-axes:
    - Left: Training MSE
    - Right: Cross-validation MSE
    """

    # Get learning curve data
    train_sizes, train_scores, cv_scores = learning_curve(
        estimator=model,
        X=X,
        y=y,
        cv=cv,
        scoring="neg_mean_squared_error",
        train_sizes=train_sizes,
        shuffle=True,
        random_state=42,
        n_jobs=-1
    )

    # Convert negative scores to MSE
    train_mse = -train_scores.mean(axis=1)
    cv_mse = -cv_scores.mean(axis=1)

    train_std = train_scores.std(axis=1)
    cv_std = cv_scores.std(axis=1)

    # Create figure + axes
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # LEFT AXIS → TRAINING MSE
    color_train = 'tab:blue'
    ax1.set_xlabel("Number of Training Samples")
    ax1.set_ylabel("Training MSE", color=color_train)
    ax1.plot(train_sizes, train_mse, marker='o', color=color_train, label="Training MSE")
    ax1.fill_between(train_sizes, train_mse - train_std, train_mse + train_std,
                     color=color_train, alpha=0.2)
    ax1.tick_params(axis='y', labelcolor=color_train)

    # RIGHT AXIS → CV MSE
    ax2 = ax1.twinx()
    color_cv = 'tab:red'
    ax2.set_ylabel("Cross-Validation MSE", color=color_cv)
    ax2.plot(train_sizes, cv_mse, marker='s', color=color_cv, label="CV MSE")
    ax2.fill_between(train_sizes, cv_mse - cv_std, cv_mse + cv_std,
                     color=color_cv, alpha=0.2)
    ax2.tick_params(axis='y', labelcolor=color_cv)

    # Title & grid
    plt.title("Learning Curve with Training & CV MSE vs Number of Samples")
    ax1.grid(True)

    fig.tight_layout()
    plt.show()

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
plot_learning_curve_two_axes(xgb_best_model, DF_TRAIN[FEATURES], DF_TRAIN[TARGET])

