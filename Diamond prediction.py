from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

RANDOM_STATE = 42

# Path to your dataset (raw string so the backslashes work on Windows)
DATA_PATH = Path(r"C:\Users\ANIL KUMAR SINGH\OneDrive\Desktop\Dimond dataset\diamonds.csv")

# The model and charts are saved in the same folder as this script
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "diamond_model.joblib"
CHART_DIR = BASE_DIR / "charts"
CHART_DIR.mkdir(exist_ok=True)

SHOW_PLOTS = True  # True: show charts in windows. False: only save PNG files


def finish_chart(figure, filename):
    """Save a chart as PNG and (optionally) show it in a window / PyCharm."""
    figure.tight_layout()
    figure.savefig(CHART_DIR / filename, dpi=130)
    if SHOW_PLOTS:
        plt.show()  # close the window to continue to the next chart
    plt.close(figure)


# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
df = pd.read_csv(DATA_PATH)
print("Dataset shape:", df.shape)
print(df.head(), "\n")

# ---------------------------------------------------------------------------
# 2. Clean data (some rows have 0 for x, y or z, which is impossible)
# ---------------------------------------------------------------------------
df = df.drop_duplicates()
df = df[(df[["x", "y", "z"]] > 0).all(axis=1)]
df = df[(df["y"] < 30) & (df["z"] < 30)]  # remove obvious outliers
print("After cleaning:", df.shape, "\n")

cut_order = ["Fair", "Good", "Very Good", "Premium", "Ideal"]
color_order = ["J", "I", "H", "G", "F", "E", "D"]
clarity_order = ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]

# ---------------------------------------------------------------------------
# 3. Exploratory charts (understand the data before modelling)
# ---------------------------------------------------------------------------
# Chart 1: price distribution, before and after the log transform
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
ax[0].hist(df["price"], bins=60, color="steelblue", edgecolor="white")
ax[0].set_title("Price distribution (skewed)")
ax[0].set_xlabel("Price ($)")
ax[0].set_ylabel("Number of diamonds")
ax[1].hist(np.log1p(df["price"]), bins=60, color="seagreen", edgecolor="white")
ax[1].set_title("log(Price) distribution (more balanced)")
ax[1].set_xlabel("log(Price)")
finish_chart(fig, "1_price_distribution.png")

# Chart 2: carat vs price
fig, ax = plt.subplots(figsize=(8, 5))
sample = df.sample(8000, random_state=RANDOM_STATE)
ax.scatter(sample["carat"], sample["price"], alpha=0.3, s=8, color="darkorange")
ax.set_title("Carat vs Price: bigger diamonds cost much more")
ax.set_xlabel("Carat")
ax.set_ylabel("Price ($)")
finish_chart(fig, "2_carat_vs_price.png")

# Chart 3: average price by cut, color and clarity
fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
for a, col, order in zip(ax, ["cut", "color", "clarity"],
                         [cut_order, color_order, clarity_order]):
    avg = df.groupby(col)["price"].mean().reindex(order)
    a.bar(avg.index, avg.values, color="mediumpurple")
    a.set_title(f"Average price by {col}")
    a.set_ylabel("Average price ($)")
    a.tick_params(axis="x", rotation=45)
fig.suptitle("Worst quality on the left, best on the right", y=1.02)
finish_chart(fig, "3_price_by_category.png")

# Chart 4: correlation heatmap of numeric columns
num_df = df[["carat", "depth", "table", "x", "y", "z", "price"]]
corr = num_df.corr()
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr)), corr.columns, rotation=45)
ax.set_yticks(range(len(corr)), corr.columns)
for i in range(len(corr)):
    for j in range(len(corr)):
        ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
fig.colorbar(im)
ax.set_title("Correlation heatmap")
finish_chart(fig, "4_correlation_heatmap.png")

# ---------------------------------------------------------------------------
# 4. Features / target
#    We predict log(price) because price is heavily right-skewed.
# ---------------------------------------------------------------------------
X = df.drop(columns="price")
y_log = np.log1p(df["price"])

cat_cols = ["cut", "color", "clarity"]
num_cols = ["carat", "depth", "table", "x", "y", "z"]

preprocess = ColumnTransformer(
    [
        (
            "cat",
            OrdinalEncoder(categories=[cut_order, color_order, clarity_order]),
            cat_cols,
        ),
        ("num", StandardScaler(), num_cols),
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_log, test_size=0.2, random_state=RANDOM_STATE
)

# ---------------------------------------------------------------------------
# 5. Compare models
# ---------------------------------------------------------------------------
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=200, n_jobs=-1, random_state=RANDOM_STATE
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=300, learning_rate=0.1, max_depth=4, random_state=RANDOM_STATE
    ),
}

results = []
fitted = {}
for name, estimator in models.items():
    pipe = Pipeline([("prep", preprocess), ("model", estimator)])
    pipe.fit(X_train, y_train)
    fitted[name] = pipe

    # Evaluate in real dollars, not log-dollars
    pred = np.expm1(pipe.predict(X_test))
    actual = np.expm1(y_test)
    results.append(
        {
            "Model": name,
            "R2": r2_score(actual, pred),
            "MAE ($)": mean_absolute_error(actual, pred),
            "RMSE ($)": np.sqrt(mean_squared_error(actual, pred)),
        }
    )

results_df = pd.DataFrame(results).sort_values("R2", ascending=False)
print("Model comparison (test set):")
print(results_df.round(3).to_string(index=False), "\n")

# Chart 5: model comparison
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
colors = ["#4c72b0", "#55a868", "#c44e52"]
ax[0].bar(results_df["Model"], results_df["R2"], color=colors)
ax[0].set_ylim(results_df["R2"].min() - 0.02, 1.0)
ax[0].set_title("R² score (higher is better)")
ax[1].bar(results_df["Model"], results_df["MAE ($)"], color=colors)
ax[1].set_title("Average error in $ (lower is better)")
for a in ax:
    a.tick_params(axis="x", rotation=15)
for i, v in enumerate(results_df["R2"]):
    ax[0].text(i, v + 0.001, f"{v:.3f}", ha="center")
for i, v in enumerate(results_df["MAE ($)"]):
    ax[1].text(i, v + 5, f"${v:,.0f}", ha="center")
finish_chart(fig, "5_model_comparison.png")

# ---------------------------------------------------------------------------
# 6. Pick the best model and cross-validate it
# ---------------------------------------------------------------------------
best_name = results_df.iloc[0]["Model"]
best_model = fitted[best_name]
kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)  # shuffle: data is ordered
cv = cross_val_score(best_model, X, y_log, cv=kf, scoring="r2", n_jobs=-1)
print(f"Best model: {best_name}")
print(f"5-fold CV R2 (log-price): {cv.mean():.4f} (+/- {cv.std():.4f})\n")

# ---------------------------------------------------------------------------
# 7. Result charts for the best model
# ---------------------------------------------------------------------------
pred = np.expm1(best_model.predict(X_test))
actual = np.expm1(y_test)

# Chart 6: actual vs predicted + feature importance
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].scatter(actual, pred, alpha=0.25, s=8)
lims = [0, max(actual.max(), pred.max())]
axes[0].plot(lims, lims, "r--", label="Perfect prediction")
axes[0].set_xlabel("Actual price ($)")
axes[0].set_ylabel("Predicted price ($)")
axes[0].set_title(f"{best_name}: Actual vs Predicted")
axes[0].legend()

final_estimator = best_model.named_steps["model"]
if hasattr(final_estimator, "feature_importances_"):
    imp = pd.Series(final_estimator.feature_importances_, index=cat_cols + num_cols)
    imp.sort_values().plot.barh(ax=axes[1], color="teal")
    axes[1].set_title("Feature importance (what drives price?)")
else:
    axes[1].axis("off")
finish_chart(fig, "6_actual_vs_predicted_and_importance.png")

# Chart 7: prediction errors
errors = pred - actual
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(errors, bins=80, range=(-2000, 2000), color="indianred", edgecolor="white")
ax.axvline(0, color="black", linestyle="--")
ax.set_title("Prediction errors (centered near $0 = good)")
ax.set_xlabel("Predicted - Actual price ($)")
ax.set_ylabel("Number of diamonds")
finish_chart(fig, "7_prediction_errors.png")

# ---------------------------------------------------------------------------
# 8. Save the model and predict a new diamond
# ---------------------------------------------------------------------------
joblib.dump(best_model, MODEL_PATH)
print(f"Saved model to {MODEL_PATH}")
print(f"Saved charts to {CHART_DIR}\n")


def predict_price(carat, cut, color, clarity, depth, table, x_mm, y_mm, z_mm):
    """Return the predicted price (USD) for one diamond."""
    loaded_model = joblib.load(MODEL_PATH)
    row = pd.DataFrame(
        [
            {
                "carat": carat, "cut": cut, "color": color, "clarity": clarity,
                "depth": depth, "table": table, "x": x_mm, "y": y_mm, "z": z_mm,
            }
        ]
    )
    return float(np.expm1(loaded_model.predict(row))[0])


if __name__ == "__main__":
    example = dict(
        carat=1.0, cut="Ideal", color="G", clarity="VS1",
        depth=61.5, table=56.0, x_mm=6.4, y_mm=6.43, z_mm=3.95,
    )
    price = predict_price(**example)
    print("Example diamond:", example)
    print(f"Predicted price: ${price:,.2f}")