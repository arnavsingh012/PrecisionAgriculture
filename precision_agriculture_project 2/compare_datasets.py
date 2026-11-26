
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
)

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
    ExtraTreesRegressor
)
from sklearn.tree import DecisionTreeRegressor


# 1. Dataset paths
DATASETS = {
    "Original Dataset": "data/crop_yield_in_indian_states.csv",
    "APY Dataset": "data/APY.csv"
}

# 2. Common features
FEATURES = ["Crop", "Crop_Year", "Season", "State"]
TARGET = "Yield"

# 3. APY sample size
APY_SAMPLE_SIZE = 30000

# 4. Six ML models
MODELS = {
    "Linear Regression": LinearRegression(),
    "SVR": SVR(),
    "Gradient Boosting": GradientBoostingRegressor(
        random_state=42
    ),
    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ),
    "Decision Tree": DecisionTreeRegressor(
        random_state=42
    ),
    "Extra Trees": ExtraTreesRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
}


def load_and_clean_data(path, dataset_name):

    df = pd.read_csv(path)

    df.columns = df.columns.str.strip()

    for col in ["Crop", "Season", "State"]:
        df[col] = df[col].fillna("Unknown").astype(str).str.strip()

    required = FEATURES + [TARGET]

    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{dataset_name} is missing columns: {missing}"
        )

    df["Crop_Year"] = pd.to_numeric(
        df["Crop_Year"], errors="coerce"
    )

    df[TARGET] = pd.to_numeric(
        df[TARGET], errors="coerce"
    )

    df = df.dropna(subset=[TARGET])

    # Use a reproducible sample only for APY
    if dataset_name == "APY Dataset":
        sample_size = min(
            APY_SAMPLE_SIZE, len(df)
        )

        df = df.sample(
            n=sample_size,
            random_state=42
        )

    print(f"\n{dataset_name}")
    print("Rows used:", len(df))
    print("Missing values:")
    print(df[FEATURES].isnull().sum())

    return df


def evaluate_dataset(df, dataset_name):

    X = df[FEATURES]
    y = df[TARGET]

    numeric_features = ["Crop_Year"]
    categorical_features = [
        "Crop", "Season", "State"
    ]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore")
        )
    ])

    preprocessing = ColumnTransformer([
        (
            "numeric",
            numeric_pipeline,
            numeric_features
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    results = []

    for model_name, model in MODELS.items():

        print(f"\nTraining {model_name} on {dataset_name}...")

        pipeline = Pipeline([
            ("preprocessing", preprocessing),
            ("model", model)
        ])

        # Use a smaller training subset for SVR
        if model_name == "SVR" and len(X_train) > 5000:
            X_model, _, y_model, _ = train_test_split(
                X_train,
                y_train,
                train_size=5000,
                random_state=42
            )
        else:
            X_model = X_train
            y_model = y_train

        pipeline.fit(X_model, y_model)

        predictions = pipeline.predict(X_test)

        result = {
            "Dataset": dataset_name,
            "Model": model_name,
            "Rows": len(df),
            "R2 Score": r2_score(y_test, predictions),
            "RMSE": mean_squared_error(
                y_test, predictions
            ) ** 0.5,
            "MAE": mean_absolute_error(
                y_test, predictions
            )
        }

        results.append(result)

        print(
            f"{model_name}: "
            f"R2 = {result['R2 Score']:.4f}, "
            f"RMSE = {result['RMSE']:.4f}, "
            f"MAE = {result['MAE']:.4f}"
        )

    return results


# 5. Run comparison
all_results = []

for dataset_name, path in DATASETS.items():

    df = load_and_clean_data(path, dataset_name)

    dataset_results = evaluate_dataset(
        df, dataset_name
    )

    all_results.extend(dataset_results)


# 6. Save comparison table
results_df = pd.DataFrame(all_results)

results_df.to_csv(
    "dataset_model_comparison.csv",
    index=False
)

print("\n========== FINAL COMPARISON ==========")
print(results_df.round(4).to_string(index=False))


# 7. Save comparison graphs
for metric in ["R2 Score", "RMSE", "MAE"]:

    pivot = results_df.pivot(
        index="Model",
        columns="Dataset",
        values=metric
    )

    ax = pivot.plot(
        kind="bar",
        figsize=(11, 6)
    )

    ax.set_title(
        f"Dataset Comparison - {metric}"
    )

    ax.set_ylabel(metric)
    ax.set_xlabel("Machine Learning Models")

    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    filename = (
        metric.lower().replace(" ", "_")
        + "_comparison.png"
    )

    plt.savefig(filename)
    plt.close()

print("\nComparison completed!")
print("CSV and all three graphs have been saved.")
