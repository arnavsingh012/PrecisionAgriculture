from pathlib import Path

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
    ExtraTreesRegressor,
)
from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)


# 1. Load dataset from the project's data folder
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "crop_yield_in_indian_states.csv"
MODEL_PATH = BASE_DIR / "yield_model.pkl"

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}\n"
        "Place crop_yield_in_indian_states.csv inside the data folder."
    )

df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()

features = [
    "Crop",
    "Crop_Year",
    "Season",
    "State",
    "Annual_Rainfall",
    "Fertilizer",
    "Pesticide"
]

target = "Yield"

missing_columns = [col for col in features + [target] if col not in df.columns]
if missing_columns:
    raise ValueError(f"Dataset is missing required columns: {missing_columns}")

df[target] = pd.to_numeric(df[target], errors="coerce")

model_df = df[features + [target]].copy()
model_df = model_df.replace([np.inf, -np.inf], np.nan)
model_df = model_df.dropna(subset=[target])

X = model_df[features]
y = model_df[target]

if len(model_df) < 2:
    raise ValueError("Not enough valid rows to train and evaluate the models.")

# 2. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# 3. Preprocessing
num_cols = X.select_dtypes(include=np.number).columns
cat_cols = X.select_dtypes(exclude=np.number).columns

preprocessor = ColumnTransformer([
    (
        "num",
        Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]),
        num_cols
    ),
    (
        "cat",
        Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]),
        cat_cols
    )
])

# 4. Define models
models = {
    "Linear Regression": LinearRegression(),
    "SVR": SVR(),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    "Extra Trees": ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    "Decision Tree": DecisionTreeRegressor(random_state=42)
}

# 5. Train and evaluate
results = []
trained_models = {}

for name, model in models.items():
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    metrics = {
        "Model": name,
        "R2": r2_score(y_test, predictions),
        "RMSE": np.sqrt(mean_squared_error(y_test, predictions)),
        "MAE": mean_absolute_error(y_test, predictions)
    }

    results.append(metrics)
    trained_models[name] = pipeline

    print(f"\n{name}")
    print("R2:", round(metrics["R2"], 4))
    print("RMSE:", round(metrics["RMSE"], 4))
    print("MAE:", round(metrics["MAE"], 4))

# 6. Select best model using lowest RMSE
results_df = pd.DataFrame(results).sort_values("RMSE")
best_model_name = results_df.iloc[0]["Model"]
best_model = trained_models[best_model_name]

print("\nBest Model:", best_model_name)

# 7. Save model and information in the project root
bundle = {
    "model": best_model,
    "model_name": best_model_name,
    "metrics": results_df.to_dict("records"),
    "features": features,
    "target": target,
    "all_models": trained_models
}

joblib.dump(bundle, MODEL_PATH)
print(f"\nModel successfully saved as {MODEL_PATH}")
