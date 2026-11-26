

from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import joblib


# ==========================================================
# 1. PAGE CONFIGURATION AND FILE PATHS
# ==========================================================

st.set_page_config(
    page_title="Precision Agriculture AI",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"

ORIGINAL_PATH = DATA_DIR / "crop_yield_in_indian_states.csv"
APY_PATH = DATA_DIR / "APY.csv"

MODEL_PATH = BASE_DIR / "yield_model.pkl"

RESULTS_PATH = BASE_DIR / "dataset_model_comparison.csv"

FEATURES = [
    "Crop",
    "Crop_Year",
    "Season",
    "State",
    "Annual_Rainfall",
    "Fertilizer",
    "Pesticide"
]

TARGET = "Yield"


# ==========================================================
# 2. WEBSITE DESIGN
# ==========================================================

st.markdown("""
<style>

.stApp {
    background-color: #111A16;
    color: #F0F5F1;
}

[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
    background-color: #111A16;
}

[data-testid="stSidebar"] {
    background-color: #172D22;
}

h1, h2, h3 {
    color: #A3E635 !important;
}

[data-testid="stMetric"] {
    background-color: #1D2A23;
    padding: 18px;
    border: 1px solid #344C3A;
    border-radius: 14px;
}

[data-testid="stMetricValue"] {
    color: #A3E635;
}

.info-card {
    background: #1D2A23;
    border: 1px solid #344C3A;
    border-radius: 14px;
    padding: 20px;
    height: 100%;
}

.info-card h3 {
    color: #A3E635;
}

.info-card p {
    color: #D4DED6;
}

.result-card {
    background: #1D3325;
    border: 1px solid #65A30D;
    border-radius: 16px;
    padding: 25px;
    text-align: center;
}

.result-value {
    color: #A3E635;
    font-size: 42px;
    font-weight: bold;
}

.stButton button,
.stFormSubmitButton button,
.stDownloadButton button {
    background-color: #65A30D;
    color: white;
    border-radius: 9px;
    border: none;
    font-weight: bold;
}

.stButton button:hover {
    background-color: #4D7C0F;
}

.step-card {
    background: #1D2A23;
    border-left: 4px solid #84CC16;
    border-radius: 10px;
    padding: 15px;
    margin-bottom: 12px;
}

</style>
""", unsafe_allow_html=True)


# ==========================================================
# 3. DATA LOADING FUNCTIONS
# ==========================================================

@st.cache_data
def load_csv(path):

    data = pd.read_csv(path)

    # Clean column names
    data.columns = data.columns.str.strip()

    # Clean categorical values
    for col in ["Crop", "Season", "State"]:
        if col in data.columns:
            data[col] = (
                data[col]
                .fillna("Unknown")
                .astype(str)
                .str.strip()
            )

    return data


@st.cache_resource
def load_model():

    return joblib.load(MODEL_PATH)


@st.cache_data
def load_comparison_results():

    return pd.read_csv(RESULTS_PATH)


# ==========================================================
# 4. CHECK PROJECT FILES
# ==========================================================

required_files = [
    ORIGINAL_PATH,
    APY_PATH,
    MODEL_PATH,
    RESULTS_PATH
]

missing_files = [
    path.name for path in required_files
    if not path.exists()
]

if missing_files:

    st.error("Some project files are missing.")

    st.write("Please check these files:")

    for filename in missing_files:
        st.write("- " + filename)

    st.stop()


# ==========================================================
# 5. LOAD BOTH DATASETS AND THE MODEL
# ==========================================================

try:

    original_df = load_csv(str(ORIGINAL_PATH))

    apy_df = load_csv(str(APY_PATH))

    bundle = load_model()

    comparison_df = load_comparison_results()

except Exception as error:

    st.error(f"Unable to load project files: {error}")
    st.stop()


# ==========================================================
# 6. LOAD THE EXISTING PREDICTION MODEL
# ==========================================================

if isinstance(bundle, dict):

    model = bundle.get("model")

    model_name = bundle.get(
        "model_name",
        "Machine Learning Model"
    )

    model_metrics = pd.DataFrame(
        bundle.get("metrics", [])
    )

else:

    model = bundle

    model_name = type(model).__name__

    model_metrics = pd.DataFrame()


if model is None:

    st.error("The trained prediction model was not found.")

    st.stop()


# ==========================================================
# 7. CLEAN ORIGINAL DATASET FOR PREDICTION
# ==========================================================

analysis_df = original_df.copy()

for col in FEATURES + [TARGET]:

    if col not in analysis_df.columns:
        continue

    if col in [
        "Crop_Year",
        "Annual_Rainfall",
        "Fertilizer",
        "Pesticide",
        "Yield"
    ]:

        analysis_df[col] = pd.to_numeric(
            analysis_df[col],
            errors="coerce"
        )

    else:

        analysis_df[col] = (
            analysis_df[col]
            .fillna("Unknown")
            .astype(str)
            .str.strip()
        )


analysis_df = analysis_df.replace(
    [np.inf, -np.inf],
    np.nan
)

analysis_df = analysis_df.dropna(
    subset=["Yield"]
)

if analysis_df.empty:

    st.error("No valid yield records were found.")

    st.stop()


# ==========================================================
# 8. PREPARE COMPARISON RESULTS
# ==========================================================

comparison_df = comparison_df.rename(
    columns={
        "R2 Score": "R2",
        "R²": "R2"
    }
)

required_metrics = [
    "Dataset",
    "Model",
    "R2",
    "RMSE",
    "MAE"
]

missing_metrics = [
    col for col in required_metrics
    if col not in comparison_df.columns
]

if missing_metrics:

    st.error(
        "The comparison CSV is missing columns: "
        + ", ".join(missing_metrics)
    )

    st.stop()


for col in ["R2", "RMSE", "MAE"]:

    comparison_df[col] = pd.to_numeric(
        comparison_df[col],
        errors="coerce"
    )


# Clean saved model metrics
model_metrics = model_metrics.rename(
    columns={
        "R2 Score": "R2",
        "R²": "R2",
        "RMSE Score": "RMSE",
        "MAE Score": "MAE"
    }
)


# ==========================================================
# 9. SIDEBAR NAVIGATION
# ==========================================================

with st.sidebar:

    st.markdown("# 🌱")

    st.title("Precision Agriculture")

    st.caption(
        "AI-Powered Crop Yield Intelligence"
    )

    st.divider()

    page = st.radio(
        "MAIN MENU",
        [
            "Dashboard",
            "Yield Prediction",
            "Dataset Explorer",
            "Model Comparison",
            "Performance Analysis"
        ]
    )

    st.divider()

    st.subheader("🤖 Active Prediction Model")

    st.success(model_name)

    st.caption(
        "This model powers the live yield prediction page."
    )

    st.divider()

    st.subheader("📂 Available Datasets")

    st.write("1. Original Dataset")

    st.write("2. APY Dataset")

    st.divider()

    st.caption(
        "Developed using Python, Streamlit "
        "and Machine Learning."
    )


# ==========================================================
# 10. HELPER FUNCTION FOR DATASET STATISTICS
# ==========================================================

def dataset_statistics(data):

    records = len(data)

    crops = (
        data["Crop"].nunique()
        if "Crop" in data.columns else 0
    )

    states = (
        data["State"].nunique()
        if "State" in data.columns else 0
    )

    average_yield = (
        pd.to_numeric(
            data["Yield"], errors="coerce"
        ).mean()
        if "Yield" in data.columns else np.nan
    )

    return records, crops, states, average_yield


# ==========================================================
# 11. DASHBOARD PAGE
# ==========================================================

if page == "Dashboard":

    st.title("🌱 Precision Agriculture Dashboard")

    st.caption(
        "Monitor agricultural datasets and explore "
        "machine learning performance."
    )

    st.divider()

    st.subheader("📊 Dataset Overview")

    # Statistics for both datasets

    records1, crops1, states1, yield1 = (
        dataset_statistics(original_df)
    )

    records2, crops2, states2, yield2 = (
        dataset_statistics(apy_df)
    )

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("### Original Dataset")

        a, b = st.columns(2)

        a.metric("Records", f"{records1:,}")

        b.metric("Crop Varieties", f"{crops1:,}")

        a, b = st.columns(2)

        a.metric("States", f"{states1:,}")

        b.metric(
            "Average Yield",
            f"{yield1:,.2f}" if pd.notna(yield1) else "N/A"
        )

    with col2:

        st.markdown("### APY Dataset")

        a, b = st.columns(2)

        a.metric("Records", f"{records2:,}")

        b.metric("Crop Varieties", f"{crops2:,}")

        a, b = st.columns(2)

        a.metric("States", f"{states2:,}")

        b.metric(
            "Average Yield",
            f"{yield2:,.2f}" if pd.notna(yield2) else "N/A"
        )

    st.divider()

    st.subheader("🏆 Best Model for Each Dataset")

    for dataset_name in [
        "Original Dataset",
        "APY Dataset"
    ]:

        selected = comparison_df[
            comparison_df["Dataset"] == dataset_name
        ]

        if not selected.empty:

            best = selected.loc[
                selected["R2"].idxmax()
            ]

            st.write(
                f"**{dataset_name}:** "
                f"{best['Model']} "
                f"(R² = {best['R2']:.4f})"
            )

    st.divider()

    st.subheader("📈 Model Comparison Preview")

    st.dataframe(
        comparison_df[
            ["Dataset", "Model", "Rows", "R2", "RMSE", "MAE"]
        ].round(4),
        use_container_width=True,
        hide_index=True
    )

    st.info(
        "Open Model Comparison or Performance Analysis "
        "from the sidebar for detailed results."
    )

    st.divider()

    st.subheader("🧠 About This Project")

    a, b, c = st.columns(3)

    with a:

        st.markdown("""
        <div class="info-card">

        <h3>📁 Two Datasets</h3>

        <p>
        Agricultural records from the original
        Indian crop yield dataset and APY dataset.
        </p>

        </div>
        """, unsafe_allow_html=True)

    with b:

        st.markdown("""
        <div class="info-card">

        <h3>🤖 Six Algorithms</h3>

        <p>
        Six regression algorithms are evaluated
        using R², RMSE and MAE.
        </p>

        </div>
        """, unsafe_allow_html=True)

    with c:

        st.markdown("""
        <div class="info-card">

        <h3>🌾 Yield Prediction</h3>

        <p>
        A saved machine learning model estimates
        crop yield using agricultural inputs.
        </p>

        </div>
        """, unsafe_allow_html=True)


# ==========================================================
# 12. YIELD PREDICTION PAGE
# ==========================================================

elif page == "Yield Prediction":

    st.title("🌾 Crop Yield Prediction")

    st.caption(
        "Generate an estimated crop yield using "
        "the trained prediction model."
    )

    st.info(
        "Live predictions use the original dataset "
        "and its seven trained input features."
    )

    st.divider()

    # Dropdown options

    crop_options = sorted(
        analysis_df["Crop"].dropna().astype(str).unique()
    )

    season_options = sorted(
        analysis_df["Season"].dropna().astype(str).unique()
    )

    state_options = sorted(
        analysis_df["State"].dropna().astype(str).unique()
    )

    # Numeric default values

    defaults = {}

    for col, fallback in {
        "Annual_Rainfall": 1000.0,
        "Fertilizer": 1000.0,
        "Pesticide": 100.0
    }.items():

        values = pd.to_numeric(
            analysis_df[col],
            errors="coerce"
        ).dropna()

        defaults[col] = (
            float(values.median())
            if not values.empty else fallback
        )

    years = pd.to_numeric(
        analysis_df["Crop_Year"],
        errors="coerce"
    ).dropna()

    min_year = (
        max(1900, int(years.min()))
        if not years.empty else 1990
    )

    latest_year = (
        int(years.max())
        if not years.empty else 2025
    )

    max_year = max(latest_year + 15, min_year)

    with st.form("prediction_form"):

        st.subheader("1. Crop and Location")

        a, b = st.columns(2)

        with a:

            crop = st.selectbox(
                "Select Crop",
                crop_options
            )

            state = st.selectbox(
                "Select State",
                state_options
            )

        with b:

            season = st.selectbox(
                "Select Season",
                season_options
            )

            crop_year = st.number_input(
                "Crop Year",
                min_value=min_year,
                max_value=max_year,
                value=latest_year
            )

        st.divider()

        st.subheader("2. Agricultural Inputs")

        a, b, c = st.columns(3)

        with a:

            rainfall = st.number_input(
                "Annual Rainfall",
                min_value=0.0,
                value=defaults["Annual_Rainfall"]
            )

        with b:

            fertilizer = st.number_input(
                "Fertilizer Usage",
                min_value=0.0,
                value=defaults["Fertilizer"]
            )

        with c:

            pesticide = st.number_input(
                "Pesticide Usage",
                min_value=0.0,
                value=defaults["Pesticide"]
            )

        st.caption(
            "Use the same units as the original training dataset."
        )

        submitted = st.form_submit_button(
            "🌱 GENERATE PREDICTION",
            use_container_width=True
        )

    if submitted:

        input_data = pd.DataFrame(
            [{
                "Crop": crop,
                "Crop_Year": int(crop_year),
                "Season": season,
                "State": state,
                "Annual_Rainfall": rainfall,
                "Fertilizer": fertilizer,
                "Pesticide": pesticide
            }],
            columns=FEATURES
        )

        if int(crop_year) > latest_year:

            st.warning(
                "The selected year is beyond the latest "
                "training year. This prediction may be "
                "less reliable."
            )

        try:

            prediction = float(
                model.predict(input_data)[0]
            )

        except Exception as error:

            st.error(f"Prediction failed: {error}")
            st.stop()

        if not np.isfinite(prediction):

            st.error("The model returned an invalid prediction.")

            st.stop()

        st.divider()

        st.subheader("🎯 Prediction Result")

        st.markdown(
            f"""
            <div class="result-card">

                <p>ESTIMATED CROP YIELD</p>

                <div class="result-value">
                    {prediction:,.2f}
                </div>

                <p>{model_name}</p>

            </div>
            """,
            unsafe_allow_html=True
        )

        if prediction < 0:

            st.warning(
                "The predicted value is negative and "
                "is not a physically meaningful yield."
            )

        # Historical comparison

        historical = analysis_df[
            analysis_df["Crop"].astype(str) == str(crop)
        ]["Yield"].dropna()

        if not historical.empty:

            historical_average = historical.mean()

            difference = prediction - historical_average

            st.subheader("📊 Historical Comparison")

            a, b, c = st.columns(3)

            a.metric(
                "Predicted Yield",
                f"{prediction:,.2f}"
            )

            b.metric(
                "Historical Average",
                f"{historical_average:,.2f}"
            )

            c.metric(
                "Difference",
                f"{difference:+,.2f}"
            )

        st.divider()

        st.subheader("📋 Input Summary")

        st.dataframe(
            input_data,
            use_container_width=True,
            hide_index=True
        )

        st.success("Prediction generated successfully!")

        # Prediction history

        if "history" not in st.session_state:

            st.session_state.history = []

        st.session_state.history.append({
            "Crop": crop,
            "State": state,
            "Season": season,
            "Crop_Year": int(crop_year),
            "Predicted_Yield": prediction
        })

    # Show previous predictions

    if st.session_state.get("history"):

        st.divider()

        st.subheader("🕒 Prediction History")

        history_df = pd.DataFrame(
            st.session_state.history
        )

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "Download Prediction History",
            history_df.to_csv(index=False).encode("utf-8"),
            "prediction_history.csv",
            "text/csv"
        )


# ==========================================================
# 13. DATASET EXPLORER PAGE
# ==========================================================

elif page == "Dataset Explorer":

    st.title("📁 Dataset Explorer")

    st.caption(
        "Explore and compare agricultural records "
        "from both available datasets."
    )

    selected_dataset = st.selectbox(
        "Choose Dataset",
        [
            "Original Dataset",
            "APY Dataset"
        ]
    )

    if selected_dataset == "Original Dataset":

        selected_df = original_df

    else:

        selected_df = apy_df

    st.divider()

    st.subheader(f"📊 {selected_dataset}")

    st.write(
        f"Total records: **{len(selected_df):,}**"
    )

    a, b, c = st.columns(3)

    a.metric(
        "Crop Varieties",
        selected_df["Crop"].nunique()
        if "Crop" in selected_df.columns else 0
    )

    b.metric(
        "States",
        selected_df["State"].nunique()
        if "State" in selected_df.columns else 0
    )

    c.metric(
        "Missing Values",
        int(selected_df.isna().sum().sum())
    )

    st.divider()

    st.subheader("🔍 Filter Records")

    filtered = selected_df.copy()

    available_filters = [
        col for col in ["Crop", "State", "Season"]
        if col in filtered.columns
    ]

    filter_columns = st.columns(
        max(1, len(available_filters))
    )

    for i, col in enumerate(available_filters):

        options = sorted(
            filtered[col].dropna().astype(str).unique()
        )

        with filter_columns[i]:

            selected_value = st.selectbox(
                f"Select {col}",
                ["All"] + options,
                key=f"filter_{selected_dataset}_{col}"
            )

        if selected_value != "All":

            filtered = filtered[
                filtered[col].astype(str) == selected_value
            ]

    st.write(
        f"Showing **{len(filtered):,}** matching records."
    )

    st.dataframe(
        filtered.head(1000),
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "Download Filtered Dataset",
        filtered.to_csv(index=False).encode("utf-8"),
        "filtered_agriculture_data.csv",
        "text/csv"
    )

    st.divider()

    st.subheader("📉 Missing Value Analysis")

    missing_report = pd.DataFrame({
        "Missing Values": selected_df.isna().sum(),
        "Missing Percentage": (
            selected_df.isna().mean() * 100
        ).round(2)
    })

    st.dataframe(
        missing_report,
        use_container_width=True
    )

    st.info(
        "The two datasets have different available columns. "
        "The comparison experiment uses four common features: "
        "Crop, Crop_Year, Season and State."
    )


# ==========================================================
# 14. MODEL COMPARISON PAGE
# ==========================================================

elif page == "Model Comparison":

    st.title("🤖 Machine Learning Model Comparison")

    st.caption(
        "Compare six regression algorithms across "
        "the Original and APY datasets."
    )

    st.divider()

    dataset_choice = st.selectbox(
        "Choose Dataset for Comparison",
        [
            "Both Datasets",
            "Original Dataset",
            "APY Dataset"
        ]
    )

    if dataset_choice == "Both Datasets":

        display_df = comparison_df.copy()

    else:

        display_df = comparison_df[
            comparison_df["Dataset"] == dataset_choice
        ].copy()

    if display_df.empty:

        st.warning("No comparison results found.")

    else:

        st.subheader("📊 Actual Experiment Results")

        st.dataframe(
            display_df[
                ["Dataset", "Model", "Rows", "R2", "RMSE", "MAE"]
            ].round(4),
            use_container_width=True,
            hide_index=True
        )

        st.divider()

        st.subheader("🏆 Best Performing Algorithm")

        best_index = display_df["R2"].idxmax()

        best_model = display_df.loc[best_index]

        a, b, c = st.columns(3)

        a.metric(
            "Best Model",
            best_model["Model"]
        )

        b.metric(
            "Dataset",
            best_model["Dataset"]
        )

        c.metric(
            "R² Score",
            f"{best_model['R2']:.4f}"
        )

        st.divider()

        st.subheader("📈 R² Score Comparison")

        r2_chart = display_df.pivot(
            index="Model",
            columns="Dataset",
            values="R2"
        )

        st.bar_chart(r2_chart)

        st.caption(
            "A higher R² generally indicates better "
            "prediction performance."
        )

        st.divider()

        st.subheader("📉 RMSE Comparison")

        rmse_chart = display_df.pivot(
            index="Model",
            columns="Dataset",
            values="RMSE"
        )

        st.bar_chart(rmse_chart)

        st.caption(
            "Lower RMSE indicates smaller prediction errors."
        )

        st.divider()

        st.subheader("📉 MAE Comparison")

        mae_chart = display_df.pivot(
            index="Model",
            columns="Dataset",
            values="MAE"
        )

        st.bar_chart(mae_chart)

        st.caption(
            "Lower MAE indicates a smaller average "
            "absolute prediction error."
        )

        st.download_button(
            "Download Comparison Results",
            display_df.to_csv(index=False).encode("utf-8"),
            "model_comparison_results.csv",
            "text/csv"
        )

    st.divider()

    st.info(
        "The APY experiment uses a reproducible random sample "
        "of 30,000 rows. SVR uses at most 5,000 training rows "
        "per dataset. These are evaluation results, not six "
        "live prediction models."
    )


# ==========================================================
# 15. PERFORMANCE ANALYSIS PAGE
# ==========================================================

elif page == "Performance Analysis":

    st.title("📈 Model Performance Analysis")

    st.caption(
        "Study model accuracy and prediction errors "
        "using three evaluation metrics."
    )

    st.divider()

    metric_choice = st.selectbox(
        "Select Performance Metric",
        [
            "R2",
            "RMSE",
            "MAE"
        ]
    )

    metric_information = {
        "R2": (
            "R² Score",
            "Higher is generally better. It measures "
            "how well the model explains variation "
            "in the target values."
        ),
        "RMSE": (
            "Root Mean Squared Error",
            "Lower is better. It gives more weight "
            "to larger prediction errors."
        ),
        "MAE": (
            "Mean Absolute Error",
            "Lower is better. It measures the average "
            "absolute difference between actual "
            "and predicted values."
        )
    }

    metric_title, explanation = metric_information[
        metric_choice
    ]

    st.subheader(metric_title)

    st.write(explanation)

    st.divider()

    metric_chart = comparison_df.pivot(
        index="Model",
        columns="Dataset",
        values=metric_choice
    )

    st.bar_chart(metric_chart)

    st.divider()

    st.subheader("📋 Detailed Metric Values")

    st.dataframe(
        comparison_df[
            ["Dataset", "Model", "Rows", metric_choice]
        ].sort_values(
            by=metric_choice,
            ascending=(metric_choice != "R2")
        ),
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader("🧠 Understanding the Results")

    if metric_choice == "R2":

        st.write(
            "The original dataset achieved its highest "
            "R² score with Random Forest (0.9857). "
            "The APY sample achieved its highest R² "
            "with Gradient Boosting (0.8159)."
        )

    elif metric_choice == "RMSE":

        st.write(
            "RMSE measures the size of prediction errors. "
            "Smaller values indicate better performance."
        )

    else:

        st.write(
            "MAE measures the average absolute prediction "
            "error. Smaller values indicate better performance."
        )

    st.info(
        "All comparison metrics are loaded from the "
        "completed dataset_model_comparison.csv experiment."
    )
import pandas as pd

for file in [
    "data/crop_yield_in_indian_states.csv",
    "data/APY.csv"
]:
    df = pd.read_csv(file)
    df.columns = df.columns.str.strip()

    print("\nFILE:", file)
    print("Unique region names:", df["State"].nunique())
    print(sorted(df["State"].dropna().astype(str).str.strip().unique()))