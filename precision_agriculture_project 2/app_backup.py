
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import joblib


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Precision Agriculture AI",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "crop_yield_in_indian_states.csv"
MODEL_PATH = BASE_DIR / "yield_model.pkl"

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


# ============================================================
# 2. CUSTOM DARK AI AGRICULTURE THEME
# ============================================================

st.markdown("""
<style>

:root {
    color-scheme: dark;
}

.stApp {
    background-color: #111A16 !important;
    color: #F0F5F1 !important;
}

[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
    background-color: #111A16 !important;
}

[data-testid="stHeader"] {
    background-color: #111A16 !important;
}

/* Headings */

h1, h2, h3, h4 {
    color: #A3E635 !important;
}

/* Normal text */

[data-testid="stMain"] p,
[data-testid="stMain"] label,
[data-testid="stMain"] li {
    color: #D4DED6 !important;
}

/* Sidebar */

[data-testid="stSidebar"] {
    background-color: #172D22 !important;
}

[data-testid="stSidebar"] * {
    color: #FFFFFF !important;
}

/* Metric cards */

[data-testid="stMetric"] {
    background-color: #1D2A23 !important;
    padding: 20px !important;
    border: 1px solid #344C3A !important;
    border-radius: 15px !important;
}

[data-testid="stMetricLabel"] {
    color: #B8C9BA !important;
}

[data-testid="stMetricValue"] {
    color: #A3E635 !important;
    font-weight: 800 !important;
}

/* Information cards */

.info-card {
    background-color: #1D2A23 !important;
    padding: 22px;
    border: 1px solid #344C3A;
    border-radius: 15px;
    height: 100%;
}

.info-card h3 {
    color: #A3E635 !important;
}

.info-card p {
    color: #D4DED6 !important;
}

/* Prediction result */

.result-card {
    background-color: #1D3325 !important;
    border: 1px solid #65A30D;
    border-radius: 18px;
    padding: 30px;
    text-align: center;
}

.result-label {
    color: #BEF264 !important;
}

.result-value {
    color: #A3E635 !important;
    font-size: 45px;
    font-weight: 800;
}

.result-unit {
    color: #D4DED6 !important;
}

/* Prediction explanation steps */

.step-card {
    background: #1D2A23;
    border: 1px solid #344C3A;
    border-left: 4px solid #84CC16;
    border-radius: 12px;
    padding: 18px;
    margin-bottom: 14px;
}

.step-card b {
    color: #A3E635;
    font-size: 17px;
}

.step-card p {
    color: #D4DED6;
    margin: 8px 0 0 0;
    font-size: 14px;
}

.step-number {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    background: #365314;
    color: #BEF264;
    width: 30px;
    height: 30px;
    border-radius: 50%;
    font-weight: bold;
    margin-right: 10px;
}

/* Input fields */

.stTextInput input,
.stNumberInput input,
[data-baseweb="select"] > div {
    background-color: #25352B !important;
    color: #FFFFFF !important;
    border-color: #4B6650 !important;
}

[role="listbox"] * {
    background-color: #25352B !important;
    color: #FFFFFF !important;
}

/* Buttons */

.stButton button,
.stFormSubmitButton button,
.stDownloadButton button {
    background-color: #65A30D !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 9px !important;
    font-weight: 700 !important;
}

.stButton button:hover,
.stFormSubmitButton button:hover {
    background-color: #4D7C0F !important;
}

/* Tables */

[data-testid="stDataFrame"] {
    background-color: #1D2A23 !important;
    border-radius: 12px;
}

/* Dividers */

hr {
    border-color: #344C3A !important;
}

/* Captions */

[data-testid="stCaptionContainer"] {
    color: #AABBAE !important;
}

/* Expander */

[data-testid="stExpander"] {
    border: 1px solid #344C3A !important;
    border-radius: 12px !important;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# 3. LOAD DATA AND TRAINED MODEL
# ============================================================

@st.cache_data
def load_data():

    data = pd.read_csv(DATA_PATH)
    data.columns = data.columns.str.strip()

    return data


@st.cache_resource
def load_model():

    return joblib.load(MODEL_PATH)


if not DATA_PATH.exists() or not MODEL_PATH.exists():

    st.error("Project files are missing!")

    st.write("Please check this project structure:")

    st.code("""
app.py
yield_model.pkl
data/
    crop_yield_in_indian_states.csv
""")

    st.stop()


try:

    df = load_data()
    bundle = load_model()

except Exception as e:

    st.error(f"Error loading files: {e}")
    st.stop()


# Handle model bundle

if isinstance(bundle, dict):

    model = bundle.get("model")

    model_name = bundle.get(
        "model_name",
        "Machine Learning Regression Model"
    )

    metrics_df = pd.DataFrame(
        bundle.get("metrics", [])
    )

else:

    model = bundle
    model_name = type(model).__name__
    metrics_df = pd.DataFrame()


if model is None:

    st.error("No trained model found in yield_model.pkl.")
    st.stop()


# Validate dataset columns

required_columns = FEATURES + [TARGET]

missing = [
    col for col in required_columns
    if col not in df.columns
]

if missing:

    st.error(f"Missing dataset columns: {missing}")
    st.stop()


# ============================================================
# 4. DATA CLEANING
# ============================================================

analysis_df = df.copy()

for col in FEATURES + [TARGET]:

    if col in analysis_df.columns:

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
                .astype("string")
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

    st.error("No valid yield data found.")
    st.stop()


# ============================================================
# 5. MODEL METRICS CLEANING
# ============================================================

metrics_df = metrics_df.rename(columns={

    "R2 Score": "R2",
    "R²": "R2",
    "RMSE Score": "RMSE",
    "MAE Score": "MAE"

})


# ============================================================
# 6. SIDEBAR NAVIGATION
# ============================================================

with st.sidebar:

    st.markdown(
        "<h1 style='font-size:42px'>🌱</h1>",
        unsafe_allow_html=True
    )

    st.markdown("## Precision Agriculture")

    st.caption(
        "AI-Powered Crop Yield Intelligence"
    )

    st.divider()

    page = st.radio(
        "MAIN MENU",
        [
            "Dashboard",
            "Yield Prediction",
            "Dataset Explorer"
        ]
    )

    st.divider()

    st.markdown("### 🤖 Active AI Model")

    st.success(model_name)

    st.caption(
        "Using your saved and trained machine learning model."
    )

    st.divider()

    st.markdown("### Project Information")

    st.caption(
        "Precision Agriculture AI\n\n"
        "Crop Yield Prediction System"
    )

    st.divider()

    st.caption(
        "Developed using Python, Streamlit "
        "and Machine Learning."
    )


# ============================================================
# 7. MODEL PERFORMANCE FUNCTION
# ============================================================

def show_model_performance():

    st.subheader("🤖 Machine Learning Performance")

    st.caption(
        "Evaluation results from the saved model."
    )

    if metrics_df.empty or not {
        "R2", "RMSE", "MAE"
    }.issubset(metrics_df.columns):

        st.info(
            "Model evaluation metrics are unavailable. "
            "Check your saved model bundle."
        )

        return

    selected = pd.DataFrame()

    if "Model" in metrics_df.columns:

        selected = metrics_df[
            metrics_df["Model"] == model_name
        ]

    else:

        selected = metrics_df

    if selected.empty:

        st.warning(
            "Performance metrics for the active model "
            "could not be found."
        )

        return

    row = selected.iloc[0]

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "R² Score",
        f"{float(row['R2']):.4f}"
    )

    c2.metric(
        "RMSE",
        f"{float(row['RMSE']):,.3f}"
    )

    c3.metric(
        "MAE",
        f"{float(row['MAE']):,.3f}"
    )


# ============================================================
# 8. DASHBOARD
# ============================================================

if page == "Dashboard":

    st.title("🌱 Precision Agriculture Dashboard")

    st.caption(
        "Explore agricultural data and machine learning "
        "insights for crop yield prediction."
    )

    st.divider()

    st.subheader("📊 Dataset Overview")

    total_records = len(analysis_df)

    total_crops = analysis_df["Crop"].nunique()

    total_states = analysis_df["State"].nunique()

    avg_yield = analysis_df["Yield"].mean()

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Total Records",
        f"{total_records:,}"
    )

    c2.metric(
        "Crop Varieties",
        f"{total_crops:,}"
    )

    c3.metric(
        "States",
        f"{total_states:,}"
    )

    c4.metric(
        "Average Yield",
        f"{avg_yield:,.2f}"
    )

    st.divider()

    # Model performance

    show_model_performance()

    st.divider()

    # Agricultural analysis

    st.subheader("🌾 Agricultural Data Analysis")

    left, right = st.columns(2)

    with left:

        st.markdown("#### Average Yield by Crop")

        crop_data = (
            analysis_df
            .groupby("Crop")["Yield"]
            .mean()
            .sort_values(ascending=False)
            .head(10)
        )

        st.bar_chart(
            crop_data,
            height=330
        )

        st.caption(
            "Top 10 crops based on average recorded yield."
        )

    with right:

        st.markdown("#### Crop Distribution")

        crop_distribution = (
            analysis_df["Crop"]
            .value_counts()
            .head(10)
        )

        st.bar_chart(
            crop_distribution,
            height=330
        )

        st.caption(
            "Number of records available for each crop."
        )

    left, right = st.columns(2)

    with left:

        st.markdown("#### Year-wise Yield Trends")

        yearly_data = (
            analysis_df
            .groupby("Crop_Year")["Yield"]
            .mean()
            .dropna()
            .sort_index()
        )

        if not yearly_data.empty:

            st.line_chart(
                yearly_data,
                height=300
            )

        else:

            st.info("Yearly data is unavailable.")

        st.caption(
            "Historical average yield across crop years."
        )

    with right:

        st.markdown("#### State-wise Agricultural Performance")

        state_data = (
            analysis_df
            .groupby("State")["Yield"]
            .mean()
            .sort_values(ascending=False)
            .head(10)
        )

        st.bar_chart(
            state_data,
            height=300
        )

        st.caption(
            "Top 10 states by average recorded yield."
        )

    st.divider()

    # Algorithm comparison

    st.subheader("📈 Algorithm Comparison")

    if not metrics_df.empty and "Model" in metrics_df.columns:

        st.dataframe(
            metrics_df,
            use_container_width=True,
            hide_index=True
        )

        if "RMSE" in metrics_df.columns:

            chart = (
                metrics_df
                .set_index("Model")[["RMSE"]]
                .sort_values("RMSE")
            )

            st.markdown("#### RMSE Comparison")

            st.bar_chart(chart)

            st.caption(
                "Lower RMSE indicates smaller prediction "
                "errors on the held-out test dataset."
            )

    else:

        st.info(
            "No model comparison data was saved "
            "in the current model bundle."
        )

    st.divider()

    # System explanation

    st.subheader("🧠 How Our System Works")

    a, b, c = st.columns(3)

    with a:

        st.markdown("""
        <div class="info-card">

        <h3>🌦️ Climate Analysis</h3>

        <p>
        Uses annual rainfall information to study
        historical climate-related patterns.
        </p>

        </div>
        """, unsafe_allow_html=True)

    with b:

        st.markdown("""
        <div class="info-card">

        <h3>🌱 Agricultural Inputs</h3>

        <p>
        Considers fertilizer and pesticide usage
        along with crop and location information.
        </p>

        </div>
        """, unsafe_allow_html=True)

    with c:

        st.markdown("""
        <div class="info-card">

        <h3>🤖 AI Prediction</h3>

        <p>
        Uses a trained regression model to estimate
        crop yield from the provided features.
        </p>

        </div>
        """, unsafe_allow_html=True)

    st.divider()

    with st.expander("Dataset and Model Information"):

        st.markdown("""
        **Prediction target:** Yield

        **Seven model input features:**

        - Crop
        - Crop Year
        - Season
        - State
        - Annual Rainfall
        - Fertilizer
        - Pesticide

        **Algorithms in the comparative experiment:**

        - Linear Regression
        - Support Vector Regression
        - Gradient Boosting Regressor
        - Random Forest Regressor
        - Decision Tree Regressor
        - Extra Trees Regressor

        The six algorithms are being compared separately
        across the two datasets.

        Area and Production are excluded from the model
        inputs to avoid target leakage when predicting Yield.

        The dataset does not contain direct soil pH or
        laboratory NPK measurements.
        """)


# ============================================================
# 9. YIELD PREDICTION
# ============================================================

elif page == "Yield Prediction":

    st.title("🌾 Crop Yield Prediction")

    st.caption(
        "Enter agricultural and environmental details "
        "to generate an AI-based yield estimate."
    )

    st.divider()

    st.subheader("Enter Agricultural Details")

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

    if not crop_options or not season_options or not state_options:

        st.error("Some categorical input values are missing.")
        st.stop()

    # Numeric defaults

    numeric_defaults = {}

    for col, fallback in {

        "Annual_Rainfall": 1000.0,
        "Fertilizer": 1000.0,
        "Pesticide": 100.0

    }.items():

        values = pd.to_numeric(
            analysis_df[col],
            errors="coerce"
        ).dropna()

        numeric_defaults[col] = (

            max(0.0, float(values.median()))
            if not values.empty else fallback

        )

    # Year range

    years = pd.to_numeric(
        analysis_df["Crop_Year"],
        errors="coerce"
    ).dropna()

    min_year = max(
        1900,
        int(years.min()) if not years.empty else 1990
    )

    latest_year = (
        int(years.max()) if not years.empty else 2025
    )

    max_year = max(
        latest_year + 15,
        min_year
    )

    # Prediction form

    with st.form("yield_prediction_form"):

        st.markdown("### 1. Crop and Location")

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
                value=latest_year,
                step=1
            )

        st.divider()

        st.markdown("### 2. Environmental and Farm Inputs")

        a, b, c = st.columns(3)

        with a:

            rainfall = st.number_input(
                "🌧️ Annual Rainfall",
                min_value=0.0,
                value=numeric_defaults["Annual_Rainfall"],
                step=10.0,
                format="%.2f"
            )

        with b:

            fertilizer = st.number_input(
                "🌱 Fertilizer Usage",
                min_value=0.0,
                value=numeric_defaults["Fertilizer"],
                step=100.0,
                format="%.2f"
            )

        with c:

            pesticide = st.number_input(
                "🧪 Pesticide Usage",
                min_value=0.0,
                value=numeric_defaults["Pesticide"],
                step=10.0,
                format="%.2f"
            )

        st.caption(
            "Enter values using the same units and scales "
            "as those used in your training dataset."
        )

        st.divider()

        submitted = st.form_submit_button(
            "🌱 GENERATE YIELD PREDICTION",
            use_container_width=True
        )

    # Generate prediction

    if submitted:

        input_data = pd.DataFrame([{

            "Crop": crop,
            "Crop_Year": int(crop_year),
            "Season": season,
            "State": state,
            "Annual_Rainfall": rainfall,
            "Fertilizer": fertilizer,
            "Pesticide": pesticide

        }], columns=FEATURES)

        # Future year warning

        if int(crop_year) > latest_year:

            st.warning(
                f"The selected year is beyond the training "
                f"dataset's latest year ({latest_year}). "
                "This prediction involves extrapolation "
                "and may be less reliable."
            )

        try:

            prediction = float(
                model.predict(input_data)[0]
            )

        except Exception as e:

            st.error(f"Prediction failed: {e}")
            st.stop()

        # Validate prediction

        if not np.isfinite(prediction):

            st.error(
                "The model returned an invalid prediction. "
                "Please check the input values and trained model."
            )

            st.stop()

        st.divider()

        st.subheader("🎯 Prediction Result")

        st.markdown(
            f"""
            <div class="result-card">

                <div class="result-label">
                    ESTIMATED CROP YIELD
                </div>

                <div class="result-value">
                    {prediction:,.2f}
                </div>

                <div class="result-unit">
                    Dataset yield units · {model_name}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

        if prediction < 0:

            st.warning(
                "The model returned a negative estimate. "
                "This is not a physically meaningful yield value. "
                "Check whether the inputs are within the training "
                "data range."
            )

        else:

            st.success(
                "Your crop yield prediction was generated successfully!"
            )

        # Historical comparison

        st.subheader("📊 Historical Crop Comparison")

        historical = analysis_df[
            analysis_df["Crop"].astype(str) == str(crop)
        ]["Yield"].dropna()

        if not historical.empty:

            historical_avg = float(historical.mean())

            difference = prediction - historical_avg

            percentage = (
                difference / abs(historical_avg) * 100
                if historical_avg != 0 else np.nan
            )

            a, b, c = st.columns(3)

            a.metric(
                "Predicted Yield",
                f"{prediction:,.2f}"
            )

            b.metric(
                "Historical Crop Average",
                f"{historical_avg:,.2f}"
            )

            c.metric(
                "Difference",
                f"{difference:+,.2f}",
                delta=(
                    f"{percentage:+.1f}%"
                    if np.isfinite(percentage) else None
                )
            )

            st.caption(
                f"Historical benchmark calculated using "
                f"{len(historical):,} records for {crop} "
                f"in the supplied dataset."
            )

        # Input summary

        st.divider()

        st.subheader("📋 Prediction Input Summary")

        st.dataframe(
            input_data,
            use_container_width=True,
            hide_index=True
        )

        # Explainable workflow

        st.divider()

        st.subheader("⚙️ How the AI Generated This Prediction")

        steps = [

            (
                "Input Collection",
                "The seven agricultural and climatic features "
                "were collected through the input form."
            ),

            (
                "Missing Value Handling",
                "The trained preprocessing pipeline applies "
                "the imputers fitted during model training."
            ),

            (
                "Feature Transformation",
                "Numerical variables are standardized and "
                "categorical variables are one-hot encoded "
                "using the fitted transformations."
            ),

            (
                "Machine Learning Processing",
                f"The transformed data is passed through "
                f"the trained {model_name} regression model."
            ),

            (
                "Yield Prediction",
                f"The trained model generates an estimated "
                f"crop yield of {prediction:,.2f}."
            )

        ]

        for i, (title, description) in enumerate(steps, 1):

            st.markdown(
                f"""
                <div class="step-card">

                    <span class="step-number">{i}</span>

                    <b>{title}</b>

                    <p>{description}</p>

                </div>
                """,
                unsafe_allow_html=True
            )

        st.info(
            "The prediction represents an estimate based on "
            "historical data. It is not a guaranteed agricultural outcome."
        )

        # Save prediction history

        if "history" not in st.session_state:

            st.session_state.history = []

        st.session_state.history.append({

            "Crop": crop,
            "State": state,
            "Season": season,
            "Crop_Year": int(crop_year),
            "Predicted_Yield": prediction

        })

    # Prediction history

    if st.session_state.get("history"):

        st.divider()

        st.subheader("🕒 Previous Predictions")

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
            data=history_df.to_csv(index=False).encode("utf-8"),
            file_name="yield_predictions.csv",
            mime="text/csv"
        )


# ============================================================
# 10. DATASET EXPLORER
# ============================================================

elif page == "Dataset Explorer":

    st.title("📁 Dataset Explorer")

    st.caption(
        "Explore agricultural records, filter the dataset "
        "and examine data quality."
    )

    st.divider()

    st.subheader("Dataset Quality Overview")

    missing_count = int(
        df[required_columns].isna().sum().sum()
    )

    duplicates = int(
        df.duplicated().sum()
    )

    a, b, c = st.columns(3)

    a.metric(
        "Total Records",
        f"{len(df):,}"
    )

    b.metric(
        "Missing Model Values",
        f"{missing_count:,}"
    )

    c.metric(
        "Duplicate Records",
        f"{duplicates:,}"
    )

    st.divider()

    st.subheader("🔍 Filter Agricultural Records")

    a, b, c = st.columns(3)

    with a:

        crop_filter = st.selectbox(
            "Crop",
            ["All"] + sorted(
                df["Crop"].dropna().astype(str).unique()
            )
        )

    with b:

        state_filter = st.selectbox(
            "State",
            ["All"] + sorted(
                df["State"].dropna().astype(str).unique()
            )
        )

    with c:

        season_filter = st.selectbox(
            "Season",
            ["All"] + sorted(
                df["Season"].dropna().astype(str).unique()
            )
        )

    filtered = df.copy()

    if crop_filter != "All":

        filtered = filtered[
            filtered["Crop"].astype(str) == crop_filter
        ]

    if state_filter != "All":

        filtered = filtered[
            filtered["State"].astype(str) == state_filter
        ]

    if season_filter != "All":

        filtered = filtered[
            filtered["Season"].astype(str) == season_filter
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
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="agriculture_filtered_data.csv",
        mime="text/csv"
    )

    st.divider()

    st.subheader("📊 Missing Value Analysis")

    missing_report = pd.DataFrame({

        "Missing Values": df.isna().sum(),

        "Missing Percentage": (
            df.isna().mean() * 100
        ).round(2)

    }).sort_values(
        "Missing Values",
        ascending=False
    )

    st.dataframe(
        missing_report,
        use_container_width=True
    )

    st.divider()

    st.warning(
        "The current prediction dataset does not contain "
        "direct soil pH, laboratory NPK or satellite imagery "
        "measurements. The model uses its seven defined input features."
    )

    with st.expander("About the Dataset"):

        st.write(
            "This dataset contains historical crop yield "
            "records from Indian states."
        )

        st.write(
            "The prediction target is Yield. Area and Production "
            "are not used as model input features."
        )

        st.write(
            "The second APY dataset is being used separately "
            "for the comparative machine learning experiment."
        )
