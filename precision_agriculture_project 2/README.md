# Precision Agriculture AI — VS Code project

This project uses your two supplied codes:
- `train_model.py`: trains and compares Linear Regression, SVR, and Gradient Boosting.
- `app.py`: Streamlit dashboard, yield prediction page, and dataset explorer.

## 1. Add the real dataset

Copy `crop_yield_in_indian_states.csv` into the `data/` folder.

Expected location:
`precision_agriculture_project/data/crop_yield_in_indian_states.csv`

The CSV should contain these columns:
`Crop, Crop_Year, Season, State, Annual_Rainfall, Fertilizer, Pesticide, Yield`

## 2. Open the folder in VS Code

Open the extracted `precision_agriculture_project` folder in VS Code.

## 3. Create and activate a virtual environment (Mac)

Open Terminal > New Terminal, then run:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 4. Train the models

```bash
python train_model.py
```

This evaluates the three models and saves the selected model bundle as `yield_model.pkl` in the project root.

## 5. Start the Streamlit website

```bash
streamlit run app.py
```

Streamlit will show a local address in the terminal. Open that address in your browser.

## Important notes

- The supplied `app.py` is a **Streamlit** application, not a Flask application.
- The dataset path has been changed from a computer-specific Downloads path to the project's `data/` folder.
- This dataset supports crop-yield prediction. It does not contain direct measured soil pH or laboratory NPK values, so do not describe the current model as a validated soil-quality predictor.
- Model metrics are produced only when you run `train_model.py` on your dataset. No performance scores are pre-filled.
