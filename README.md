# Flight Price Prediction

An end-to-end data science project that predicts flight ticket prices using historical flight data.
The project demonstrates the **full machine learning lifecycle**, with particular emphasis on
**real-world data issues**, **feature engineering**, and **model interpretability**.

---

## Project Objectives

- Analyze historical flight price patterns
- Identify key factors influencing ticket prices
- Handle missing values and injected anomalies
- Engineer meaningful time-based and categorical features
- Train and compare multiple regression models
- Translate model predictions into **buy vs wait** decisions

---

## Project Workflow

### 1. Data Exploration & Cleaning
- One full year of synthetic flight data
- Intentional anomalies (outliers, missing values)
- Data type validation and sanity checks

### 2. Exploratory Data Analysis (EDA)
- Price trends over time
- Route, airline, and seasonality effects
- Correlation analysis and distribution inspection

### 3. Feature Engineering
- Time-based features (days to departure, month, weekend)
- Binning of booking windows
- One-hot encoding of categorical variables
- Log transformation of price to reduce skew

### 4. Modeling & Comparison
- Baseline Linear Regression
- Regularized models (Ridge, Lasso)
- Tree-based models (Random Forest)
- Light hyperparameter tuning
- Model comparison using multiple metrics

### 5. Buy vs Wait Strategy
- Generate future price predictions
- Visualize expected price trends
- Define interpretable decision rules
- Simulate example booking scenarios

---

## Target Variable Design

Flight prices are **right-skewed**, which can negatively impact regression performance.

To address this:
- `price` is log-transformed using `log1p`
- Models are trained and evaluated in **log space**
- Predictions are inverse-transformed (`expm1`) for interpretation

This design improves model stability while preserving business interpretability.

---

## Models Used

- Linear Regression (baseline)
- Ridge Regression
- Lasso Regression
- Random Forest Regressor

Tree-based models outperform linear baselines due to their ability to capture
nonlinear relationships and feature interactions.

---

## Evaluation Metrics

Models are evaluated using:
- **RMSE**
- **MAE**
- **R²**

Metrics are computed in **log space** to align with the training objective.
Predictions are converted back to original price scale for reporting and visualization.

---

## Tech Stack

- **Language:** Python
- **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib
- **Environment:** Jupyter Notebook

---

## How to Run 

1. Clone the repository
bash
   git clone https://github.com/jjianhhao/Flight-Price-Predictor.git

2. Install dependencies
pip install -r requirements.txt

3. Run Notebooks in order

## Results

Tree-based models outperform linear models
Flight prices show strong seasonal and weekday patterns
Airline and route significantly impact price
Log-transformed targets improve predictive stability

## Future Improvements

Add real-world scraped data
Include more routes and airlines
Hyperparameter tuning
Deploy as a web app (Streamlit / FastAPI)
