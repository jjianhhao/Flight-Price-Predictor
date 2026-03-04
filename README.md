# Flight Price Prediction

An end-to-end data science project that predicts flight ticket prices using historical flight data.
The project demonstrates the **full machine learning lifecycle**, with particular emphasis on
**real-world data issues**, **feature engineering**, **model interpretability**, and a
**interactive Streamlit dashboard**.

---

## Project Objectives

- Analyze sample flight price patterns
- Identify key factors influencing ticket prices
- Handle missing values and injected anomalies
- Engineer meaningful time-based and categorical features
- Train and compare multiple regression models
- Translate model predictions into **buy vs wait** decisions
- Visualise all findings in an **interactive dashboard**

---

## Project Workflow

### 1. Data Exploration & Cleaning
- One full year of synthetic flight data (1,809 flights across 4 routes and 5 airlines)
- Intentional anomalies (outliers, missing values, 30 duplicate rows)
- Data type validation and sanity checks

### 2. Exploratory Data Analysis (EDA)
- Price trends over time
- Route, airline, and seasonality effects
- Statistical hypothesis testing (ANOVA, Welch's t-test) for each feature
- Correlation analysis and distribution inspection

### 3. Feature Engineering
- Time-based features (days to departure, month, weekend flag)
- Binning of booking windows
- One-hot encoding of categorical variables
- Log transformation of price to reduce skew

### 4. Modeling & Comparison
- Baseline Linear Regression
- Regularized models (Ridge, Lasso)
- Tree-based models (Random Forest)
- Light hyperparameter tuning
- Model comparison using RMSE, MAE, and R²

### 5. Buy vs Wait Strategy
- Generate future price predictions across a 90-day booking window
- Visualize expected price trends
- Define interpretable decision rules (bottom 20th percentile = Buy)
- Simulate example booking scenarios

### 6. Interactive Dashboard
- 4-page Streamlit app with dark theme
- Live model inference and configurable flight scenarios
- Covers: Overview, Price Drivers, Model Performance, Buy vs Wait

---

## Key Findings

| Factor | Effect | p-value |
|---|---|---|
| Route | Strongest driver — JFK-LAX ~3× more than LHR-CDG | 3.31×10⁻²⁸⁷ |
| Airline | Full-service (United/Delta) >> low-cost (Ryanair/Indigo) | 1.40×10⁻²⁸⁷ |
| Booking lead time | Earlier = cheaper; sweet spot is 91–180 days out | 1.13×10⁻²⁵ |
| Day of week | Fri/weekend slightly more expensive than Mon–Wed | 2.97×10⁻⁸ |
| Season | Summer peaks; Winter/Fall cheapest | 0.00255 |
| Departure month | Jun–Aug and Oct–Dec elevated | 0.00756 |
| Holiday | Holiday departures are more expensive | 0.032 |
| Stops | No significant price difference (non-stop vs 1-stop) | 0.536 |

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

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression | 0.2810 | 0.3601 | 0.620 |
| Ridge Regression | 0.2810 | 0.3601 | 0.620 |
| Lasso Regression | 0.2813 | 0.3604 | 0.620 |
| Random Forest | 0.1921 | 0.2706 | 0.789 |
| **Random Forest (Tuned)** | **0.1874** | **0.2652** | **0.796** |

> All metrics computed in log space. Tree-based models outperform linear baselines by ~17 R² points due to their ability to capture nonlinear relationships and feature interactions.

---

## Evaluation Metrics

Models are evaluated using:
- **RMSE** — penalises large errors
- **MAE** — robust to outliers
- **R²** — proportion of variance explained

Metrics are computed in **log space** to align with the training objective.
Predictions are converted back to original price scale for reporting and visualization.

---

## Dashboard

The project includes a Streamlit dashboard (`dashboard.py`) with 4 pages:

- **Overview** — KPI summary, price distribution, route breakdown, key findings
- **Price Drivers** — Interactive tabs for each feature with charts, stats tables, and hypothesis test results
- **Model Performance** — Model comparison charts, R² leaderboard, feature importance
- **Buy vs Wait** — Configurable flight scenario with live BUY/WAIT recommendation and 90-day price trend

---

## Tech Stack

- **Language:** Python
- **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Plotly, Streamlit
- **Environment:** Jupyter Notebook

---

## How to Run

1. Clone the repository
```bash
git clone https://github.com/jjianhhao/Flight-Price-Predictor.git
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Run Notebooks in order (optional — models are pre-trained)

4. Launch the dashboard
```bash
streamlit run dashboard.py
```

---

## Future Improvements

- Add real-world scraped data
- Include more routes and airlines
- Deeper hyperparameter tuning (GridSearchCV / Optuna)
- Deploy as a hosted web app (Streamlit Cloud / FastAPI)
