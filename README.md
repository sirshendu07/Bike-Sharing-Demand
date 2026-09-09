
# 🚲 Bike Sharing Demand Prediction

**Predicting urban rental demand using historical patterns, weather conditions, and regression modeling.**

---

## 📌 Project Overview
This repository contains the end-to-end Machine Learning workflow for predicting hourly bike rental demand. The project leverages historical data spanning two years to forecast total bike rentals based on environmental and temporal features. 

The primary objective is to navigate the complexities of high-dimensional feature spaces, non-linear human behavior (such as rush-hour commute cycles), and hyperparameter tuning to build a robust predictive model that minimizes the **Root Mean Squared Logarithmic Error (RMSLE)**.

## 📁 Repository Structure
* `ML Assignment 1/`
  * `ML_Assignment_1.ipynb` - The primary Jupyter Notebook containing the full analysis, modeling, and visualization pipeline.
* `README.md` - Project documentation and summary.

---

## 📊 Dataset Description
The dataset consists of hourly rental data encompassing both weather metrics and temporal factors. 
* **Temporal Features:** `datetime`, `season`, `holiday`, `workingday`
* **Weather Features:** `weather` (condition classification), `temp` (Celsius), `atemp` (feels-like), `humidity`, `windspeed`
* **Target Variables:** `casual` (non-registered), `registered` (subscribers), `count` (total rentals)

> **Note on Data Leakage:** The `casual` and `registered` columns perfectly sum to the target `count` variable. To maintain a mathematically sound predictive model, these variables were deliberately dropped during the feature engineering phase to prevent data leakage.

---

## ⚙️ Methodology & Machine Learning Pipeline

### 1. Exploratory Data Analysis (EDA)
* **Correlation Masking:** Utilized Pearson correlation coefficients via a masked heatmap to identify primary drivers (e.g., strong positive correlation with temperature, strong negative with humidity).
* **Categorical Variance:** Deployed boxplots to visualize massive demand shifts across categorical variables like `season` and `weather`.
* **Cyclical Time Analysis:** Mapped the "M-shaped" bi-modal distribution of daily rentals, proving standard linear models fail on non-linear commute hours.

### 2. Feature Engineering & Expansion
* **Polynomial Transformation:** Expanded the feature space to Degree-2 polynomial features to capture critical non-linear interactions (e.g., the combined effect of high temperature and high humidity).
* **Feature Scaling:** Applied `StandardScaler` dynamically inside model pipelines to ensure distance-based regularization penalties were applied fairly without leaking validation data.

### 3. Predictive Modeling & Regularization
To combat the high variance (overfitting) introduced by the high-dimensional polynomial expansion, several regression architectures were evaluated:
* **Baseline Linear Regression:** Served as a naive benchmark (suffered from high bias/underfitting).
* **Pure Polynomial Regression:** Demonstrated the dangers of high variance and overfitting without penalties.
* **Ridge Regression (L2 Penalty):** Tuned across logarithmic $\alpha$ values to shrink feature weights smoothly, yielding the highest overall predictive accuracy.
* **Lasso Regression (L1 Penalty):** Tuned to aggressively push less useful feature coefficients exactly to zero, acting as automatic feature selection and resulting in a highly sparse, interpretable model.

### 4. Residual Diagnostics
Evaluated model health via dual-axis diagnostic plotting:
* **Scatter Checking:** Ensured errors were randomly distributed (homoscedasticity) rather than forming funnel patterns.
* **Density Distribution:** Confirmed the residual errors formed a normal, zero-centered bell curve.

---

## 🏆 Key Results & Model Performance

The models were evaluated using **RMSLE**, which scales logarithmically to gently penalize large under-predictions while compressing extreme outliers.

| Model Stage | Validation RMSLE | Key Observations |
| :--- | :--- | :--- |
| **1. Baseline Linear** | High | Underfits; misses complex weather/time interactions. |
| **2. Pure Polynomial** | Very High | Overfits severely; extreme variance without regularization. |
| **3. Ridge (Best L2)** | **Lowest (Best)** | Controls variance perfectly; retains nuanced feature weights. |
| **4. Lasso (Best L1)** | Moderate | Highly sparse; eliminated massive amounts of features for simplicity. |

**Conclusion:** The **Ridge Regression** model built on Degree-2 polynomial features won by perfectly balancing the bias-variance tradeoff. It maintained the complex curves needed to track rush-hour demand while using the L2 penalty to suppress the noise of the expanded feature space.

---

## 🚀 How to Run the Code
1. Clone this repository to your local machine.
2. Open the `ML_Assignment_1.ipynb` notebook using Google Colab or Jupyter Notebook.
3. Ensure you have the required libraries installed:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn
