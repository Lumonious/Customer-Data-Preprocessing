# Customer Data Preprocessing Assignment

This repository contains an assignment focused on data preprocessing techniques using a synthetic customer dataset. The goal is to perform various data cleaning and transformation steps before building and evaluating a regression model.

-----

## 📦 Dataset: `customer_data`

The `customer_data` is a synthetically generated dataset designed to simulate customer information, including demographics, income, and purchasing behavior. It includes common data quality issues like missing values and outliers, providing a realistic scenario for practicing data preprocessing.

### Dataset Generation Code

The following Python code snippet demonstrates how the `customer_data` DataFrame is generated:

```python
import pandas as pd
import numpy as np

np.random.seed(42)
n = 120

df = pd.DataFrame({
    "age": np.random.randint(18, 65, size=n),
    "gender": np.random.choice(["Male", "Female"], size=n),
    "monthly_income": np.random.normal(8000, 2500, size=n),
    "city": np.random.choice(["İstanbul", "Ankara", "İzmir"], size=n),
    "purchase_score": np.random.randint(1, 100, size=n),
})

# Add missing values
df.loc[np.random.choice(n, 10, replace=False), "monthly_income"] = np.nan
df.loc[np.random.choice(n, 8, replace=False), "city"] = np.nan

# Create meaningful target: spending amount
df["spending"] = df["monthly_income"] * 0.8 + df["age"] * 30 + np.random.normal(0, 700, size=n)

# Add outliers
df.loc[3, "spending"] = 60000
df.loc[10, "monthly_income"] = 50000

df.head()
```

-----

## 📘 Assignment: Data Preprocessing Task (10 Steps)

This assignment guides you through a comprehensive data preprocessing pipeline. Each step builds upon the previous one, preparing the data for effective model training.

### 🧹 1. Missing Value Analysis

  * Identify which columns contain **missing data**.
  * Calculate the **missing rate** for each column.

### 🧼 2. Imputing Missing Values

  * Fill **numerical variables** using `SimpleImputer`.
  * Fill **categorical variables** with the most frequent category.

### 📊 3. Outlier Analysis

  * Create **box plots** for the `spending` and `monthly_income` variables.
  * Detect **outliers** using the Interquartile Range (IQR) method.

### ✂️ 4. Outlier Handling

  * **Cap** or **Winsorize** outliers (e.g., using `clip` method).

### 🔣 5. Encoding Categorical Data

  * Encode `gender` using **`LabelEncoder`**.
  * Encode `city` using **One-Hot Encoding**.

### ⚖️ 6. Scaling Numerical Variables

  * Apply `StandardScaler` or `RobustScaler`.
  * **Scale only the numerical columns**.

### 📈 7. Apply Log Transformation

  * Apply `np.log1p()` to the `spending` variable.
  * **Visualize its distribution** after transformation.

### 🤖 8. Model Building

  * Define `spending_log` as the **dependent variable**.
  * Build a **Linear Regression model** and evaluate it.

### 📉 9. Model Evaluation

  * What is the **R² score**? Interpret its meaning.
  * Consider and discuss the **limitations** of the model.

### 💬 10. Extra Bonus: Random Forest

  * If your R² score is low, don't worry\!
  * Try building the model again using a **Random Forest Regressor**.

-----
