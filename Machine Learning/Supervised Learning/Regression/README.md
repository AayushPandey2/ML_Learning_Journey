# Multiple Linear Regression

This project explains and implements **Multiple Linear Regression**, a supervised machine learning algorithm used to predict a continuous numerical value from two or more input features.

For example, a model could predict a house price using its area, number of bedrooms, location score, and age.

## What is Multiple Linear Regression?

Multiple Linear Regression finds the linear relationship between one dependent variable (the value to predict) and multiple independent variables (the inputs used for prediction).

Its general equation is:

```text
y = b₀ + b₁x₁ + b₂x₂ + ... + bₙxₙ
```

Where:

- `y` is the predicted target value.
- `b₀` is the intercept, or the predicted value when every feature is zero.
- `b₁, b₂, ..., bₙ` are coefficients learned by the model.
- `x₁, x₂, ..., xₙ` are the input features.

Each coefficient represents the expected change in the target value for a one-unit change in that feature, while the other features remain constant.

## Project Objective

The goal of this notebook is to:

1. Load and explore a dataset.
2. Prepare features and the target variable.
3. Split the data into training and test sets.
4. Train a Multiple Linear Regression model.
5. Make predictions on unseen data.
6. Evaluate the model's performance.

## Workflow

```text
Dataset → Data exploration → Preprocessing → Train/test split
        → Model training → Predictions → Evaluation
```

## Libraries Used

- **Pandas** — loading and working with tabular data.
- **NumPy** — numerical calculations.
- **Matplotlib** and **Seaborn** — data visualization.
- **Scikit-learn** — splitting data, building the model, and measuring performance.

## How to Run the Notebook

1. Clone this repository.
2. Navigate to this folder.
3. Install the required libraries:

   ```bash
   pip install jupyter numpy pandas matplotlib seaborn scikit-learn
   ```

4. Start Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

5. Open `multiple_linear_regression.ipynb` and run the cells in order.

## Model Evaluation

The notebook may use the following regression metrics:

| Metric | Meaning |
|---|---|
| Mean Absolute Error (MAE) | Average absolute difference between actual and predicted values. Lower is better. |
| Mean Squared Error (MSE) | Average of squared prediction errors. Lower is better. |
| Root Mean Squared Error (RMSE) | Square root of MSE, expressed in the same unit as the target. Lower is better. |
| R² Score | Proportion of variation in the target explained by the model. Closer to 1 is generally better. |

## Assumptions of Multiple Linear Regression

For reliable results, the data should reasonably meet these assumptions:

- **Linearity:** Features have a linear relationship with the target.
- **Independence:** Observations are independent of each other.
- **Homoscedasticity:** Prediction errors have roughly constant variance.
- **Normality of residuals:** Errors are approximately normally distributed.
- **Low multicollinearity:** Input features should not be strongly correlated with one another.

## Key Takeaways

- Multiple Linear Regression is useful when a continuous value depends on several features.
- The model learns a coefficient for each feature to make predictions.
- Data quality, relevant features, and checking assumptions are important for meaningful results.
- Evaluation metrics help determine how well the model generalizes to unseen data.
