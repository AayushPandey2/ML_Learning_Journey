Simple Linear Regression

This project introduces **Simple Linear Regression**, a supervised machine learning algorithm used to predict a continuous numerical value from one input feature.

For example, it can be used to predict a person's salary from their years of experience, or a house price from its area.

## What is Simple Linear Regression?

Simple Linear Regression models the relationship between one independent variable (input feature) and one dependent variable (target value).

It finds the best-fitting straight line through the data using the equation:

```text
y = b₀ + b₁x
```

Where:

- `y` is the predicted target value.
- `x` is the input feature.
- `b₀` is the intercept, the predicted value when `x` is zero.
- `b₁` is the slope (coefficient), showing how much `y` is expected to change when `x` increases by one unit.

## Project Objective

The goal of this notebook is to:

1. Load and understand a dataset.
2. Select the feature and target variable.
3. Split the data into training and testing sets.
4. Train a Simple Linear Regression model.
5. Make predictions on unseen data.
6. Visualize the regression line.
7. Evaluate the model's performance.

## Workflow

```text
Dataset → Data exploration → Train/test split → Model training
        → Predictions → Visualization → Evaluation
```

## Libraries Used

- **Pandas** — loading and working with tabular data.
- **NumPy** — numerical calculations.
- **Matplotlib** — visualizing data and the regression line.
- **Scikit-learn** — splitting data, training the model, and measuring performance.

## How to Run the Notebook

1. Clone this repository.
2. Navigate to this folder.
3. Install the required libraries:

   ```bash
   pip install jupyter numpy pandas matplotlib scikit-learn
   ```

4. Start Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

5. Open the Simple Linear Regression notebook and run the cells in order.

## Model Evaluation

The following metrics can be used to evaluate the model:

| Metric | Meaning |
|---|---|
| Mean Absolute Error (MAE) | Average absolute difference between actual and predicted values. Lower is better. |
| Mean Squared Error (MSE) | Average of squared prediction errors. Lower is better. |
| Root Mean Squared Error (RMSE) | Square root of MSE; it uses the same unit as the target variable. Lower is better. |
| R² Score | Measures the proportion of variation in the target explained by the model. Closer to 1 is generally better. |

## Assumptions

Simple Linear Regression works best when these assumptions are reasonably satisfied:

- **Linearity:** The input and target have an approximately linear relationship.
- **Independence:** Each observation is independent.
- **Homoscedasticity:** Errors have roughly constant variance.
- **Normality of residuals:** Prediction errors are approximately normally distributed.

## Key Takeaways

- Simple Linear Regression predicts a continuous value using one input feature.
- It represents the relationship between variables with a straight line.
- The slope explains the expected change in the prediction for each one-unit change in the input.
- Visualizing the data and regression line helps assess whether a linear model is suitable.
