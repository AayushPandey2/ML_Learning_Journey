Logistic Regression

Logistic regression is a **supervised machine-learning algorithm** used mainly for **classification**. Despite its name, it does not predict a continuous number like linear regression. Instead, it estimates the probability that an observation belongs to a class.

For binary classification, the target has two possible values:

- `0` — negative class, such as *benign*, *not spam*, or *no churn*
- `1` — positive class, such as *malignant*, *spam*, or *will churn*

## What problem does it solve?

Logistic regression answers questions such as:

- Is this email spam?
- Is a transaction fraudulent?
- Will a customer leave the service?
- Does this medical record belong to the positive or negative diagnostic class?

It returns a probability, for example `0.82`, which means the model estimates an 82% chance that the observation belongs to class `1`.

## How it works

### 1. Calculate a linear score

The model combines input features using weights and a bias:

\[
z = w_1x_1 + w_2x_2 + \dots + w_nx_n + b
\]

where:

- `x₁, x₂, …, xₙ` are input features
- `w₁, w₂, …, wₙ` are learned weights
- `b` is the bias (intercept)
- `z` is the linear score

### 2. Apply the sigmoid function

The score can be any number, but a probability must be between 0 and 1. Logistic regression uses the sigmoid function:

\[
\sigma(z) = \frac{1}{1 + e^{-z}}
\]

The predicted probability is therefore:

\[
\hat{y} = P(y=1 \mid x) = \sigma(w^Tx+b)
\]

Key behaviour:

| Linear score `z` | Predicted probability `ŷ` |
|---:|---:|
| Large negative value | Close to 0 |
| 0 | 0.5 |
| Large positive value | Close to 1 |

### 3. Convert probability into a class

Using the usual threshold of 0.5:

```text
If ŷ ≥ 0.5, predict class 1
If ŷ < 0.5, predict class 0
```

The threshold can be adjusted. For example, in disease screening a lower threshold may be chosen to reduce false negatives.

## Decision boundary

With the default threshold, the boundary occurs where:

\[
w^Tx+b=0
\]

This makes logistic regression a model with a **linear decision boundary** in the original feature space. It does not need perfectly linearly separable data, but it is most suitable when the classes can be reasonably separated by a linear relationship between the selected features.

## Training: loss and cost

For one example, logistic regression uses **binary cross-entropy loss** (also called log loss):

\[
L(y,\hat{y})=-\left[y\log(\hat{y})+(1-y)\log(1-\hat{y})\right]
\]

This loss penalizes confident wrong predictions heavily:

- If the true label is `1`, predicting a probability close to `1` gives low loss.
- If the true label is `1`, predicting a probability close to `0` gives high loss.
- If the true label is `0`, the reverse is true.

For `m` training examples, the cost is the average loss:

\[
J(w,b)=-\frac{1}{m}\sum_{i=1}^{m}
\left[y^{(i)}\log(\hat{y}^{(i)})+(1-y^{(i)})\log(1-\hat{y}^{(i)})\right]
\]

The model learns weights and bias that minimize this cost. This is equivalent to maximizing the likelihood of the observed labels.

## Gradient descent

Training commonly follows this sequence:

1. Initialise weights and bias.
2. Calculate `z = wᵀx + b`.
3. Convert `z` to probability with sigmoid.
4. Calculate binary cross-entropy loss.
5. Update weights and bias to reduce the cost.
6. Repeat until convergence or until the maximum number of iterations is reached.

The learning rate controls the update size. A learning rate that is too high can prevent convergence, while one that is too low can make training slow.

## Regularization

Regularization reduces overfitting by discouraging excessively large weights.

- **L1 regularization:** can make some weights exactly zero and can help select features.
- **L2 regularization:** shrinks weights smoothly and is commonly used.
- **Elastic Net:** combines L1 and L2 regularization.

In scikit-learn's `LogisticRegression`, `C` is the inverse of regularization strength:

- Smaller `C` → stronger regularization → simpler model
- Larger `C` → weaker regularization → model fits training data more closely

`max_iter` controls the maximum optimization iterations. Increase it if the optimizer has not converged.

## Evaluating a logistic-regression classifier

Use a confusion matrix and class-specific metrics, not accuracy alone.

|  | Predicted 0 | Predicted 1 |
|---|---:|---:|
| Actual 0 | True Negative (TN) | False Positive (FP) |
| Actual 1 | False Negative (FN) | True Positive (TP) |

Important metrics:

\[
\text{Accuracy}=\frac{TP+TN}{TP+TN+FP+FN}
\]

\[
\text{Precision}=\frac{TP}{TP+FP}
\]

\[
\text{Recall}=\frac{TP}{TP+FN}
\]

\[
F_1=2\cdot\frac{\text{Precision}\cdot\text{Recall}}{\text{Precision}+\text{Recall}}
\]

For imbalanced datasets, precision, recall, F1-score, ROC-AUC, and a confusion matrix are generally more informative than accuracy alone.

## Advantages

- Easy to understand and implement.
- Fast to train and predict.
- Produces useful probability estimates.
- Works well for approximately linear classification problems.
- Can be regularized to reduce overfitting.
- Coefficients can be interpreted as changes in log-odds.

## Limitations

- The basic model has a linear decision boundary.
- Complex nonlinear relationships may need feature transformations or another model.
- It can overfit high-dimensional data without appropriate regularization.
- Outliers and strongly correlated features can affect coefficient estimates.
- Class imbalance needs deliberate evaluation and possibly class weighting or resampling.
