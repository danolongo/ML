# Your implementation must include:
# 1. Closed-form solution
# 2. Gradient descent method

# code must output two plots, one for closed form and one for gradient descent

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def preprocess(filename: str) -> pd.DataFrame:
    """
    Open excel and format into a clean dataframe

    input: filename: str
    output: pandas.DataFrame
    """

    data = pd.read_excel(filename)
    data = data.dropna()

    return data

def lin_reg_closed_form(data: pd.DataFrame):
    """
    Implements closed form solution of linear regression.

    input: data: pd.DataFrame
    output: tuple of (weights, X, t); weights is List[bias, slope]
    """

    x = data["Weight"].values
    t = data["Horsepower"].values

    # Construct design matrix X with bias column (N x 2)
    # Each row is [1, x_i] for intercept and slope
    N = len(x)
    X = np.column_stack([np.ones(N), x])

    # Closed-form solution: w = (X^T X)^(-1) X^T t
    XtX = X.T @ X
    XtT = X.T @ t
    w = np.linalg.inv(XtX) @ XtT

    return w, X, t

def lin_reg_gradient_descent(data: pd.DataFrame, learning_rate: float, max_iters: int = 1000):
    """
    Implements gradient descent solution of linear regression.

    input: data: pd.DataFrame, learning_rate: float, max_iters: int
    output: tuple of (weights, X, t); weights is List[bias, slope]
    """

    x = data["Weight"].values
    t = data["Horsepower"].values

    # Construct design matrix X with bias column (N x 2)
    N = len(x)
    X = np.column_stack([np.ones(N), x])

    # Initialize weights to zeros
    w = np.zeros(2)

    # Gradient descent iterations
    # Update rule: w = w - α * X^T (Xw - t)
    for _ in range(max_iters):
        error = X @ w - t
        gradient = X.T @ error
        w = w - learning_rate * gradient

    return w, X, t


def plot(results: tuple, title: str, filename: str):
    """
    Plot data points and regression line, then save to file.

    input: results: tuple of (weights, X, t), title: str, filename: str
    """
    w, X, t = results

    # X[:, 1] is the Weight column (predictor)
    x_vals = X[:, 1]

    # Predictions using the regression line: y = w[0] + w[1] * x
    predictions = X @ w

    plt.figure()
    plt.scatter(x_vals, t, label="Data", alpha=0.6)
    plt.plot(x_vals, predictions, color="red", label="Regression Line")
    plt.xlabel("Weight")
    plt.ylabel("Horsepower")
    plt.title(title)
    plt.legend()
    plt.savefig(filename)
    plt.close()




if __name__ == "__main__":
    data = preprocess("proj1Dataset.xlsx")

    plot(lin_reg_closed_form(data), "linear regression (closed form)", "closed_form.png")
    plot(lin_reg_gradient_descent(data, learning_rate=1e-8), "linear regression (gradient descent)", "gradient_descent.png")
