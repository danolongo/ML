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

    # Extract predictor (Weight) and target (Horsepower)
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

def lin_reg_gradient_descent(data: pd.DataFrame, learning_rate: float):
    """
    Implements gradient descent solution of linear regression.
    
    input: data: pd.DataFrame
    output: results (unsure in what format)
    """

    # must choose “Weight” as the predictor and “Horsepower” as the target variables

    pass


def plot(info: pd.DataFrame): # unsure about params for this
    """
    Plot and save results in image


    """
    pass




if __name__ == "__main__":
    data = preprocess("proj1Dataset.xlsx")

    # plot(lin_reg_closed_form(data))
    # plot(lin_reg_gradient_descent(data))
