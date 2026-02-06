# Your implementation must include:
# 1. Closed-form solution
# 2. Gradient descent method

# Your code must output two plots, one for closed form and one for gradient descent

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def preprocess(filename: str) -> pd.DataFrame:
    """
    Open excel and format into a clean dataframe
    
    input: filename: str
    output: pandas.DataFrame
    """
    # must choose “Weight” as the predictor and “Horsepower” as the target variables
    pass

def lin_reg_closed_form(data: pd.DataFrame):
    """
    
    
    input: data: pd.DataFrame
    output: results (unsure in what format)
    """
    pass

def gradient_descent_sol(data: pd.DataFrame):
    """
    
    
    input: data: pd.DataFrame
    output: results (unsure in what format)
    """
    pass


def plot(info: pd.DataFrame): # unsure about params for this
    """
    Plot and save results in image


    """
    pass




if __name__ == "__main__":
    data = pd.read_excel("proj1Dataset.xlsx")
    preprocess(data)

    # plot(lin_reg_closed_form(data))
    # plot(gradient_descent_sol(data))
