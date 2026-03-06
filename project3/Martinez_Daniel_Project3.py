"""

1. Generate the data set D

2. Select a set of permissible values for the regularization parameter 𝜆

3. For each value of 𝜆, use the method of “linear regression with non-linear models”
   to fit Gaussian basis functions to each of the datasets. Use 𝑠 = 0.1

4. Produce the plot 

5. The test error curve is the average error for a test data set of 1000 points

"""

import numpy as np
import matplotlib.pyplot as plt

SIGMA = 0.3

np.random.seed(42)

def generate_data(n_samples: int, sigma: float=0.3):
    """
    t = sin(2*pi*x) + epsilon, where epsilon ~ N(0, sigma^2)
    """
    x = np.random.uniform(0, 1, n_samples)
    noise = np.random.normal(0, sigma, n_samples)
    t = np.sin(2 * np.pi * x) + noise

    return x, t

def gaussian_basis_func(x, mu, s=0.1):
    """
    Φ(x, mu) = exp(- (x - mu)^2 / (2*s^2))
    """
    return np.exp(-((x - mu) ** 2) / (2 * s ** 2))

def build_design_matrix(x, mus, s):
    """
    erm
    """
    N, M = len(x), len(mus)

    phi = np.ones((N, M + 1))
    for j, mu in enumerate(mus):
        phi[:, j + 1] = gaussian_basis_func(x, mu, s)

    return phi

def fit_model(phi, t, lam):
    """
    
    """
    M1 = phi.shape[1]
    I = np.eye(M1)
    w = np.linalg.solve(phi.T @ phi + lam * I, phi.T @ t)
    
    return w

def predict(phi, w):
    """

    """
    return phi @ w

def f_bar(all_preds):
    """

    """
    return np.mean(all_preds, axis=0)

def bias_squared(f_mean, x_test):
    """

    """
    h = np.sin(2 * np.pi * x_test) # true model
    return np.mean((f_mean - h) ** 2)

def variance(all_preds, f_mean):
    """

    """
    return np.mean((all_preds - f_mean) ** 2)

def test_error(all_preds, t_test):
    """

    """
    return np.mean((all_preds - t_test) ** 2)

def run_experiment(lambdas, L, N, mus, s, n_test):
    """

    """
    x_test, t_test = generate_data(n_test, sigma=SIGMA)
    phi_test = build_design_matrix(x_test, mus, s)

    bias2_list, var_list, terr_list = [], [], []

    for lam in lambdas:
        all_preds = np.zeros((L, n_test))

        for l in range(L):
            x_train, t_train = generate_data(N, sigma=SIGMA)
            phi_train = build_design_matrix(x_train, mus, s)
            w = fit_model(phi_train, t_train, lam)
            all_preds[l] = predict(phi_test, w)

        f_mean = f_bar(all_preds)

        bias2_list.append(bias_squared(f_mean, x_test))
        var_list.append(variance(all_preds, f_mean))
        terr_list.append(test_error(all_preds, t_test))

    return bias2_list, var_list, terr_list

def plot_results(lambdas, bias2_list, var_list, terr_list):
    """

    """

    ln_lam = np.log(lambdas)
    b2 = np.array(bias2_list)
    v = np.array(var_list)
    te = np.array(terr_list)

    plt.figure(figsize=(8, 5))
    plt.plot(ln_lam, b2, color='blue', label=r'$(bias)^2$')
    plt.plot(ln_lam, v, color='red', label='variance')
    plt.plot(ln_lam, b2 + v, color='magenta', label=r'$(bias)^2$ + variance')
    plt.plot(ln_lam, te, color='black', label='test error')

    plt.xlabel(r'$\ln \lambda$')
    plt.ylim(bottom=0)
    plt.legend()
    plt.tight_layout()
    plt.savefig('project3_plot.png', dpi=150)
    plt.show()

if __name__ == "__main__":
    lambdas = np.exp(np.linspace(-3, 2, 50))
    L = 100
    N = 25
    mus = np.linspace(0, 1, 25)
    s = 0.1
    n_test = 1000

    bias2_list, var_list, terr_list = run_experiment(lambdas, L, N, mus, s, n_test)
    plot_results(lambdas, bias2_list, var_list, terr_list)
