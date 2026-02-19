import numpy as np
import matplotlib.pyplot as plt

np.random.seed(175)


def generate_data(n_samples, sigma=0.3):
    """Generate noisy sinusoidal data.
    
    t = sin(2*pi*x) + epsilon, where epsilon ~ N(0, sigma^2)
    """
    x = np.random.uniform(0, 1, n_samples)
    noise = np.random.normal(0, sigma, n_samples)
    t = np.sin(2 * np.pi * x) + noise

    return x, t


def build_design_matrix(x, degree):
    """Build the design matrix Phi for polynomial regression.
    
    Phi = [1, x, x^2, ..., x^M] where M = degree
    """
    n = len(x)
    phi = np.zeros((n, degree + 1))

    for j in range(degree + 1):
        phi[:, j] = x ** j

    return phi


def fit_polynomial(phi, t):
    """Solve for weights using the normal equations.
    
    w* = (Phi^T Phi)^(-1) Phi^T t
    """
    phi_t_phi = phi.T @ phi
    phi_t_t = phi.T @ t
    w = np.linalg.solve(phi_t_phi, phi_t_t)

    return w


def compute_rms_error(phi, w, t):
    """Compute root mean square error.
    
    E_RMS = sqrt(1/N * sum((y - t)^2))
    """
    y = phi @ w
    n = len(t)
    mse = np.sum((y - t) ** 2) / n

    return np.sqrt(mse)


def run_experiment(n_train, n_test=100, max_degree=9, sigma=0.3):
    """Run polynomial regression experiment for degrees 0 to max_degree."""
    # Generate data
    x_train, t_train = generate_data(n_train, sigma)
    x_test, t_test = generate_data(n_test, sigma)
    
    degrees = list(range(max_degree + 1))
    train_errors = []
    test_errors = []
    
    for m in degrees:
        # Build design matrices
        phi_train = build_design_matrix(x_train, m)
        phi_test = build_design_matrix(x_test, m)
        
        # Fit model
        w = fit_polynomial(phi_train, t_train)
        
        # Compute errors
        train_err = compute_rms_error(phi_train, w, t_train)
        test_err = compute_rms_error(phi_test, w, t_test)
        
        train_errors.append(train_err)
        test_errors.append(test_err)
        
        # print(f"M={m}: Train E_RMS={train_err:.4f}, Test E_RMS={test_err:.4f}")
    
    return degrees, train_errors, test_errors


def plot_errors(degrees, train_errors, test_errors, n_train, filename):
    """Create E_RMS vs M plot."""
    plt.figure(figsize=(8, 6))
    
    plt.plot(degrees, train_errors, 'b-o', label='Training', markersize=8)
    plt.plot(degrees, test_errors, 'r-o', label='Test', markersize=8)
    
    plt.xlabel('M', fontsize=12)
    plt.ylabel('$E_{RMS}$', fontsize=12)
    plt.title(f'N_train={n_train}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(degrees)
    plt.ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Plot saved to {filename}")


if __name__ == "__main__":
    # Experiment 1: N_train = 10
    degrees, train_err_10, test_err_10 = run_experiment(n_train=10)
    plot_errors(degrees, train_err_10, test_err_10, n_train=10, 
                filename="n_train10.png")
    
    # Experiment 2: N_train = 100
    degrees, train_err_100, test_err_100 = run_experiment(n_train=100)
    plot_errors(degrees, train_err_100, test_err_100, n_train=100,
                filename="n_train100.png")
