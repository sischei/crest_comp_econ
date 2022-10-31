"""
Script to explore Bayesian active learning (BAL) and Gaussian Process regression

"""

import numpy as np
import matplotlib.pyplot as plt

from scipy.linalg import cholesky, cho_solve, solve_triangular
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel as C

def compute_moments(x, gp):
    """Calculate posterior predictive mean and variance of x* conditional on the training set."""

    e_gp, K = gp.predict(x, return_cov=True)
    k_xnew = K[-1, -1]
    k_xnew_xtrain = K[-1, :-1]
    K_xtrain = K[:-1, :-1]
    # TODO(lpupp) best way to invert?
    var_gp = k_xnew - np.dot(np.dot(k_xnew_xtrain, np.linalg.inv(K_xtrain)), k_xnew_xtrain.T)

    return e_gp[-1], var_gp


def bal_utility(e_gp, var_gp, rho=0.5, beta=0.5):
    """Calculate utility for Bayesian active learning."""
    utility = rho * e_gp + (beta / 2.0) * np.log(var_gp)
    
    return utility


def f(x):
    """Calculate f(x) = x^sin(x)."""
    return x ** np.sin(x)


def plot_fn(X, y, gp, fn):
    """Plot the function, the prediction and the 95% confidence interval based on the MSE."""
    x = np.atleast_2d(np.linspace(0, 5, 1000)).T
    y_pred, sigma = gp.predict(x, return_std=True)

    plt.figure()
    plt.plot(x, f(x), 'r:', label=r'$f(x) = x^{\sin(x)}$')
    plt.plot(X, y, 'r.', markersize=10, label=u'Observations')
    plt.plot(x, y_pred, 'b-', label=u'Prediction')
    plt.fill(np.concatenate([x, x[::-1]]),
             np.concatenate([y_pred - 1.9600 * sigma,
                            (y_pred + 1.9600 * sigma)[::-1]]),
             alpha=.5, fc='b', ec='None', label='95% CI')
    plt.xlabel('$x$')
    plt.ylabel(r'$f(x) = x^{\sin(x)}$')
    plt.ylim(0.0, 2.3)
    plt.legend(loc='upper right')
    # plt.savefig('./output/bal_test/' + fn)
    plt.show()
    plt.close()


def main_(n_iters):
    X = np.array([[0.1], [0.5], [4.3], [4.5], [4.9]])
    y = f(X).ravel()

    x_candidates = np.arange(0, 5, 0.05)
    x_candidates = x_candidates[1:]

    # kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
    # kernel = RBF(10, (1e-2, 1e2))
    kernel = 1.0 * Matern(length_scale=1.0, length_scale_bounds=(1e-1, 10.0), nu=1.5)
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=1e-6)
    gp.fit(X, y)

    plot_fn(X, y, gp, 'x0.png')

    for j in range(n_iters):
        U_bal = np.zeros_like(x_candidates)

        for i, _x in enumerate(x_candidates):
            x_tmp = np.concatenate([X, np.atleast_2d(_x)])
            e_gp, var_gp = compute_moments(x_tmp, gp)
            U_bal[i] = bal_utility(e_gp, var_gp)

        i_new = np.nanargmax(U_bal)
        x_new = x_candidates[i_new]
        x_candidates = np.delete(x_candidates, i_new, 0)
        
        print('Augmenting {}th candidate {} with U_BAL = {}'.format(i_new, x_new, U_bal[i_new]))

        X = np.concatenate([X, np.atleast_2d(x_new)])
        y = f(X).ravel()

        plot_fn(X, y, gp, 'x{}.png'.format(j + 1))
    return X
    

def main(n_iters):
    X = np.array([[0.1], [0.5], [4.3], [4.5], [4.9]])
    y = f(X).ravel()

    x_candidates = np.arange(0, 5, 0.05)
    x_candidates = x_candidates[1:]

    # kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
    # kernel = RBF(10, (1e-2, 1e2))
    kernel = 1.0 * Matern(length_scale=1.0, length_scale_bounds=(1e-1, 10.0), nu=1.5)
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=1e-6)
    gp.fit(X, y)

    plot_fn(X, y, gp, 'x0.png')

    for j in range(n_iters):
        U_bal = np.zeros_like(x_candidates)

        for i, _x in enumerate(x_candidates):
            x_tmp = np.concatenate([X, np.atleast_2d(_x)])
            e_gp, var_gp = compute_moments(x_tmp, gp)
            U_bal[i] = bal_utility(e_gp, var_gp)

        i_new = np.nanargmax(U_bal)
        x_new = x_candidates[i_new]
        x_candidates = np.delete(x_candidates, i_new, 0)
        
        print('Augmenting {}th candidate {} with U_BAL = {}'.format(i_new, x_new, U_bal[i_new]))

        X = np.concatenate([X, np.atleast_2d(x_new)])
        y = f(X).ravel()

        # Update data
        gp.X_train_ = np.copy(X)
        gp.y_train_ = np.copy(f(X).ravel())
        K = gp.kernel_(gp.X_train_)
        K[np.diag_indices_from(K)] += gp.alpha
        gp.L_ = cholesky(K, lower=True)
        gp.alpha_ = cho_solve((gp.L_, True), gp.y_train_)
        L_inv = solve_triangular(gp.L_.T, np.eye(gp.L_.shape[0]))
        gp._K_inv = L_inv.dot(L_inv.T)

        plot_fn(X, y, gp, 'x{}.png'.format(j + 1))
    return X


if __name__ == "__main__":
    X = main(10)
