import pymc as pm
import numpy as np
import arviz as az
import pytensor.tensor as pt
from baseball import load_data
# 🔹 Evaluate predictions
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression

def bayesian_nb_model(X_train, X_test, y_train, y_test, num_samples=2000, tune=1000):
    """
    Bayesian Negative Binomial model for count prediction using PyMC.
    
    Parameters:
        X_train (np.ndarray): Training feature matrix (shape: [n_samples, n_features])
        X_test (np.ndarray): Test feature matrix (shape: [n_samples, n_features])
        y_train (np.ndarray): Training target counts
        y_test (np.ndarray): Test target counts (only for evaluation, not used in training)
        num_samples (int): Number of posterior samples
        tune (int): Number of tuning steps

    Returns:
        trace (arviz.InferenceData): MCMC samples from posterior
        posterior_preds (np.ndarray): Predicted posterior means for X_test
    """
    n_features = X_train.shape[1]  # Get number of features


    # Train your linear model on X_train
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)

    # Get trained coefficients & intercept
    pretrained_coefs = linear_model.coef_
    pretrained_intercept = linear_model.intercept_
    print(pretrained_coefs)
    with pm.Model():
        # Priors
        coefs = pm.Normal("coefs", mu=pretrained_coefs, sigma=0.5, shape=X_train.shape[1])
        intercept = pm.Normal("intercept", mu=pretrained_intercept, sigma=1.0)
        alpha = pm.Gamma("alpha", alpha=1, beta=.5)
        # Expected value of y (mu) using log-link
        mu_train = pm.math.exp(intercept + pm.math.dot(X_train, coefs))
        # Likelihood function: Negative Binomial
        pm.NegativeBinomial("obs", mu=mu_train, alpha=alpha, observed=y_train)

        # Inference with NumPyro backend for speed
        trace = pm.sample(2000, tune=1000, target_accept=0.9, return_inferencedata=True, cores=4, chains=4, nuts_sampler="numpyro")

        # Generate posterior predictive samples for test set
        mu_test = pm.math.exp(intercept + pm.math.dot(X_test, coefs))  # Predict on X_test
        posterior_preds = pm.draw(pm.NegativeBinomial.dist(mu=mu_test, alpha=alpha), draws=1000).mean(axis=0)

    return trace, posterior_preds


##MAiN
X_train,X_test,y_train,y_test,mid_season_wins = load_data(split=.01)
# Assume X_train, X_test, y_train, y_test are preprocessed numpy arrays
trace, y_pred = bayesian_nb_model(X_train, X_test, y_train, y_test)

# 🔹 Posterior analysis
az.summary(trace)
az.plot_posterior(trace)

# 🔹 Evaluate predictions

print("MAE on Test Set:", mean_absolute_error(y_test, y_pred))