# Version 0.2
# Problems in v0.1: 
# - Homogenous effect not distinguishable from alpha at baseline
# - Poly3 suffering from multicolinearity
# Solution: Drop both specifications, as they are not important.

# NEW:
# Make treatment a beta-distribution

# %%
import numpyro
from numpyro import distributions as dist
import jax.numpy as jnp
import jax.random as random
import numpy as np
from numpyro.infer import Predictive
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from numpyro.infer import MCMC, NUTS, Predictive
import time

# %%
def model_POF(Z, X, T, Y=None, rng_key=None, funcForm="linear"):
    """
    Continuous potential outcome model with various functional forms for the
    dose-response function and with village and project fixed effects.

    Args:
        Z (jnp.array): Explanatory variables (standardized).
        X (jnp.array): Explanatory variables (standardized).
        T (jnp.array, optional): Treatment dosage. Defaults to None.
        Y (jnp.array, optional): Observed outcome. Defaults to None.
        funcForm (str, optional): A string specifying the functional form.
                                  Options: "homogenous", "linear", "poly2", "poly3",
                                  "exponential", "sigmoid". Defaults to "linear".
    """
    
    N = X.shape[0]

    # Handle optional clustering variables
    if rng_key is None:
      rng_key = random.PRNGKey(0)
    
    # Coefficients for the baseline outcome (Y0)
    alpha = numpyro.sample("alpha", dist.Normal(0., 1).expand([X.shape[1]]))
    
    # Noise parameter for the outcome
    sigma_Y = numpyro.sample("sigma_Y", dist.Exponential(1))

    # Y0 represents the potential outcome when T=0.
    Y0 = X @ alpha 
    
    # Determine the treatment effect (tau) based on the funcForm string
    # For non-homogenous cases, tau is a function of T
    if funcForm == "linear":
        #b_intercept = numpyro.sample("b_intercept", dist.Normal(0., 1))
        beta = numpyro.sample("beta", dist.Normal(0., 1))
        tau = T * beta

    elif funcForm == "poly2":
        #b_intercept = numpyro.sample("b_intercept", dist.Normal(0., 1))
        beta = numpyro.sample("beta", dist.Normal(0., 1))
        betaSq = numpyro.sample("betaSq", dist.Normal(0., 1))
        tau = T * beta + (T**2) * betaSq

    elif funcForm == "exponential":
        # Parameters for the exponential function: f(x) = a * (1 - exp(-b*x)) + c
        a = numpyro.sample("a", dist.Normal(1, 1))
        b = numpyro.sample("b", dist.Normal(2, 1))
        c = numpyro.sample("c", dist.Normal(0, 1))

        tau = a * (1 - jnp.exp(-b * T)) + c

    elif funcForm == "sigmoid":
        a = numpyro.sample("a", dist.Normal(0, 1))
        b = numpyro.sample("b", dist.Exponential(1))
        c = numpyro.sample("c", dist.Exponential(0.02))
        d = numpyro.sample("d", dist.Uniform(0., 1.))
        tau = a + b * (1 / (1 + jnp.exp(-c * (T - d))))

    # Version 0.2: Model T with a dosage propensity based on Z
    #numpyro.sample("T", dist.Uniform(0, 1), obs=T)
    # Coefficients for the beta distribution
    beta_1 = numpyro.sample("beta_1", dist.Exponential(1))
    beta_2 = numpyro.sample("beta_2", dist.Exponential(1))

    numpyro.sample("T", dist.Beta(beta_1, beta_2), obs=T)

    # The observed outcome Y is modeled with a dose-response function
    Y_mean = Y0 + T * tau
    numpyro.sample("Y", dist.Normal(Y_mean, sigma_Y), obs=Y)
    
    # Collect Y0 and tau as deterministic for later use
    numpyro.deterministic("Y0", Y0)
    numpyro.deterministic("tau", tau)

# %%
# Modify T to be a beta-distributed variable with mean determined by a logit of Z
def logit(x):
    return jnp.log(x / (1 - x))

def inv_logit(x):
    return 1 / (1 + jnp.exp(-x))

def generate_T_from_Z(rng_key, Z, a=2.0, b=2.0):
    # Compute mean of beta using logit of Z[:, 1] (or any column of Z)
    z_raw = Z[:, 1] if Z.shape[1] > 1 else Z[:, 0]
    # Scale z_raw to reasonable range for logit
    z_scaled = (z_raw - z_raw.mean()) / (z_raw.std() + 1e-8)
    # Map to [0.1, 0.9] via inv_logit and rescale
    mean_beta = inv_logit(z_scaled) * 0.8 + 0.1
    # Convert mean to alpha/beta parameters for beta distribution
    # Fix concentration parameter for simplicity
    concentration = 4.0
    alpha = mean_beta * concentration
    beta = (1 - mean_beta) * concentration
    T = random.beta(rng_key, a=alpha, b=beta)
    return T

# %%
def data_generating(rng_key,
                    funcForm='linear',
                    N=10000,
                    K=5,
                    X=None,
                    showplot=False):
    """
    Helper function to generate data using a continuous treatment model.
    Creates diagnostic plots to illustrate the DGP.

    Args:
        rng_key (jax.random.PRNGKey): A JAX random key.
        funcForm (str, optional): A string specifying the functional form.
                                  Options: "homogenous", "linear", "poly2", "poly3",
                                  "exponential", "sigmoid". Defaults to "linear".
        N (int, optional): Number of samples. Defaults to 10000.
        K (int, optional): Number of explanatory variables (ignored if X is provided). Defaults to 5.
        X (jnp.array, optional): Predefined explanatory variables. Defaults to None.
        showplot (bool, optional): If True, shows diagnostic plots. Defaults to False.

    Raises:
        ValueError: if funcForm is not recognized.
    """
    # Replace T generation in data_generating
    if X is None:
        rng_key, key_split_X, key_split_Z, key_split_T = random.split(rng_key, 4)
        X = random.normal(key_split_X, shape=(N, K - 1))
        X = jnp.hstack([jnp.ones((N, 1)), X])
        Z = random.normal(key_split_Z, shape=(N, K - 1))
        Z = jnp.hstack([jnp.ones((N, 1)), Z])
        T = random.beta(key_split_T, a=3, b=5, shape=(N, 1))
    else:
        Z = X
        rng_key, key_split_T = random.split(rng_key)
        T = random.beta(key_split_T, a=3, b=5, shape=(N, 1))

    model_kwargs = {'Z': Z, 'X': X, 'T': T, 'funcForm': funcForm}

    # Run the DGP once to get values for latent variables
    rng_key, rng_key_ = random.split(rng_key)
    lat_predictive = Predictive(model_POF, num_samples=1)
    lat_samples = lat_predictive(rng_key_, **model_kwargs)

    coefTrue = {}

    if funcForm == 'linear':
        #coefTrue['b_intercept'] = jnp.array(0.1, dtype='float32')
        coefTrue['beta'] = jnp.array(1, dtype='float32')
    elif funcForm == 'poly2':
        #coefTrue['b_intercept'] = jnp.array(0., dtype='float32')
        coefTrue['beta'] = jnp.array(4, dtype='float32')
        coefTrue['betaSq'] = jnp.array(-5, dtype='float32')
    elif funcForm == 'sigmoid':
        # True coefficients for the sigmoid function 
        # f(x) = a + b * (1 / (1 + exp(-c * (x - d))))
        coefTrue['a'] = jnp.array(0.1, dtype='float32')
        coefTrue['b'] = jnp.array(0.6, dtype='float32')
        coefTrue['c'] = jnp.array(50., dtype='float32')
        coefTrue['d'] = jnp.array(0.4, dtype='float32')
    elif funcForm == 'exponential':
        # True coefficients for the exponential function
        # f(x)=a\left(1-e^{-bx}\right)+c
        coefTrue['a'] = jnp.array(0.6, dtype='float32')
        coefTrue['b'] = jnp.array(3.0, dtype='float32')
        coefTrue['c'] = jnp.array(0.1, dtype='float32')
    else:
        raise ValueError("Invalid funcForm specified")


    condition_model = numpyro.handlers.condition(model_POF, data=coefTrue)
    conditioned_predictive = Predictive(condition_model, num_samples=1)
    prior_samples = conditioned_predictive(rng_key_, **model_kwargs)

    Y_unscaled = prior_samples['Y'].squeeze()
    Y0 = prior_samples['Y0'].squeeze()
    tau_true = prior_samples['tau'].squeeze()

    Y_mean = Y_unscaled.mean(axis=0)
    Y_std = Y_unscaled.std(axis=0)
    Y = (Y_unscaled - Y_mean) / Y_std

    print(f'Mean(Y)={jnp.mean(Y):.4f}; std(Y)={jnp.std(Y):.4f}')
    print(f'Mean(T)={jnp.mean(T):.4f}; std(T)={jnp.std(T):.4f}')
        
    return Y, Y_unscaled, Y_mean, Y_std, T, Z, X, Y0, tau_true, conditioned_predictive, model_kwargs, coefTrue

    

# %%
# inspect T histogram of test data
test = data_generating(
    rng_key=random.PRNGKey(0),
    funcForm='linear',
    N=10000,
    K=5
)

# Inspect T histogram of test data
plt.figure(figsize=(8, 6))
sns.histplot(test[4], bins=30, kde=True)
plt.title('Histogram of Treatment Variable (T)')
plt.xlabel('Treatment (T)')
plt.ylabel('Frequency')
plt.grid()
plt.show()


# %%

# --- MCMC Simulation Parameters ---
N = 1000  # Number of samples
K = 5      # Number of explanatory variables
n_plot = 300

# Define the MCMC kernel
kernel = NUTS(model_POF)

# Set up MCMC
mcmc = MCMC(
    kernel,
    num_warmup=100,
    num_samples=200,
    num_chains=1,
    progress_bar=True
)

# Define DGP and Inference Model types
dgp_types = ['linear', 'poly2', 'sigmoid', 'exponential']
inference_model_configs = [
    ('linear', 'ii) Linear'),
    ('poly2', 'iii) Squared'),
    ('sigmoid', 'v) Sigmoid'),
    ('exponential', 'vi) Exponential')
]

# Initialize a single random key for reproducibility across the entire figure
rng_key = random.PRNGKey(0)

# Create one large figure with subplots
fig, axes = plt.subplots(len(dgp_types), len(inference_model_configs), figsize=(20, 16), tight_layout=True)

# Add a super title for the entire figure
fig.suptitle('Dose-Response Function by DGP (Row) and Inference Model (Column)', fontsize=24, y=1.02)

# empty list to store posterior summaries
posterior_summaries = []

# Loop through each Data Generating Process (DGP) for rows
for i, dgp_funcForm in enumerate(dgp_types):
    print(f'\n\nGenerating data for funcForm: {dgp_funcForm}')
    
    # Split rng_key for data generation
    rng_key, rng_key_data_gen = random.split(rng_key)
    (Y, Y_unscaled, Y_mean, Y_std, T, Z, X, Y0, tau_true, 
     conditioned_predictive, model_kwargs, coefTrue) = data_generating(
        rng_key=rng_key_data_gen,
        funcForm=dgp_funcForm,
        N=N,
        K=K
    )
    
    # Loop through each Inference Model for columns
    for j, (inf_funcForm, figTitle) in enumerate(inference_model_configs):
        ax = axes[i, j]
        print(f'\n\nEstimating model: {inf_funcForm} for DGP: {dgp_funcForm}')
        
        # Define the data to pass to the MCMC model
        # The new model takes a single funcForm string
        datXY = {'Z': Z, 'X': X, 'Y': Y_unscaled, 'T': T, 'funcForm': inf_funcForm}
        
        # Run MCMC
        start = time.time()
        rng_key, rng_key_mcmc = random.split(rng_key)
        mcmc.run(rng_key_mcmc, **datXY)
        print("\nMCMC elapsed time:", time.time() - start)
        mcmc_samples = mcmc.get_samples()
        
        # make dataframe with posterior summary
        posterior_summary_tmp = pd.DataFrame({
            'DGP_Form': dgp_funcForm,
            'Inference_Form': inf_funcForm,
            'Parameter': list(mcmc_samples.keys()),
            'Mean': [jnp.mean(mcmc_samples[param]) for param in mcmc_samples.keys()],
            'StdDev': [jnp.std(mcmc_samples[param]) for param in mcmc_samples.keys()],
            'CI95_Lower': [jnp.percentile(mcmc_samples[param], 2.5) for param in mcmc_samples.keys()],
            'CI95_Upper': [jnp.percentile(mcmc_samples[param], 97.5) for param in mcmc_samples.keys()]
        })
        posterior_summaries.append(posterior_summary_tmp)
        
        # Plot results
        k = 1
        x_percentile = np.percentile(T, q=[0.1, 99])
        x_range = np.linspace(x_percentile[0], x_percentile[1], 100)
        x_mean = jnp.mean(X, axis=0)
        x_plot = jnp.repeat(x_mean.reshape(1, -1), 100, axis=0)
        x_plot = x_plot.at[:, k].set(x_range)
        
        # Get posterior predictions for tau
        rng_key, rng_key_post_predict_plot = random.split(rng_key)
        predictivePosterior = Predictive(model_POF, posterior_samples=mcmc_samples)
        post_predict = predictivePosterior(rng_key_post_predict_plot, Z=x_plot, X=x_plot, T=x_range, funcForm=inf_funcForm)
        
        # Plot 300 random draws from the posterior
        tau_post_samples = post_predict['tau']
        
        # instead of all samples, only plot random 300
        for idx in range(1, n_plot):
            ax.plot(x_range, tau_post_samples[idx], color='k', alpha=0.1)
        
        # Add "true" effect in red
        if dgp_funcForm == 'homogenous':
            ax.axhline(y=tau_true[0], color='r', alpha=1, label='True Effect')
        else:
            ax.plot(T, tau_true, 'o', color='r', alpha=1, label='True Effect')

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_xlabel(f'Treatment (T)', fontsize=14)
        ax.set_ylabel(r'$\tau$ (Dose-Response)', fontsize=14)
        
        if i == 0:
            ax.set_title(figTitle, fontsize=16)
        
        if j == 0:
            ax.text(-0.3, 0.5, f'DGP: {dgp_funcForm.title()}', 
                    transform=ax.transAxes, rotation=90, va='center', ha='right', fontsize=16)
        
        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            label.set_fontsize(12)
        
        ax.set_xlim([x_percentile[0], x_percentile[1]])
        # set y axis limit -0.5 to 1.5
        ax.set_ylim([-0.5, 1.5])

        if dgp_funcForm == inf_funcForm:
            ax.set_facecolor('#f0f0f0')

# Save the combined figure
plt.savefig(f'POF_MCMC_v02.png', dpi=300)
plt.show()



# %%
# show posterior summaries
# Concatenate all posterior summaries into a single DataFrame
all_posterior_summaries = pd.concat(posterior_summaries)
print(all_posterior_summaries)

#all_posterior_summaries.to_csv('post_sum_v02.csv')
