# Version 0.4
# Problem v0.3
# - Treatment purely driven by Z, no random REDD component

# Solution: model treatment as bivariate beta distribution
# with two alpha-values, randomly sampled from bernoulli
# this reflects random REDD-placement, further increasing
# treatment intensity in addition to non-REDD interventions.

# %%
import numpyro
from numpyro import distributions as dist
import jax.numpy as jnp
import jax.random as random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from numpyro.infer import MCMC, NUTS, Predictive

rng_key = random.PRNGKey(0)
rng_key, rng_key_ = random.split(rng_key)

# %% Model
def model_POF(Z, X, T, D, Y=None, funcForm="linear"):
    """
    Continuous potential outcome model with a flexible dose-response and an observed
    treatment placement D that shifts the Beta concentration of T via Z.

    Args:
        Z (N, pz): covariates for treatment assignment
        X (N, px): covariates for baseline outcome
        T (N,):    treatment dosage in [0,1]
        D (N,):    binary placement indicator (observed)
        Y (N,):    outcome (unscaled)
        funcForm:  'linear' | 'poly2' | 'exponential' | 'sigmoid'
    """
    N = X.shape[0]

    # --- Baseline outcome coefficients (Y0 = X @ alpha)
    alpha = numpyro.sample("alpha", dist.Normal(0., 1.).expand([X.shape[1]]))
    Y0 = jnp.dot(X, alpha)  # (N,)

    # --- Dose-response tau(T)
    if funcForm == "linear":
        beta = numpyro.sample("beta", dist.Normal(0., 1.))
        tau = T * beta
    elif funcForm == "poly2":
        a = numpyro.sample("a", dist.Normal(0., 2.))
        b = numpyro.sample("b", dist.Normal(0., 1.))
        tau = a * T * (1 - T) + b * T
    elif funcForm == "exponential":
        a = numpyro.sample("a", dist.Normal(1., 1.))
        b = numpyro.sample("b", dist.Normal(2., 1.))  # keep sign free; exponent handles shape
        c = numpyro.sample("c", dist.Normal(0., 1.))
        tau = a * (1 - jnp.exp(-b * T))
    elif funcForm == "sigmoid":
        #a = numpyro.sample("a", dist.Normal(0., 1.))
        b = numpyro.sample("b", dist.Normal(0., 1.))
        c = numpyro.sample("c", dist.Exponential(0.02))
        d = numpyro.sample("d", dist.Uniform(0., 1.))
        tau = b * (1.0 / (1.0 + jnp.exp(-c * (T - d))))
    else:
        numpyro.deterministic("tau", jnp.zeros_like(T))
        raise ValueError("Invalid funcForm specified")

    # --- Treatment dosage model: T | Z, D ~ Beta(alpha(Z,D), beta_param)
    gamma = numpyro.sample("gamma", dist.Normal(0., 1.).expand([Z.shape[1]]))
    omega = numpyro.sample("omega", dist.Exponential(1.))
    beta_param = numpyro.sample("beta_param", dist.Exponential(1.))

    logits = jnp.dot(Z, gamma)               # (N,)
    base_alpha = jnp.exp(logits)             # >0
    alpha_param = base_alpha + omega * D     # shift when D==1

    sigma_Y = numpyro.sample("sigma_Y", dist.Exponential(1.))
        
    # Vectorize obs in a plate
    #with numpyro.plate("N", N):
    # D is observed placement, not latent
    numpyro.sample("D_obs", dist.Bernoulli(0.5), obs=D)

    # Treatment dosage is observed
    numpyro.sample("T_obs", dist.Beta(alpha_param, beta_param), obs=T)

    # Outcome
    Y_mean = Y0 + tau
    numpyro.sample("Y_obs", dist.Normal(Y_mean, sigma_Y), obs=Y)

    # Useful deterministics for diagnostics
    numpyro.deterministic("Y0", Y0)
    numpyro.deterministic("tau", tau)
    numpyro.deterministic("alpha_param", alpha_param)

# %% Data generating process
def data_generating(rng_key, funcForm='linear', N=10000, K=5, X=None, showplot=False):
    """
    Simulate (Y, T, D, Z, X) and return ground truth parameters for recovery tests.
    """
    rng_key, rng_key_, kX, kZ, kT, kD = random.split(rng_key, 6)

    if X is None:
        X = random.normal(kX, shape=(N, K - 1))
        X = jnp.hstack([jnp.ones((N, 1)), X])      # intercept
        Z = random.normal(kZ, shape=(N, K - 1))
        Z = jnp.hstack([jnp.ones((N, 1)), Z])      # intercept
    else:
        Z = X

    # Placement
    D = dist.Bernoulli(0.5).sample(key=kD, sample_shape=(N,))

    # Treatment model truths
    gamma_true = jnp.array([1., -0.5, 0.3, -0.8, 0.2], dtype=jnp.float32)
    omega_true = jnp.array(2., dtype=jnp.float32)
    beta_param_true = jnp.array(4., dtype=jnp.float32)

    logits = Z @ gamma_true
    base_alpha = jnp.exp(logits)
    alpha_param = base_alpha + omega_true * D

    T = dist.Beta(alpha_param, beta_param_true).sample(key=kT)

    # Choose tau truths
    coefTrue = {
        "gamma": gamma_true,
        "omega": omega_true,
        "beta_param": beta_param_true,
        # alpha for baseline left random (keeps problem realistic)
    }
    if funcForm == 'linear':
        coefTrue['beta'] = jnp.array(0.8, dtype=jnp.float32)
    elif funcForm == 'poly2':
        coefTrue['a']   = jnp.array(1., dtype=jnp.float32)
        coefTrue['b'] = jnp.array(0.3, dtype=jnp.float32)
    elif funcForm == 'sigmoid':
        #coefTrue['a'] = jnp.array(0.1, dtype=jnp.float32)
        coefTrue['b'] = jnp.array(0.6, dtype=jnp.float32)
        coefTrue['c'] = jnp.array(50., dtype=jnp.float32)
        coefTrue['d'] = jnp.array(0.4, dtype=jnp.float32)
    elif funcForm == 'exponential':
        coefTrue['a'] = jnp.array(0.6, dtype=jnp.float32)
        coefTrue['b'] = jnp.array(3.0, dtype=jnp.float32)
        #coefTrue['c'] = jnp.array(0.1, dtype=jnp.float32)
    else:
        raise ValueError("Invalid funcForm specified")

    # Get one draw of Y under the true parameters (leaving alpha random)
    model_kwargs = {'Z': Z, 'X': X, 'T': T, 'D': D, 'funcForm': funcForm}
    condition_model = numpyro.handlers.condition(model_POF, data=coefTrue)
    predictive = Predictive(condition_model, num_samples=1)
    prior_samples = predictive(rng_key_, **model_kwargs)

    Y_unscaled = prior_samples['Y_obs'].squeeze()
    Y0         = prior_samples['Y0'].squeeze()
    tau_true   = prior_samples['tau'].squeeze()

    # If you *want* to standardize Y, standardize inside the model instead.
    # Here we use raw scale for consistency with the likelihood:
    Y = Y_unscaled

    print(f'Mean(Y)={jnp.mean(Y):.4f}; std(Y)={jnp.std(Y):.4f}')
    print(f'Mean(T)={jnp.mean(T):.4f}; std(T)={jnp.std(T):.4f}')

    return Y, Y_unscaled, T, D, Z, X, Y0, tau_true, predictive, model_kwargs, coefTrue


# %%
# show real data

# load csv file from \data
try:
    df1 = pd.read_csv('C:/Users/DaSchulz/OneDrive - European Forest Institute/Dokumente/research/pp_agecon_erae/data/hh_step_short_BRA_ALLINT.csv')
except FileNotFoundError:
    print("Error: The CSV file was not found. Please check and correct the file path.")
    exit()

# **NEW:** Drop rows only when missing values are in the variables of interest.
# This prevents dropping rows with missing data in columns you aren't using.
relevant_columns = [
    "DY_forest_share",
    "Marital_status", "Hh_cooking_tech", "Hh_floor2", "Hh_walls2", "Hh_roof2",
    "Hh_water", "Hh_toilet", "Hh_electric", "Hh_cooking_fuel",
    "int_nonREDD_ongoing_sum", "int_nonREDD_ended_sum",
    "Village_Type", "int_cat_tot_involALL"
]
print("Checking for and removing any rows with missing data in variables of interest...")
initial_rows = len(df1)
df1.dropna(subset=relevant_columns, inplace=True)
print(f"Removed {initial_rows - len(df1)} rows with missing values in relevant columns.")

N = len(df1)
# subset variabels of interest
# Y = DY_forest_share
# X = "Marital_status","Hh_cooking_tech","Hh_floor2","Hh_walls2","Hh_roof2","Hh_water","Hh_toilet","Hh_electric","Hh_cooking_fuel"
# Z = int_nonREDD_ongoing_sum, int_nonREDD_ended_sum
# D = Village_Type
# T = int_cat_tot_involALL

# make variables
X = df1[["Marital_status","Hh_cooking_tech","Hh_floor2","Hh_walls2","Hh_roof2","Hh_water","Hh_toilet","Hh_electric","Hh_cooking_fuel"]]
rng_key, key_split_X, key_split_Z, key_split_T, key_split_D = random.split(random.PRNGKey(0), 5)
#X = random.normal(key_split_X, shape=(N, 5))
#X = jnp.hstack([jnp.ones((N, 1)), X])

Z = df1[["int_nonREDD_ongoing_sum","int_nonREDD_ended_sum"]]
D = df1["Village_Type"]
T = df1["int_cat_tot_involALL"]
Y = df1["DY_forest_share"]

# standardize predictor variables X and Z
X = (X - X.mean())/X.std()
Z = (Z - Z.mean())/Z.std()

# standardize T between 0.01 and 0.99
T = (T - T.min()) / (T.max() - T.min())
T = T * 0.98 + 0.01

# add constant of same length to X and Z
X['X1'] = 1.0
Z['Z1'] = 1.0

# %%
# inspect D, make table of values
print("Village Type (D) counts:")
print(D.value_counts())

# %%
# plot histogram of T colored by D
plt.figure(figsize=(10, 6))
for d in D.unique():
    plt.hist(T[D == d], bins=30, alpha=0.5, label=str(d))
plt.xlabel("Treatment Intensity (T)")
plt.ylabel("Frequency")
plt.title("Histogram of Treatment Intensity (T) by Village Type (D)")
plt.legend(title="Village Type (D)")
plt.show()

# %%
# plot T vs Y color by D
plt.figure(figsize=(10, 6))
for d in D.unique():
    plt.scatter(T[D == d], Y[D == d], alpha=0.5, label=str(d))
plt.axhline(0, color='k', linestyle='--', linewidth=1)
plt.xlabel("Treatment Intensity (T)")
plt.ylabel("Potential Outcome (Y)")
plt.title("Scatter Plot of Treatment Intensity (T) vs Potential Outcome (Y)")
plt.legend(title="Village Type (D)")
plt.show()

# %% initialize kernel
funcForm='linear'
kernel = NUTS(model_POF)
mcmc = MCMC(
    kernel,
    num_warmup=500,
    num_samples=1000,
    num_chains=2,
    progress_bar=True
)

# Initialize a single random key for reproducibility.
rng_key = random.PRNGKey(0)

# Run MCMC on the real data for the linear model.
# The `funcForm` is passed directly to the model through the kernel.
# We are using the unscaled `Y` as per your previous code's intent.
print("\nRunning MCMC for the linear model on the real data...")
mcmc.run(rng_key, Z=Z.values, X=X.values, T=T.values, D=D.values, Y=Y.values)
print("\nMCMC run complete.")

# %%
import arviz as az
# Get and print the posterior samples.
mcmc_samples = mcmc.get_samples()
                
ppc = Predictive(model_POF, mcmc_samples)(
    rng_key_, Z=Z.values, X=X.values, T=T.values, D=D.values
)
idata = az.from_numpyro(
    mcmc,
    posterior_predictive=ppc,
    coords={"obs_id": np.arange(len(Y))},
    dims={"Y_obs": ["obs_id"]}
)

# Posterior summary
posterior_summary_tmp = az.summary(idata, round_to=4).reset_index()
# %%
# print summary, exclude Y0, tau
posterior_summary = posterior_summary_tmp[
    ~posterior_summary_tmp["index"].str.contains(r"^(tau|Y0|alpha|gamma)", regex=True)
]

print(posterior_summary)

# %%
import arviz as az
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import jax.numpy as jnp
from numpyro.infer import MCMC, NUTS, Predictive
from jax import random
import time
# now use real data and loop over all DGPs. collect model fit PSIS

# --- Parameters ---
n_plot = 200
rng_key = random.PRNGKey(0)

inference_model_configs = [
    ('linear', 'ii) Linear'),
    ('poly2', 'iii) Squared'),
    ('sigmoid', 'v) Sigmoid'),
    ('exponential', 'vi) Exponential')
]

kernel = NUTS(model_POF)
mcmc = MCMC(
    kernel,
    num_warmup=100,
    num_samples=200,
    num_chains=2,
    progress_bar=True
)

# Convert to arrays to avoid dtype issues
Z_arr = jnp.array(Z.values, dtype=jnp.float32)
X_arr = jnp.array(X, dtype=jnp.float32)  # already numeric
T_arr = jnp.array(T.values, dtype=jnp.float32)
D_arr = jnp.array(D.values, dtype=jnp.int32)
Y_arr = jnp.array(Y.values, dtype=jnp.float32)

fig, axes = plt.subplots(1, len(inference_model_configs),
                         figsize=(5 * len(inference_model_configs), 5),
                         tight_layout=True)

posterior_summaries = []
diagnostic_summaries = []
row_loo_values = []

for j, (inf_funcForm, figTitle) in enumerate(inference_model_configs):
    ax = axes[j] if len(inference_model_configs) > 1 else axes
    print(f"\nEstimating model: {inf_funcForm} on real data")

    datXY = {
        'Z': Z_arr,
        'D': D_arr,
        'X': X_arr,
        'Y': Y_arr,
        'T': T_arr,
        'funcForm': inf_funcForm
    }

    start = time.time()
    rng_key, rng_key_mcmc = random.split(rng_key)
    mcmc.run(rng_key_mcmc, **datXY)
    print("MCMC elapsed time:", time.time() - start)

    mcmc_samples = mcmc.get_samples()

    ppc = Predictive(model_POF, mcmc_samples)(
        rng_key, Z=Z_arr, X=X_arr, T=T_arr, D=D_arr
    )
    idata = az.from_numpyro(
        mcmc,
        posterior_predictive=ppc,
        coords={"obs_id": np.arange(len(Y_arr))},
        dims={"Y_obs": ["obs_id"]}
    )

    # Posterior summary — filter out Y0 and tau
    posterior_summary_tmp = az.summary(idata, round_to=4).reset_index()
    posterior_summary_tmp = posterior_summary_tmp[
        ~posterior_summary_tmp["index"].str.contains(r"^(tau|Y0)", regex=True)
    ]
    posterior_summary_tmp.insert(0, 'Inference_Form', inf_funcForm)
    posterior_summaries.append(posterior_summary_tmp)

    # PSIS-LOO & WAIC
    loo_res = az.loo(idata, pointwise=True, var_name='Y_obs')
    waic_res = az.waic(idata, pointwise=True, var_name='Y_obs')
    diagnostic_summaries.append({
        'Inference_Form': inf_funcForm,
        "LOO": [round(loo_res.elpd_loo, 4)],
        "LOO_SE": [round(loo_res.se, 4)],
        "WAIC": [round(waic_res.elpd_waic, 4)],
        "WAIC_SE": [round(waic_res.se, 4)]
    })

    row_loo_values.append(loo_res.elpd_loo.item())

    
loo_values_array = np.array(row_loo_values)

print(loo_values_array)
# %%
print(loo_values_array)
# %%
