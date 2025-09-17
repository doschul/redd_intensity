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
import arviz as az
import time

from numpyro.infer import MCMC, NUTS, Predictive

# %%
import jax.numpy as jnp
from jax import random

def raw_to_legendre(a, b, c, n_grid=200):
    """
    Project tau(T) = a*T^3 + b*T^2 + c*T   with T in [0,1]
    onto Legendre basis functions P1, P2, P3 evaluated on scaled T in [-1,1].
    Returns (beta1, beta2, beta3).
    """
    # grid of T in [0,1]
    T = jnp.linspace(0., 1., n_grid)
    tau = a*T**3 + b*T**2 + c*T
    T_scaled = 2*(T - 0.5)

    # Legendre basis
    P1 = T_scaled
    P2 = 0.5*(3*T_scaled**2 - 1)
    P3 = 0.5*(5*T_scaled**3 - 3*T_scaled)

    # Stack basis
    Phi = jnp.vstack([P1, P2, P3]).T  # shape (n_grid, 3)

    # Solve least squares: Phi @ beta ≈ tau
    beta, _, _, _ = jnp.linalg.lstsq(Phi, tau, rcond=None)
    return beta

beta1, beta2, beta3 = raw_to_legendre(a=-3.0, b=4.0, c=-0.8)
print(beta1, beta2, beta3)


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
        a = numpyro.sample("a", dist.Normal(0., 1.))
        b = numpyro.sample("b", dist.Normal(0., 1.))
        tau = a * T * (1 - T) + b * T
    elif funcForm == "poly3":
        a = numpyro.sample("a", dist.Normal(0., 1.))
        b = numpyro.sample("b", dist.Normal(0., 1.))
        c = numpyro.sample("c", dist.Normal(0., 1.))
        tau = a * T**3 + b * T**2 + c * T
    elif funcForm == "exponential":
        a = numpyro.sample("a", dist.Normal(0., 1.))
        #b = numpyro.sample("b", dist.Normal(0., 5.))  # keep sign free; exponent handles shape
        tau = a * (1 - jnp.exp(-T))
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
    This version uses Legendre polynomials (orthonormal basis on [-1,1]) for poly2/poly3,
    and fixes b=3.0 for the exponential DGP so it matches the model (which fixes b).
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

    # Treatment model truths (same as before)
    gamma_true = jnp.array([1., -0.5, 0.3, -0.8, 0.2], dtype=jnp.float32)
    omega_true = jnp.array(2., dtype=jnp.float32)
    beta_param_true = jnp.array(4., dtype=jnp.float32)

    logits = Z @ gamma_true
    base_alpha = jnp.exp(logits)
    alpha_param = base_alpha + omega_true * D

    T = dist.Beta(alpha_param, beta_param_true).sample(key=kT)

    # Scale T to [-1,1] for basis evaluation
    T_scaled = 2.0 * (T - 0.5)   # maps [0,1] -> [-1,1]

    # Choose tau truths using the SAME parameter names as the inference model:
    coefTrue = {
        "gamma": gamma_true,
        "omega": omega_true,
        "beta_param": beta_param_true,
        # alpha for baseline left random (keeps problem realistic)
    }

    # Set true coefficients *in the Legendre basis* (these are the parameters the model now estimates)
    if funcForm == 'linear':
        coefTrue['beta'] = jnp.array(0.8, dtype=jnp.float32)

    elif funcForm == 'poly2':
        # Legendre basis:
        # P1(x) = x
        # P2(x) = 0.5*(3x^2 - 1)
        # Choose plausible truth values for beta1,beta2 (these are recoverable by the model)
        coefTrue['beta1'] = jnp.array(1.0, dtype=jnp.float32)   # weight on P1
        coefTrue['beta2'] = jnp.array(0.3, dtype=jnp.float32)   # weight on P2

    elif funcForm == 'poly3':
        # Legendre basis up to degree 3:
        # P1(x) = x
        # P2(x) = 0.5*(3x^2 - 1)
        # P3(x) = 0.5*(5x^3 - 3x)
        coefTrue['beta1'] = jnp.array(0.25, dtype=jnp.float32)
        coefTrue['beta2'] = jnp.array(0.08, dtype=jnp.float32)
        coefTrue['beta3'] = jnp.array(-0.15, dtype=jnp.float32)

    elif funcForm == 'sigmoid':
        # Keep the same names as model
        coefTrue['b'] = jnp.array(0.6, dtype=jnp.float32)
        coefTrue['c'] = jnp.array(50., dtype=jnp.float32)
        coefTrue['d'] = jnp.array(0.4, dtype=jnp.float32)

    elif funcForm == 'exponential':
        coefTrue['a'] = jnp.array(0.6, dtype=jnp.float32)
        # NOTE: do NOT include 'b' because the model uses a fixed b (3.0)

    else:
        raise ValueError("Invalid funcForm specified")

    # Now build one draw of Y using the same model parameterization (condition the model on coefTrue)
    model_kwargs = {'Z': Z, 'X': X, 'T': T, 'D': D, 'funcForm': funcForm}
    # condition the same model you will fit, so the deterministic names match exactly
    condition_model = numpyro.handlers.condition(model_POF, data=coefTrue)
    predictive = Predictive(condition_model, num_samples=1)
    prior_samples = predictive(rng_key_, **model_kwargs)

    Y_unscaled = prior_samples['Y_obs'].squeeze()
    Y0         = prior_samples['Y0'].squeeze()
    tau_true   = prior_samples['tau'].squeeze()

    # Observed Y (no scaling)
    Y = Y_unscaled

    print(f'Mean(Y)={jnp.mean(Y):.4f}; std(Y)={jnp.std(Y):.4f}')
    print(f'Mean(T)={jnp.mean(T):.4f}; std(T)={jnp.std(T):.4f}')

    return Y, Y_unscaled, T, D, Z, X, Y0, tau_true, predictive, model_kwargs, coefTrue

# %% Quick test of the bimodal-like dosage by D
rng_key = random.PRNGKey(0)
rng_key, rng_key_ = random.split(rng_key)

Y, Y_unscaled, T, D, Z, X, Y0, tau_true, predictive, model_kwargs, coefTrue = data_generating(
    rng_key, funcForm="linear", N=10000, K=5
)

df = pd.DataFrame({"T": np.array(T), "D": np.array(D)})
sns.histplot(data=df, x='T', bins=30, kde=True, hue='D', palette="viridis")
plt.title("Histogram of T by D")
plt.xlabel("T")
plt.ylabel("Density")
plt.show()

# %% Inference on a smaller sample

# Create a dummy dataset for shape and size
N = 500
K = 5

kernel = NUTS(model_POF)
mcmc = MCMC(
    kernel,
    num_warmup=100,
    num_samples=200,
    num_chains=2,
    progress_bar=True
)

datXY = {
    'Z': Z,
    'D': D,
    'X': X,
    'Y': Y,
    'T': T,
    'funcForm': "poly2"
}

start = time.time()
rng_key, rng_key_mcmc = random.split(rng_key)
mcmc.run(rng_key_mcmc, **datXY)
print("\nMCMC elapsed time:", time.time() - start)

# %%
mcmc_samples = mcmc.get_samples()

# inspect posterior sample summaries
ppc = Predictive(model_POF, mcmc_samples)(
            rng_key_, Z=Z, X=X, T=T, D=D
        )
idata = az.from_numpyro(
    mcmc,
    posterior_predictive=ppc,
    coords={"obs_id": np.arange(len(Y))},
    dims={"Y_obs": ["obs_id"]}
)

# Posterior summary
posterior_summary_tmp = az.summary(idata, round_to=4).reset_index()
print("\nPosterior Summary (filtered):")
print(posterior_summary_tmp[~posterior_summary_tmp['index'].str.contains('Y0|tau|alpha')])

# %% Loop over DGP types
# Now loop through the model types
dgp_types = ['linear', 'exponential', 'poly2', 'poly3', 'sigmoid']


# Combine prior predictive plots into one figure
fig, axes = plt.subplots(1, len(dgp_types), figsize=(5 * len(dgp_types), 5), tight_layout=True)
for idx, dgp_funcForm in enumerate(dgp_types):
    rng_key, rng_key_ = random.split(rng_key)
    prior_predictive = Predictive(model_POF, num_samples=1000)
    T_plot = jnp.linspace(0, 1, 100)
    Z_plot = jnp.ones((100, K)) * jnp.mean(Z, axis=0)
    X_plot = jnp.ones((100, K)) * jnp.mean(X, axis=0)
    D_plot = jnp.zeros((100,))
    prior_samples = prior_predictive(rng_key_, Z=Z_plot, X=X_plot, T=T_plot, D=D_plot, funcForm=dgp_funcForm)
    ax = axes[idx]
    for i in range(1000):
        ax.plot(T_plot, prior_samples['tau'][i], color='k', alpha=0.02)
    ax.set_title(f"Prior Predictive: {dgp_funcForm}")
    ax.set_xlabel("Treatment Dosage (T)")
    ax.set_ylabel(r"Potential Outcome ($\tau$)")
    ax.set_xlim([0, 1])
    ax.set_ylim([-2, 2])
plt.show()


# %% MCMC

import arviz as az
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import jax.numpy as jnp
from numpyro.infer import MCMC, NUTS, Predictive
from jax import random
import time
import seaborn as sns

# --- Parameters ---
N = 500
K = 5
n_plot = 100

kernel = NUTS(model_POF)
mcmc = MCMC(
    kernel,
    num_warmup=50,
    num_samples=100,
    num_chains=2,
    progress_bar=True
)

dgp_types = ['linear',  'poly2','poly3', 'sigmoid',  'exponential']
#dgp_types = ['poly2', 'poly3']

inference_model_configs = [
    ('linear', 'i) Linear'),
    ('poly2', 'ii) Squared'),
    ('poly3', 'iii) Cubic'),
    ('sigmoid', 'iv) Sigmoid'),
    ('exponential', 'v) Exponential')
]

rng_key = random.PRNGKey(0)

fig, axes = plt.subplots(len(dgp_types), len(inference_model_configs), figsize=(20, 16), tight_layout=True)
fig.suptitle('Dose-Response Function by DGP (Row) and Inference Model (Column)', fontsize=24, y=1.02)

posterior_summaries = []
diagnostic_summaries = []

# Use seaborn's rocket palette as a colormap
cmap = sns.color_palette("rocket", as_cmap=True)

for i, dgp_funcForm in enumerate(dgp_types):
    print(f'\n\nGenerating data for funcForm: {dgp_funcForm}')
    
    rng_key, rng_key_data_gen = random.split(rng_key)
    (
        Y, Y_unscaled,
        T, D, Z, X, Y0, tau_true,
        conditioned_predictive, model_kwargs, coefTrue
    ) = data_generating(
        rng_key=rng_key_data_gen,
        funcForm=dgp_funcForm,
        N=N,
        K=K
    )

    row_loo_values = []

    for j, (inf_funcForm, figTitle) in enumerate(inference_model_configs):
        ax = axes[i, j]
        print(f'\n\nEstimating model: {inf_funcForm} for DGP: {dgp_funcForm}')
        
        datXY = {
            'Z': Z,
            'D': D,
            'X': X,
            'Y': Y_unscaled,
            'T': T,
            'funcForm': inf_funcForm
        }
        
        start = time.time()
        rng_key, rng_key_mcmc = random.split(rng_key)
        mcmc.run(rng_key_mcmc, **datXY)
        print("\nMCMC elapsed time:", time.time() - start)
        
        mcmc_samples = mcmc.get_samples()
                
        ppc = Predictive(model_POF, mcmc_samples)(
            rng_key_, Z=Z, X=X, T=T, D=D
        )
        idata = az.from_numpyro(
            mcmc,
            posterior_predictive=ppc,
            coords={"obs_id": np.arange(len(Y))},
            dims={"Y_obs": ["obs_id"]}
        )

        # Posterior summary
        posterior_summary_tmp = az.summary(idata, round_to=4).reset_index()
        posterior_summary_tmp.insert(0, 'DGP_Form', dgp_funcForm)
        posterior_summary_tmp.insert(1, 'Inference_Form', inf_funcForm)
        posterior_summaries.append(posterior_summary_tmp)

        # PSIS-LOO & WAIC
        loo_res = az.loo(idata, pointwise=True, var_name='Y_obs')
        waic_res = az.waic(idata, pointwise=True, var_name='Y_obs')
        diagnostic_summaries.append({
            'DGP_Form': dgp_funcForm,
            'Inference_Form': inf_funcForm,
            "LOO": [loo_res.elpd_loo],
            "LOO_SE": [loo_res.se],
            "WAIC": [waic_res.elpd_waic],
            "WAIC_SE": [waic_res.se]
        })
        # round to 4 decimal places
        for key in ["LOO", "LOO_SE", "WAIC", "WAIC_SE"]:
            diagnostic_summaries[-1][key] = [round(x, 4) for x in diagnostic_summaries[-1][key]]

        # Append the PSIS-LOO value for the current model
        row_loo_values.append(loo_res.elpd_loo.item())

        # Posterior draws for plot
        k = 1
        x_percentile = np.percentile(T, q=[0.1, 99])
        x_range = np.linspace(x_percentile[0], x_percentile[1], 100)
        x_mean = jnp.mean(X, axis=0)
        x_plot = jnp.repeat(x_mean.reshape(1, -1), 100, axis=0)
        x_plot = x_plot.at[:, k].set(x_range)
        
        rng_key, rng_key_post_predict_plot = random.split(rng_key)
        predictivePosterior = Predictive(model_POF, posterior_samples=mcmc_samples)
        
        # FIX: Generate a new `D` array of the correct size for plotting
        rng_key, rng_key_d = random.split(rng_key)
        D_plot = dist.Bernoulli(0.5).sample(key=rng_key_d, sample_shape=(100,))
        
        rng_key, rng_key_post_predict_plot = random.split(rng_key)
        predictivePosterior = Predictive(model_POF, posterior_samples=mcmc_samples)
        
        post_predict = predictivePosterior(
            rng_key_post_predict_plot,
            Z=x_plot,
            D=D_plot,
            X=x_plot,
            T=x_range,
            funcForm=inf_funcForm
        )
        tau_post_samples = post_predict['tau']
        
        for idx in range(1, n_plot):
            ax.plot(x_range, tau_post_samples[idx], color='k', alpha=0.1)
        
        if dgp_funcForm == 'homogenous':
            ax.axhline(y=tau_true[0], color='r', alpha=1)
        else:
            ax.plot(T, tau_true, 'o', color='r', alpha=1)
        
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_xlabel(f'Treatment (T)', fontsize=14)
        ax.set_ylabel(r'$\tau$ (Dose-Response)', fontsize=14)
        
        if i == 0:
            ax.set_title(figTitle, fontsize=16)
        if j == 0:
            ax.text(-0.3, 0.5, f'DGP: {dgp_funcForm.title()}',
                    transform=ax.transAxes, rotation=90, va='center', ha='right', fontsize=16)
        
        ax.set_xlim([x_percentile[0], x_percentile[1]])
        ax.set_ylim([-0.5, 1.5])
        if dgp_funcForm == inf_funcForm:
            ax.set_facecolor('#f0f0f0')

        # FIX: Add the PSIS-LOO and SE to the plot
        loo_value = loo_res.elpd_loo.item()
        loo_se_value = loo_res.se.item()
        text_to_add = f'PSIS-LOO: {loo_value:.2f}\nSE: {loo_se_value:.2f}'

        # Use ax.text to place the text in the top-left corner
        # The transform=ax.transAxes argument is crucial here.
        ax.text(0.05, 0.95, text_to_add,
                transform=ax.transAxes,
                ha='left',
                va='top',
                fontsize=12,
                bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8))
    
    # **NEW:** After the inner loop, iterate again to color all subplots
    loo_values_array = np.array(row_loo_values)
    # Ranks are from smallest (worst) to largest (best) PSIS-LOO
    # Note: argsort returns indices for ascending order. So the smallest value
    # has the lowest index. A higher rank corresponds to a better fit (larger PSIS-LOO).
    ranks = np.argsort(loo_values_array)
    
    for j, (inf_funcForm, figTitle) in enumerate(inference_model_configs):
        ax = axes[i, j]
        # Find the rank of the current model's LOO value
        # The rank is the position of j in the sorted array of indices.
        rank = np.where(ranks == j)[0][0]
        # Normalized rank: 0 (worst fit) to 1 (best fit)
        normalized_rank = rank / (len(inference_model_configs) - 1)
        # Use the colormap to get a color based on the normalized rank
        # The colormap maps a value from 0 to 1 to a color.
        # To get light blue (best fit) for the highest rank, the `cmap` is
        # defined from red to blue. A higher normalized rank will thus
        # correspond to a bluer color.
        color = cmap(normalized_rank)
        ax.set_facecolor(color)

plt.show()
# save plot
fig.savefig("POF_MCMC_v4_diagnostics.png", dpi=300)

posterior_summary_df = pd.concat(posterior_summaries, ignore_index=True)
diagnostic_summary_df = pd.DataFrame(diagnostic_summaries)

# save
diagnostic_summary_df.to_csv("POF_MCMC_v4_diagnostics.csv", index=False)
posterior_summary_df.to_csv("POF_MCMC_v4_posteriors.csv", index=False)

# %% Print summaries
print("\nPosterior Summaries:")
print(posterior_summary_df)

print("\nModel Diagnostics (LOO & WAIC):")
print(diagnostic_summary_df)


# %%

# %% Parameter recovery with posterior tau plots
rng_key = random.PRNGKey(0)
func_forms = ["linear","poly2","poly3","exponential","sigmoid"]
N, K, V = 1000, 5, 5
num_tau_draws = 200  # posterior draws to plot

for funcForm in func_forms:
    rng_key, rng_data = random.split(rng_key)
    Y, Y_std, Y0, T, D, Z, X, village_id, tau_true, predictive, model_kwargs, coefTrue = \
        data_generating(rng_data, funcForm=funcForm, N=N, K=K, V=V)

    kernel = NUTS(model_POF)
    mcmc = MCMC(kernel, num_warmup=150, num_samples=300, num_chains=2)
    rng_key, rng_mcmc = random.split(rng_key)
    datXY = model_kwargs.copy()
    datXY['Y'] = Y
    mcmc.run(rng_mcmc, **datXY)
    
    mcmc_samples = mcmc.get_samples()
    
    print(f"\nPosterior summary for {funcForm}:")
    for k,v in coefTrue.items():
        if k in mcmc_samples:
            post_mean = jnp.mean(mcmc_samples[k])
            print(f"{k}: true={v}, posterior mean={post_mean:.3f}")
    
    # --- Posterior tau draws for plotting (ignore village effects)
    T_unique = jnp.sort(jnp.unique(T))
    T_min, T_max = T.min(), T.max()
    T_scaled_unique = 2.0 * (T_unique - T_min) / (T_max - T_min) - 1.0

    # randomly pick 200 posterior samples
    idx_draws = np.random.choice(mcmc_samples[list(mcmc_samples.keys())[0]].shape[0], num_tau_draws, replace=False)

    plt.figure(figsize=(6,4))
    for i in idx_draws:
        plt.plot(T_unique, tau_true, color='blue', alpha=0.05)

    plt.plot(T_unique, tau_true, color='red', lw=2, label='True tau')
    plt.title(f"Posterior tau draws vs True tau ({funcForm})")
    plt.xlabel("T")
    plt.ylabel("tau(T)")
    plt.legend()
    plt.show()

# %%
