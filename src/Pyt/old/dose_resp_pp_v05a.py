# %%
import jax
import jax.numpy as jnp
from jax import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpyro
from numpyro import distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive
import arviz as az
import time

rng_key = random.PRNGKey(0)

# %% Model
def model_POF(Z, X, T, D, Y=None, village_id=None, V=None, funcForm="linear"):
    """
    POF with Poisson-observed dosage T (integer counts).
    Internally we map observed counts T -> continuous T_cont in [0,1]
    and compute Legendre basis on T_leg = 2*T_cont - 1 for polynomial tau forms.
    """

    # Raise an error if one is None but the other is not.
    if V is None and village_id is None:
        # Set V to 1 to represent a single "village" for all observations
        V = 1
        # Create an array of zeros with the same shape as your data for village_id
        village_id = jnp.zeros(Z.shape[0], dtype=jnp.int32)
        # Create an array of zeros for the single village effect
        u_v = jnp.zeros(V, dtype=jnp.float32)
    else:
        # This part remains the same as your previous correct code
        sigma_u = numpyro.sample("sigma_u", dist.Exponential(1.0))
        with numpyro.plate("village", V):
            u_v = numpyro.sample("u_v", dist.Normal(0., sigma_u))

    N, Kz = Z.shape
    _, Kx = X.shape

    # --- Baseline outcome coefficients (model fits on standardized Y)
    alpha = numpyro.sample("alpha", dist.Normal(0., 1.).expand([Kx]))
    Y0 = jnp.dot(X, alpha)

    # --- Poisson dosage model (counts, allows zeros)
    numpyro.sample("D_obs", dist.Bernoulli(0.5), obs=D)
    gamma = numpyro.sample("gamma", dist.Normal(0., 1.).expand([Kz]))
    omega = numpyro.sample("omega", dist.Normal(0., 1.))
    eta = numpyro.sample("eta", dist.Exponential(1.))
    logits = jnp.dot(Z, gamma) + omega * D
    lam = jnp.exp(jnp.clip(logits, -20., 20.))
    numpyro.sample("T_obs", dist.NegativeBinomial2(lam, eta), obs=T)  # T here are integer counts

    # --- create continuous T in [0,1] from counts for internal use
    # add 0.0 cast to float and protect against division by zero if max==min
    T = jnp.asarray(T).astype(jnp.float32)
    T_min, T_max = jnp.min(T), jnp.max(T)
    T_cont = T / T_max  # now in [0,1]
    T_leg = 2.0 * T_cont - 1.0    # for Legendre polynomials internally

    # --- Dose-response tau(T) using internal T_cont / T_leg
    if funcForm == "homogenous":
        b1 = numpyro.sample("b1", dist.Normal(0., 1.))
        tau = b1 * D
    
    elif funcForm == "linear":
        b1 = numpyro.sample("b1", dist.Normal(0., 1.))
        # linear effect on Legendre P1 (T_leg)
        tau = b1 * T_cont

    elif funcForm == "poly2":
        b1 = numpyro.sample("b1", dist.Normal(0., 1.))
        b2 = numpyro.sample("b2", dist.Normal(0., 1.))
        tau = b1 * T_cont + b2 * T_cont**2
    
    elif funcForm == "poly2_legendre":
        P1 = T_leg
        P2 = 0.5 * (3.0 * T_leg**2 - 1.0)
        b = numpyro.sample("b", dist.Normal(0., 1.))
        tau = b * (P1 + P2)

    elif funcForm == "poly3":
        b1 = numpyro.sample("b1", dist.Normal(0., 1.))
        b2 = numpyro.sample("b2", dist.Normal(0., 1.))
        b3 = numpyro.sample("b3", dist.Normal(0., 1.))
        tau = b1 * T_cont + b2 * T_cont**2 + b3 * T_cont**3

    elif funcForm == "poly3_legendre":
        P1 = T_leg
        P2 = 0.5 * (3.0 * T_leg**2 - 1.0)
        P3 = 0.5 * (5.0 * T_leg**3 - 3.0 * T_leg)
        a1 = numpyro.sample("a1", dist.Normal(0., 1.))
        a2 = numpyro.sample("a2", dist.Normal(0., 1.))
        tau = a1 * (P1 + P2) + a2 * (P2 + P3)

    elif funcForm == "exponential":
        b1 = numpyro.sample("b1", dist.Normal(0., 1.))
        b2 = 1.0
        tau = b1 * (1.0 - jnp.exp(-b2 * T_cont))

    elif funcForm == "sigmoid":
        b1 = numpyro.sample("b1", dist.Normal(0., 1.))
        b2 = numpyro.sample("b2", dist.Exponential(0.02))
        b3 = numpyro.sample("b3", dist.Uniform(0., 1.))
        tau = b1 * (1.0 / (1.0 + jnp.exp(-b2 * (T_cont - b3))))

    else:
        raise ValueError("Invalid funcForm specified")

    # add village effects to tau
    tau = tau + u_v[village_id]

    # --- Outcome (we fit standardized Y)
    sigma_Y = numpyro.sample("sigma_Y", dist.Exponential(1.0))
    Y_mean = Y0 + tau
    numpyro.sample("Y_obs", dist.Normal(Y_mean, sigma_Y), obs=Y)

    # deterministics for inspection
    #numpyro.deterministic("Y0", Y0)
    numpyro.deterministic("tau", tau)
    #numpyro.deterministic("lam", lam)
    #numpyro.deterministic("T_cont", T_cont)


# %% Data generation
def data_generating(rng_key, funcForm="linear", N=1000, K=5, V=20):
    rng_key, kX, kZ, kD, kT, kVill, kv = random.split(rng_key, 7)

    # --- covariates
    X = jnp.hstack([jnp.ones((N,1)), random.normal(kX,(N,K-1))])
    Z = jnp.hstack([jnp.ones((N,1)), random.normal(kZ,(N,K-1))])

    # --- village ids
    if V is None:
        V = 1
        village_id = jnp.zeros(Z.shape[0], dtype=jnp.int32)
        village_u = jnp.zeros((V,), dtype=jnp.float32)  # no village effect
    else:
        village_id = random.randint(kVill, shape=(N,), minval=0, maxval=V)
        # --- village effect
        sigma_u_true = 0.8
        village_u = dist.Normal(0., sigma_u_true).sample(kv,(V,))
    
    u_effect = village_u[village_id]

    # --- placement D
    D = dist.Bernoulli(0.5).sample(key=kD, sample_shape=(N,)).astype(jnp.float32)

    # --- true Poisson dosage (counts, allows zeros)
    gamma_true = jnp.array([0.5, -0.1, 0.5, 0.8, -0.2])
    omega_true = 1.0
    lam = jnp.exp(jnp.dot(Z, gamma_true) + omega_true * D)
    eta_true = 2.0  # dispersion parameter for Negative Binomial
    T_counts = dist.NegativeBinomial2(lam, eta_true).sample(key=kT)  # integer counts, may contain zeros

    # --- produce continuous version in (0,1) for internal evaluation & plotting
    T_counts_f = T_counts.astype(jnp.float32)
    T_min, T_max = jnp.min(T_counts_f), jnp.max(T_counts_f)
    T_cont = T_counts_f / T_max  # in [0,1]
    T_leg = 2.0 * T_cont - 1.0

    # --- true tau without village effect (use same internal T_leg / T_cont logic)
    if funcForm == "homogenous":
        b1 = 0.5
        tau_true = b1 * D  # constant effect across T
        coefTrue = {'b1': b1, 'gamma': gamma_true, 'omega': omega_true}
    
    elif funcForm == "linear":
        b1 = 0.8
        tau_true = b1 * T_cont
        coefTrue = {'b1': b1, 'gamma': gamma_true, 'omega': omega_true}

    elif funcForm == "poly2":
        b1, b2 = 1.8, -0.9
        tau_true = T_cont*b1 + b2*T_cont**2
        coefTrue = {'b1': b1, 'b2': b2, 'gamma': gamma_true, 'omega': omega_true}
    
    elif funcForm == "poly2_legendre":
        b_true = -0.6  # Example true value for b
        P1, P2 = T_leg, 0.5*(3*T_leg**2 - 1)
        tau_true = b_true * (P1 + P2)
        coefTrue = {'b': b_true, 'gamma': gamma_true, 'omega': omega_true}
            
    elif funcForm == "poly3":
        b1, b2, b3 = -0.5, 1.7, -1.0
        tau_true = T_cont*b1 + b2*T_cont**2 + b3*T_cont**3
        coefTrue = {'b1': b1, 'b2': b2, 'b3': b3, 'gamma': gamma_true, 'omega': omega_true}

    elif funcForm == "poly3_legendre":
        a1, a2 = 0.53, -0.3
        P1 = T_leg
        P2 = 0.5 * (3.0 * T_leg**2 - 1.0)
        P3 = 0.5 * (5.0 * T_leg**3 - 3.0 * T_leg)
        tau_true = a1 * (P1 + P2) + a2 * (P2 + P3)
        coefTrue = {'a1': a1, 'a2': a2,
                    'gamma': gamma_true, 'omega': omega_true}

    elif funcForm == "exponential":
        b1 = 0.6
        b2 = 1.0 # explicitly define the exponential decay rate
        tau_true = b1 * (1 - jnp.exp(-b2 * T_cont))
        coefTrue = {'b1': b1, 'b2': b2, 'gamma': gamma_true, 'omega': omega_true}
        
    elif funcForm == "sigmoid":
        b1, b2, b3 = 0.6, 50.0, 0.4
        tau_true = b1 * (1/(1 + jnp.exp(-b2*(T_cont - b3))))
        coefTrue = {'b1': b1, 'b2': b2, 'b3': b3, 'gamma': gamma_true, 'omega': omega_true}

    else:
        raise ValueError("Invalid funcForm")
    
    tau_true += u_effect  # add village effect to tau

    # --- baseline outcome (original units)
    alpha_true = jnp.ones(K)
    Y0 = jnp.dot(X, alpha_true)

    # --- observed Y (original units)
    sigma_Y_true = 0.1
    Y_unscaled = Y0 + tau_true + dist.Normal(0., sigma_Y_true).sample(rng_key, (N,))

    # --- standardize Y for model fitting
    Y_mean, Y_std = Y_unscaled.mean(), Y_unscaled.std()
    Y_scaled = (Y_unscaled - Y_mean)/Y_std

    # --- model_kwargs: pass Poisson counts as T (observed), and Y_scaled for fitting
    model_kwargs = {"Z": Z, "X": X, "T": T_counts, "D": D, "Y": Y_unscaled,
                    "village_id": village_id, "V": V, "funcForm": funcForm}

    return (Y_scaled, Y_unscaled, Y0, T_counts, T_cont, D, Z, X,
            village_id, tau_true, T_leg, model_kwargs, coefTrue, Y_mean, Y_std)


# %% Test bimodal-like dosage by D

N = 1000
K = 5
V = None
village_id = None
num_tau_draws = 200

rng_key = random.PRNGKey(0)
rng_key, rng_key_ = random.split(rng_key)

Y_scaled, Y_unscaled, Y0, T_counts, T_cont, D, Z, X, village_id, tau_true, T_leg, model_kwargs, coefTrue, Y_mean, Y_std = data_generating(
    rng_key, funcForm="linear", N=N, K=K, V=V
)

df = pd.DataFrame({"T": np.array(T_cont), "D": np.array(D)})
sns.histplot(data=df, x='T', bins=30, kde=True, hue='D', palette="viridis")
plt.title("Histogram of T by D")
plt.xlabel("T")
plt.ylabel("Density")
plt.show()

# %% Plot priors (no data)

# include homogenous too
func_forms = ['homogenous', 'linear', 'exponential', 'poly2', 'poly3', 'sigmoid']

# Combine prior predictive plots into one figure
fig, axes = plt.subplots(1, len(func_forms), figsize=(5 * len(func_forms), 5), tight_layout=True)
for idx, dgp_funcForm in enumerate(func_forms):
    rng_key, rng_key_ = random.split(rng_key)
    prior_predictive = Predictive(model_POF, num_samples=1000)
    T_plot = jnp.linspace(0, 1, 100)
    Z_plot = jnp.ones((100, K)) * jnp.mean(Z, axis=0)
    X_plot = jnp.ones((100, K)) * jnp.mean(X, axis=0)
    D_plot = jnp.zeros((100,))
    V_plot = None  # no village effects in prior predictive
    prior_samples = prior_predictive(rng_key_, Z=Z_plot, X=X_plot, T=T_plot, D=D_plot, V=V_plot, funcForm=dgp_funcForm)
    ax = axes[idx]
    for i in range(1000):
        ax.plot(T_plot, prior_samples['tau'][i], color='k', alpha=0.02)
    ax.set_title(f"Prior Predictive: {dgp_funcForm}")
    ax.set_xlabel("Treatment Dosage (T)")
    ax.set_ylabel(r"Potential Outcome ($\tau$)")
    ax.set_xlim([0, 1])
    ax.set_ylim([-2, 2])
plt.show()


# %% Parameter recovery 

func_forms = [ 'exponential', 'poly2', 'poly3', 'sigmoid']

parameter_recovery = []

for funcForm in func_forms:
    rng_key, rng_data = random.split(rng_key)
    (Y_scaled, Y_unscaled, Y0, T_counts, T_cont, D, Z, X, village_id, tau_true, T_leg, model_kwargs, coefTrue, Y_mean, Y_std) = \
        data_generating(rng_data, funcForm=funcForm, N=N, K=K, V=V)

    # --- run MCMC (fit to standardized Y; T is Poisson counts)
    kernel = NUTS(model_POF)
    mcmc = MCMC(kernel, num_warmup=100, num_samples=200, num_chains=1)
    rng_key, rng_mcmc = random.split(rng_key)
    mcmc.run(rng_mcmc, **model_kwargs)
    samples = mcmc.get_samples()

    # --- parameter recovery
    print(f"\nPosterior summary for {funcForm}:")
    for k, v in coefTrue.items():
        if k in samples:
            post_mean = jnp.mean(samples[k])
            print(f"{k}: true={v}, posterior mean={post_mean:.3f}")

    parameter_recovery.append((funcForm, {k: (v, post_mean) for k, v in coefTrue.items() if k in samples}))

    # --- evenly spaced T grid from 0 to 1
    T_plot = jnp.linspace(0, 1, 100)
    T_leg_plot = 2.0 * T_plot - 1.0

    # pick posterior draws
    any_param = list(samples.keys())[0]
    n_post_draws = samples[any_param].shape[0]
    idx_draws = np.random.choice(n_post_draws, min(num_tau_draws, n_post_draws), replace=False)

    plt.figure(figsize=(7,4))
    for i in idx_draws:
        if funcForm == "homogenous":
            tau_draw = jnp.repeat(samples['b1'][i], len(T_plot))  # constant across T
        elif funcForm == "linear":
            tau_draw = samples['b1'][i] * T_plot
        elif funcForm == "poly2":
            #P1, P2 = T_leg_plot, 0.5*(3*T_leg_plot**2 - 1)
            tau_draw = samples['b1'][i]*T_plot + samples['b2'][i]*T_plot**2
        elif funcForm == "poly2_legendre":
            P1 = T_leg_plot
            P2 = 0.5 * (3.0 * T_leg_plot**2 - 1.0)
            tau_draw = samples['b'][i] * (P1 + P2)
        elif funcForm == "poly3":
            tau_draw = (samples['b1'][i] * T_plot + samples['b2'][i] * T_plot**2 + samples['b3'][i] * T_plot**3)
        elif funcForm == "poly3_legendre":
            P1 = T_leg_plot
            P2 = 0.5 * (3.0 * T_leg_plot**2 - 1.0)
            P3 = 0.5 * (5.0 * T_leg_plot**3 - 3.0 * T_leg_plot)
            # make tau: a1 * (P1 + P2) + a2 * (P2 + P3)
            tau_draw = samples['a1'][i] * (P1 + P2) + samples['a2'][i] * (P2 + P3)
        elif funcForm == "exponential":
            b_fixed = 1.0
            tau_draw = samples['b1'][i] * (1 - jnp.exp(-b_fixed * T_plot))
        elif funcForm == "sigmoid":
            b_s = samples['b1'][i]
            c_s = samples['b2'][i]
            d_s = samples['b3'][i]
            tau_draw = b_s * (1/(1 + jnp.exp(-c_s * (T_plot - d_s))))
        else:
            raise ValueError("Invalid funcForm for plotting")

        plt.plot(T_plot, np.asarray(tau_draw), color='blue', alpha=0.05)

    # --- True tau on same grid (deterministic, no u_v)
    if funcForm == "homogenous":
        tau_true_grid = jnp.repeat(coefTrue['b1'], len(T_plot))
    elif funcForm == "linear":
        tau_true_grid = coefTrue['b1'] * T_plot
    elif funcForm == "poly2":
        #P1, P2 = T_leg_plot, 0.5*(3*T_leg_plot**2 - 1)
        tau_true_grid = coefTrue['b1']*T_plot + coefTrue['b2']*T_plot**2
    elif funcForm == "poly2_legendre":
        P1 = T_leg_plot
        P2 = 0.5 * (3.0 * T_leg_plot**2 - 1.0)
        b_true = coefTrue['b']
        tau_true_grid = b_true * (P1 + P2)
    elif funcForm == "poly3":
        tau_true_grid = (coefTrue['b1'] * T_plot + coefTrue['b2'] * T_plot**2 + coefTrue['b3'] * T_plot**3)
    elif funcForm == "poly3_legendre":
        P1 = T_leg_plot
        P2 = 0.5*(3*T_leg_plot**2 - 1)
        P3 = 0.5*(5*T_leg_plot**3 - 3*T_leg_plot)
        a1 = coefTrue['a1']
        a2 = coefTrue['a2']
        tau_true_grid = a1 * (P1 + P2) + a2 * (P2 + P3)
    elif funcForm == "exponential":
        tau_true_grid = coefTrue['b1'] * (1 - jnp.exp(-coefTrue['b2'] * T_plot))
    elif funcForm == "sigmoid":
        tau_true_grid = coefTrue['b1'] * (1/(1 + jnp.exp(-coefTrue['b2'] * (T_plot - coefTrue['b3']))))

    plt.plot(T_plot, np.asarray(tau_true_grid), color='red', lw=2, label='True tau')

    plt.title(f"Posterior tau draws vs True tau ({funcForm}) — evenly spaced T∈[0,1]")
    plt.xlabel("T (scaled to [0,1])")
    plt.ylabel("tau (standardized Y-scale)")
    plt.legend()
    plt.show()


# %% MCMC Matrix

# --- Parameters ---
N = 240
K = 5
V = 8  # number of villages
n_plot = 200 # posterior spaghetti lines per subplot
n_warmup = 150
n_sample = 300
n_chains = 2

# --- Models to fit
inference_model_configs = [
    ('homogenous', 'Homogenous'),
    ('linear', 'Linear'),
    ('exponential', 'Exponential'),
    ('sigmoid', 'Sigmoid'),
    ('poly2', 'Poly2'),
    ('poly3', 'Poly3 (Standard)')
]

# --- DGP configurations
dgp_types = ['homogenous', 'linear', 'exponential', 'sigmoid', 'poly2', 'poly3']

rng_key = random.PRNGKey(0)

fig, axes = plt.subplots(len(dgp_types), len(inference_model_configs), figsize=(22, 20), tight_layout=True)
if len(dgp_types) == 1:
    axes = np.array([axes])
fig.suptitle('Dose-Response Function by DGP (Row) and Inference Model (Column)', fontsize=24, y=1.02)

posterior_summaries = []
diagnostic_summaries = []

# Use half of the 'rocket' colormap
full_cmap = sns.color_palette("rocket", as_cmap=True)
colors = full_cmap(np.linspace(0, 1, 256))
half_colors = colors[256//2:]
inverted_half_colors = half_colors[::-1]
cmap = plt.cm.colors.ListedColormap(inverted_half_colors)

for i, dgp_funcForm in enumerate(dgp_types):
    print(f'\nGenerating data for DGP: {dgp_funcForm}')
    
    rng_key, rng_key_data = random.split(rng_key)
    (
        Y_scaled, Y_unscaled, Y0, T_counts, T_cont, D, Z, X, village_id,
        tau_true, T_leg, model_kwargs, coefTrue, Y_mean, Y_std
    ) = data_generating(
        rng_key=rng_key_data,
        funcForm=dgp_funcForm,
        N=N,
        K=K,
        V=V
    )

    row_loo_values = []

    for j, (inf_funcForm, figTitle) in enumerate(inference_model_configs):
        ax = axes[i, j]
        print(f'Estimating {inf_funcForm} model for DGP {dgp_funcForm}')

        mcmc_args = {
            'Z': Z,
            'D': D,
            'X': X,
            'Y': Y_unscaled,
            'T': T_counts,
            'village_id': village_id,
            'V': V,
            'funcForm': inf_funcForm
        }
        
        kernel = NUTS(model_POF)
        mcmc = MCMC(
            kernel,
            num_warmup=n_warmup,
            num_samples=n_sample,
            num_chains=n_chains,
            progress_bar=True
        )

        rng_key, rng_key_mcmc = random.split(rng_key)
        start_time = time.time()
        mcmc.run(rng_key_mcmc, **mcmc_args)
        print("MCMC elapsed time:", round(time.time() - start_time, 2), "sec")
        mcmc_samples = mcmc.get_samples()

        # Posterior predictive
        rng_key, rng_key_ppc = random.split(rng_key)
        predictivePosterior = Predictive(model_POF, posterior_samples=mcmc_samples)

        # For τ(T) plotting
        #x_percentile = np.percentile(T_cont, q=[0.1, 99])
        #x_range = np.linspace(x_percentile[0], x_percentile[1], 100)
        x_range = jnp.linspace(0.0, 1.0, 100)
        x_mean = jnp.mean(X, axis=0)
        x_plot = jnp.repeat(x_mean.reshape(1, -1), 100, axis=0)
        D_plot = jnp.ones(100)
        
        # Pass T_cont for plotting, with dummy village info
        ppc_args = {
            'Z': x_plot,
            'D': D_plot,
            'X': x_plot,
            'T': x_range,
            'village_id': None,
            'V': None,
            'funcForm': inf_funcForm
        }
        
        rng_key, rng_key_tau = random.split(rng_key)
        post_predict = predictivePosterior(rng_key_tau, **ppc_args)
        tau_post_samples = post_predict['tau']

        # Posterior spaghetti
        n_draws = min(n_plot, tau_post_samples.shape[0])
        for idx in range(n_draws):
            ax.plot(x_range, tau_post_samples[idx], color='k', alpha=0.05)

        # --- True τ(T) (no village effect)
        T_leg_plot = 2.0 * x_range - 1.0
        # --- Corrected plotting for true tau
        if dgp_funcForm == "homogenous":
            tau_true_grid = jnp.repeat(coefTrue['b1'], len(x_range))
        elif dgp_funcForm == "linear":
            tau_true_grid = coefTrue['b1'] * x_range
        elif dgp_funcForm == "poly2":
            tau_true_grid = coefTrue['b1']*x_range + coefTrue['b2']*x_range**2
        elif dgp_funcForm == "poly2_legendre":
            P1 = T_leg_plot
            P2 = 0.5 * (3.0 * T_leg_plot**2 - 1.0)
            tau_true_grid = coefTrue['b'] * (P1 + P2)
        elif dgp_funcForm == "poly3":
            tau_true_grid = (coefTrue['b1'] * x_range + coefTrue['b2'] * x_range**2 + coefTrue['b3'] * x_range**3)
        elif dgp_funcForm == "poly3_legendre":
            P1 = T_leg_plot
            P2 = 0.5*(3*T_leg_plot**2 - 1)
            P3 = 0.5*(5*T_leg_plot**3 - 3*T_leg_plot)
            a1 = coefTrue['a1']
            a2 = coefTrue['a2']
            tau_true_grid = a1 * (P1 + P2) + a2 * (P2 + P3)
        elif dgp_funcForm == "exponential":
            tau_true_grid = coefTrue['b1'] * (1 - jnp.exp(-coefTrue['b2'] * x_range))
        elif dgp_funcForm == "sigmoid":
            tau_true_grid = coefTrue['b1'] * (1/(1 + jnp.exp(-coefTrue['b2'] * (x_range - coefTrue['b3']))))
        else:
            raise ValueError("Invalid funcForm for plotting true curve")

        ax.plot(x_range, np.asarray(tau_true_grid), color='red', lw=2, label='True tau')
        
        # --- Arviz summaries
        rng_key, rng_key_ppc = random.split(rng_key)
        # Pass full MCMC arguments to predictive for loo
        ppc_loo = predictivePosterior(rng_key_ppc, **mcmc_args)
        idata = az.from_numpyro(
            mcmc,
            posterior_predictive=ppc_loo,
            coords={"obs_id": np.arange(len(Y_unscaled))},
            dims={"Y_obs": ["obs_id"]}
        )

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
            "LOO": round(loo_res.elpd_loo.item(), 4),
            "LOO_SE": round(loo_res.se.item(), 4),
            "WAIC": round(waic_res.elpd_waic.item(), 4),
            "WAIC_SE": round(waic_res.se.item(), 4)
        })
        row_loo_values.append(loo_res.elpd_loo.item())

        # Axis formatting
        x_percentile = np.percentile(x_range, [2.5, 97.5])
        ax.set_xlim([x_percentile[0], x_percentile[1]])
        ax.set_ylim([-0.5, 1.5])
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_xlabel("Treatment (T)", fontsize=12)
        ax.set_ylabel("τ (Dose-Response)", fontsize=12)
        if i == 0:
            ax.set_title(figTitle, fontsize=14)
        if j == 0:
            ax.text(-0.25, 0.5, f'DGP: {dgp_funcForm.title()}',
                    transform=ax.transAxes, rotation=90, va='center', ha='right', fontsize=14)
        ax.text(0.05, 0.95, f"PSIS-LOO: {loo_res.elpd_loo:.2f}\nSE: {loo_res.se:.2f}",
                transform=ax.transAxes, ha='left', va='top', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8))

    # Color subplots by PSIS-LOO rank
    loo_values_array = np.array(row_loo_values)
    ranks = np.argsort(-loo_values_array)
    for j, _ in enumerate(inference_model_configs):
        ax = axes[i, j]
        rank = np.where(ranks == j)[0][0]
        normalized_rank = rank / (len(inference_model_configs) - 1)
        color = cmap(normalized_rank)
        ax.set_facecolor(color)

plt.show()
fig.savefig("POF_MCMC_v6_diagnostics.png", dpi=300)

# Save summaries
posterior_summary_df = pd.concat(posterior_summaries, ignore_index=True)
diagnostic_summary_df = pd.DataFrame(diagnostic_summaries)

posterior_summary_df.to_csv("POF_MCMC_v6_posteriors.csv", index=False)
diagnostic_summary_df.to_csv("POF_MCMC_v6_diagnostics.csv", index=False)
# %% Check summary
print(diagnostic_summary_df)
str_exclude = 'Y0|T_cont|alpha|gamma|u_v|tau|lam'
print(posterior_summary_df[posterior_summary_df['index'].str.contains(str_exclude) == False])

# %% Load real data

# load csv file from \data
try:
    df1 = pd.read_csv('C:/Users/DaSchulz/OneDrive - European Forest Institute/Dokumente/research/redd_intensity/data/pd_short_ALL.csv')
except FileNotFoundError:
    print("Error: The CSV file was not found. Please check and correct the file path.")
    exit()

# **NEW:** Drop rows only when missing values are in the variables of interest.
# This prevents dropping rows with missing data in columns you aren't using.
relevant_columns = [
    "DY_forest_share","forest_share",
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
X_names = ["forest_share","Marital_status","Hh_cooking_tech","Hh_floor2","Hh_walls2","Hh_roof2","Hh_water","Hh_toilet","Hh_electric","Hh_cooking_fuel"]
Z_names = ["int_nonREDD_ongoing_sum","int_nonREDD_ended_sum"]
D_name = "Village_Type"
T_name = "int_cat_tot_involALL"
Y_name = "DY_forest_share"
village_column = "Village"

# first standardize X, Z, and T within df1
df1[X_names] = (df1[X_names] - df1[X_names].mean()) / df1[X_names].std()
df1[Z_names] = (df1[Z_names] - df1[Z_names].mean()) / df1[Z_names].std()
df1['T01'] = (df1[T_name] - df1[T_name].min()) / (df1[T_name].max() - df1[T_name].min())


# %% plot histogram of T colored by D
plt.figure(figsize=(10, 6))
for d in df1[D_name].unique():
    plt.hist(df1[T_name][df1[D_name] == d], bins=30, alpha=0.5, label=str(d))
plt.xlabel("Treatment Intensity (T)")
plt.ylabel("Frequency")
plt.title("Histogram of Treatment Intensity (T) by Village Type (D)")
plt.legend(title="Village Type (D)")
plt.show()


# %% Run MCMC on real data
  
# --- Parameters ---
n_plot = 200 # posterior spaghetti lines per subplot
n_warmup = 150
n_sample = 300
n_chains = 2

# initialize kernel
kernel = NUTS(model_POF)
mcmc = MCMC(
    kernel,
    num_warmup=n_warmup,
    num_samples=n_sample,
    num_chains=n_chains,
    progress_bar=True
)

# --- Models to fit
inference_model_configs = [
    ('homogenous', 'Homogenous'),
    ('linear', 'Linear'),
    ('exponential', 'Exponential'),
    ('sigmoid', 'Sigmoid'),
    ('poly2', 'Poly2'),
    ('poly3', 'Poly3 (Standard)')
]

# --- DGP configurations
dgps = df1['country_project'].unique()

rng_key = random.PRNGKey(0)

# Use half of the 'rocket' colormap
full_cmap = sns.color_palette("rocket", as_cmap=True)
colors = full_cmap(np.linspace(0, 1, 256))
half_colors = colors[256//2:]
inverted_half_colors = half_colors[::-1]
cmap = plt.cm.colors.ListedColormap(inverted_half_colors)

# %% Plot real data posteriors
fig, axes = plt.subplots(len(dgps), len(inference_model_configs), figsize=(20, 40), tight_layout=True)
if len(dgps) == 1:
    axes = np.array([axes])
fig.suptitle('Dose-Response Function by DGP (Row) and Inference Model (Column)', fontsize=24, y=1.02)

posterior_summaries = []
diagnostic_summaries = []

for i, dgp_name in enumerate(dgps):
    print(f"\nProcessing DGP: {dgp_name}")
    df_dgp = df1[df1['country_project'] == dgp_name].copy()

    # Make variables
    X = df_dgp[X_names].to_numpy()
    Z = df_dgp[Z_names].to_numpy()
    D = df_dgp[D_name].to_numpy()
    Y = df_dgp[Y_name].to_numpy()
    T_counts = df_dgp[T_name].to_numpy().astype(int)
    T_cont = (T_counts.astype(float) - T_counts.min()) / max(1, (T_counts.max() - T_counts.min()))

    # Add constant column
    X = np.column_stack((X, np.ones(X.shape[0])))
    Z = np.column_stack((Z, np.ones(Z.shape[0])))

    # prepare village FE
    village_id = df_dgp[village_column].astype('category').cat.codes
    V = village_id.nunique()
    village_id = village_id.to_numpy()
    n_obs = len(Y)

    print(f"Number of villages: {V}, Number of observations: {len(Y)}")
    
    row_loo_values = []

    for j, (inf_funcForm, figTitle) in enumerate(inference_model_configs):
        ax = axes[i, j]
        print(f'Estimating {inf_funcForm} model for DGP {dgp_name}')

        mcmc_args = {
            'Z': Z,
            'D': D,
            'X': X,
            'Y': Y,
            'T': T_counts,
            'village_id': village_id,
            'V': V,
            'funcForm': inf_funcForm
        }
        
        kernel = NUTS(model_POF)
        mcmc = MCMC(
            kernel,
            num_warmup=n_warmup,
            num_samples=n_sample,
            num_chains=n_chains,
            progress_bar=True
        )

        rng_key, rng_key_mcmc = random.split(rng_key)
        start_time = time.time()
        mcmc.run(rng_key_mcmc, **mcmc_args)
        print("MCMC elapsed time:", round(time.time() - start_time, 2), "sec")
        mcmc_samples = mcmc.get_samples()

        # Posterior predictive
        rng_key, rng_key_ppc = random.split(rng_key)
        predictivePosterior = Predictive(model_POF, posterior_samples=mcmc_samples)

        # For τ(T) plotting
        x_percentile = np.percentile(T_cont, q=[0.1, 99])
        x_range = np.linspace(x_percentile[0], x_percentile[1], 100)
        x_mean = jnp.mean(X, axis=0)
        x_plot = jnp.repeat(x_mean.reshape(1, -1), 100, axis=0)
        z_mean = jnp.mean(Z, axis=0)
        z_plot = jnp.repeat(z_mean.reshape(1, -1), 100, axis=0)
        D_plot = jnp.ones(100)
        
        # Pass T_cont for plotting, with dummy village info
        ppc_args = {
            'Z': z_plot,
            'D': D_plot,
            'X': x_plot,
            'T': x_range,
            'village_id': None,
            'V': None,
            'funcForm': inf_funcForm
        }
        
        rng_key, rng_key_tau = random.split(rng_key)
        post_predict = predictivePosterior(rng_key_tau, **ppc_args)
        tau_post_samples = post_predict['tau']

        # Posterior spaghetti
        n_draws = min(n_plot, tau_post_samples.shape[0])
        for idx in range(n_draws):
            ax.plot(x_range, tau_post_samples[idx], color='k', alpha=0.05)

        # --- Arviz summaries
        rng_key, rng_key_ppc = random.split(rng_key)
        # Pass full MCMC arguments to predictive for loo
        ppc_loo = predictivePosterior(rng_key_ppc, **mcmc_args)
        idata = az.from_numpyro(
            mcmc,
            posterior_predictive=ppc_loo,
            coords={"obs_id": np.arange(len(Y))},
            dims={"Y_obs": ["obs_id"]}
        )

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
            "LOO": round(loo_res.elpd_loo.item(), 4),
            "LOO_SE": round(loo_res.se.item(), 4),
            "WAIC": round(waic_res.elpd_waic.item(), 4),
            "WAIC_SE": round(waic_res.se.item(), 4)
        })
        row_loo_values.append(loo_res.elpd_loo.item())

        # Axis formatting
        x_percentile = np.percentile(x_range, [2.5, 97.5])
        ax.set_xlim([x_percentile[0], x_percentile[1]])
        ax.set_ylim([-0.5, 1.5])
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)        
        ax.set_xticks([])
        #ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        if i == 0:
            ax.set_title(figTitle, fontsize=14)
        if j == 0:
            ax.yaxis.set_ticks_position("left")
            ax.text(-0.25, 0.5, f'Project: {dgp_name.title()}',
                    transform=ax.transAxes, va='center', ha='right', fontsize=14)
        else:
            ax.set_yticks([])
        ax.text(0.05, 0.95, f"PSIS-LOO: {loo_res.elpd_loo:.2f}\nSE: {loo_res.se:.2f}",
                transform=ax.transAxes, ha='left', va='top', fontsize=8,
                bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8))

    # Color subplots by PSIS-LOO rank
    loo_values_array = np.array(row_loo_values)
    ranks = np.argsort(-loo_values_array)
    for j, _ in enumerate(inference_model_configs):
        ax = axes[i, j]
        rank = np.where(ranks == j)[0][0]
        normalized_rank = rank / (len(inference_model_configs) - 1)
        color = cmap(normalized_rank)
        ax.set_facecolor(color)

plt.show()
fig.savefig("POF_MCMC_v6_diagnostics_truedata.png", dpi=300)

# Save summaries
posterior_summary_df = pd.concat(posterior_summaries, ignore_index=True)
diagnostic_summary_df = pd.DataFrame(diagnostic_summaries)

posterior_summary_df.to_csv("POF_MCMC_v6_posteriors_truedata.csv", index=False)
diagnostic_summary_df.to_csv("POF_MCMC_v6_diagnostics_truedata.csv", index=False)
# %% Check summary
print(diagnostic_summary_df)
str_exclude = 'Y0|T_cont|alpha|gamma|u_v|tau|lam'
print(posterior_summary_df[posterior_summary_df['index'].str.contains(str_exclude) == False])
