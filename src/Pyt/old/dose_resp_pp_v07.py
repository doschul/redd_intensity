# Version 7
# Problems version 6:
# - Single neg binomial spread parameter
# - single neg binom size-shift parameter
# - Impact of Z on outcome (covariate definition)
# - Y-standardization not working as expected
# - RuntimeWarning: invalid value encountered in scalar divide

# Solutions:
# - Model neg binomial baseline parameter per country
# - Model neg binom shift parameter per project
# - model neg binom precision per project
# - include Z in X for baseline intercepts
# - drop Y-standardization in DGP (recover unscaled estimate)
# - fix RuntimeWarning: invalid value encountered in scalar divide

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
def model_POF(Y=None, T=None, D=None, Z=None, X=None,
              village_id=None, project_id=None, country_id=None,
              V=0, P=0, C=0, funcForm="linear"):

    # ------------------------------
    # Non-centered random effects
    # ------------------------------
    if V and (village_id is not None):
        sigma_v = numpyro.sample("sigma_v", dist.HalfNormal(1.0))
        u_v_raw = numpyro.sample("u_v_raw", dist.Normal(0, 1).expand([V]))
        u_v = sigma_v * u_v_raw
        v_eff = u_v[village_id]
    else:
        sigma_v = 0.0
        u_v_raw = jnp.zeros((0,))
        u_v = jnp.zeros((0,))
        eff_v = jnp.zeros(N)

    if P and (project_id is not None):
        sigma_p = numpyro.sample("sigma_p", dist.HalfNormal(1.0))
        u_p_raw = numpyro.sample("u_p_raw", dist.Normal(0, 1).expand([P]))
        u_p = sigma_p * u_p_raw
        p_eff = u_p[project_id]
    else:
        sigma_p = 0.0
        u_p_raw = jnp.zeros((0,))
        u_p = jnp.zeros((0,))
        p_eff = 0.0

    if C and (country_id is not None):
        sigma_c = numpyro.sample("sigma_c", dist.HalfNormal(1.0))
        u_c_raw = numpyro.sample("u_c_raw", dist.Normal(0, 1).expand([C]))
        u_c = sigma_c * u_c_raw
        c_eff = u_c[country_id]
    else:
        sigma_c = 0.0
        u_c_raw = jnp.zeros((0,))
        u_c = jnp.zeros((0,))
        c_eff = 0.0

    numpyro.deterministic("u_v", sigma_v * u_v_raw)
    numpyro.deterministic("u_p", sigma_p * u_p_raw)
    numpyro.deterministic("u_c", sigma_c * u_c_raw)

    u_effect = v_eff + p_eff + c_eff

    # ------------------------------
    # Dimensions
    # ------------------------------
    N, Kz = Z.shape
    _, Kx = X.shape

    # ------------------------------
    # Baseline outcome (Y on standardized scale as per your setup)
    # ------------------------------
    alpha = numpyro.sample("alpha", dist.Normal(0., 1.).expand([Kx]))
    Y0 = jnp.dot(X, alpha)

    # ------------------------------
    # Placement indicator (observed)
    # ------------------------------
    numpyro.sample("D_obs", dist.Bernoulli(0.5), obs=D)

    # ------------------------------
    # Hierarchical Negative Binomial dosage model
    #   - gamma varies by country (Kz-vector per country)
    #   - omega varies by project (scalar per project)
    #   - eta varies by project × D level (two values per project)
    # ------------------------------
    # gamma per country
    gamma_global = numpyro.sample("gamma_global", dist.Normal(0., 1.).expand([Kz]))
    if C and (country_id is not None):
        sigma_gamma_c = numpyro.sample("sigma_gamma_c", dist.HalfNormal(1.0))
        gamma_off = numpyro.sample("gamma_off", dist.Normal(0., 1.).expand([C, Kz]))
        gamma_c = gamma_global + sigma_gamma_c * gamma_off            # (C, Kz)
        gamma_for_obs = gamma_c[country_id]                           # (N, Kz)
    else:
        gamma_for_obs = jnp.broadcast_to(gamma_global, (N, Kz))       # (N, Kz)

    # omega per project
    omega_global = numpyro.sample("omega_global", dist.Normal(0., 1.))
    if P and (project_id is not None):
        sigma_omega_p = numpyro.sample("sigma_omega_p", dist.HalfNormal(1.0))
        omega_off = numpyro.sample("omega_off", dist.Normal(0., 1.).expand([P]))
        omega_p = omega_global + sigma_omega_p * omega_off            # (P,)
        omega_for_obs = omega_p[project_id]                           # (N,)
    else:
        omega_for_obs = jnp.repeat(omega_global, N)                   # (N,)

    # eta per project × D (work on log scale for positivity)
    mu_log_eta = numpyro.sample("mu_log_eta", dist.Normal(0., 1.))
    if P and (project_id is not None):
        sigma_log_eta_pd = numpyro.sample("sigma_log_eta_pd", dist.HalfNormal(1.0))
        eta_off = numpyro.sample("eta_off", dist.Normal(0., 1.).expand([P, 2]))
        log_eta_pd = mu_log_eta + sigma_log_eta_pd * eta_off              # (P, 2)
        # D is {0,1}; ensure int index
        D_idx = D.astype(jnp.int32)
        eta_for_obs = jnp.exp(log_eta_pd[project_id, D_idx])          # (N,)
    else:
        eta_for_obs = jnp.exp(mu_log_eta) * jnp.ones(N)               # (N,)

    # Linear predictor for NB mean
    # batched dot: sum over Kz
    logits = jnp.sum(Z * gamma_for_obs, axis=1) + omega_for_obs * D   # (N,)
    lam = jnp.exp(jnp.clip(logits, -20., 20.))                        # (N,)

    # NB2 with mean=lam and dispersion=eta_for_obs
    numpyro.sample("T_obs", dist.NegativeBinomial2(lam, eta_for_obs), obs=T)

    # ------------------------------
    # Scale T to [0,1] internally for dose-response
    # ------------------------------
    T = jnp.asarray(T).astype(jnp.float32)
    T_max = jnp.max(T)
    T_cont = jnp.where(T_max > 0, T / T_max, T)   # safe divide
    T_leg = 2.0 * T_cont - 1.0

    # ------------------------------
    # Dose-response τ(T)
    # ------------------------------
    if funcForm == "homogenous":
        b1 = numpyro.sample("b1", dist.Normal(0., 1.))
        tau = b1 * D

    elif funcForm == "linear":
        b1 = numpyro.sample("b1", dist.Normal(0., 1.))
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
        b2 = numpyro.sample("b2", dist.LogNormal(0., 0.5))
        tau = b1 * (1.0 - jnp.exp(-b2 * T_cont))

    elif funcForm == "sigmoid":
        b1 = numpyro.sample("b1", dist.Normal(0., 1.))
        b2 = numpyro.sample("b2", dist.Exponential(0.02))
        b3 = numpyro.sample("b3", dist.Uniform(0., 1.))
        tau = b1 * (1.0 / (1.0 + jnp.exp(-b2 * (T_cont - b3))))

    else:
        raise ValueError("Invalid funcForm specified")

    # ------------------------------
    # Outcome model
    # ------------------------------
    sigma_Y = numpyro.sample("sigma_Y", dist.Exponential(1.0))
    Y_mean = Y0 + tau + u_effect
    numpyro.sample("Y_obs", dist.Normal(Y_mean, sigma_Y), obs=Y)

    # Deterministic for plotting
    numpyro.deterministic("tau", tau)



# %% DGP
def data_generating(rng_key, funcForm="linear", N=1000, K=5, V=20, P=10, C=5):
    rng_key, kX, kZ, kD, kT, kVill, kProj, kCoun, kv, kp, kc, kgc, kow, kep = random.split(rng_key, 14)

    # --- covariates
    X = jnp.hstack([jnp.ones((N,1)), random.normal(kX,(N,K-1))])
    Z = jnp.hstack([jnp.ones((N,1)), random.normal(kZ,(N,K-1))])

    coefTrue = {}

    # --- village ids (non-centered)
    if V is None or V == 0:
        V = 1
        village_id = jnp.zeros(N, dtype=jnp.int32)
        sigma_v_true = 0.0
        village_u = jnp.zeros((V,), dtype=jnp.float32)
    else:
        village_id = random.randint(kVill, shape=(N,), minval=0, maxval=V)
        sigma_v_true = 0.8
        u_v_raw = dist.Normal(0., 1.).sample(kv, (V,))
        village_u = sigma_v_true * u_v_raw
    u_effect_v = village_u[village_id]
    coefTrue['sigma_v'] = float(sigma_v_true)

    # --- project ids (non-centered)
    if P is None or P == 0:
        P = 1
        project_id = jnp.zeros(N, dtype=jnp.int32)
        sigma_p_true = 0.0
        project_u = jnp.zeros((P,), dtype=jnp.float32)
    else:
        project_id = random.randint(kProj, shape=(N,), minval=0, maxval=P)
        sigma_p_true = 0.5
        u_p_raw = dist.Normal(0., 1.).sample(kp, (P,))
        project_u = sigma_p_true * u_p_raw
    u_effect_p = project_u[project_id]
    coefTrue['sigma_p'] = float(sigma_p_true)

    # --- country ids (non-centered)
    if C is None or C == 0:
        C = 1
        country_id = jnp.zeros(N, dtype=jnp.int32)
        sigma_c_true = 0.0
        country_u = jnp.zeros((C,), dtype=jnp.float32)
    else:
        country_id = random.randint(kCoun, shape=(N,), minval=0, maxval=C)
        sigma_c_true = 0.3
        u_c_raw = dist.Normal(0., 1.).sample(kc, (C,))
        country_u = sigma_c_true * u_c_raw
    u_effect_c = country_u[country_id]
    coefTrue['sigma_c'] = float(sigma_c_true)

    # --- total random effects
    u_effect = u_effect_v + u_effect_p + u_effect_c

    # --- placement D
    D = dist.Bernoulli(0.5).sample(key=kD, sample_shape=(N,)).astype(jnp.float32)

    # --- Hierarchical NB dosage model truths
    # gamma varies by country (vector per country)
    gamma_global_true = jnp.array([0.9, -0.05, 0.04, 0.03, -0.02])
    sigma_gamma_c_true = 0.10
    gamma_off_true = dist.Normal(0., 1.).sample(kgc, (C, K))
    gamma_c_true = gamma_global_true + sigma_gamma_c_true * gamma_off_true
    gamma_for_obs_true = gamma_c_true[country_id]     # (N, K)
    coefTrue['gamma_global'] = np.array(gamma_global_true)
    coefTrue['sigma_gamma_c'] = float(sigma_gamma_c_true)

    # omega varies by project (scalar per project)
    omega_global_true = 1.0
    sigma_omega_p_true = 0.20
    omega_off_true = dist.Normal(0., 1.).sample(kow, (P,))
    omega_p_true = omega_global_true + sigma_omega_p_true * omega_off_true
    omega_for_obs_true = omega_p_true[project_id]     # (N,)
    coefTrue['omega_global'] = float(omega_global_true)
    coefTrue['sigma_omega_p'] = float(sigma_omega_p_true)

    # eta varies by project × D (positive; work on log scale)
    eta_base_true = 2.5
    sigma_log_eta_pd_true = 0.25
    eta_off_true = dist.Normal(0., 1.).sample(kep, (P, 2))
    log_eta_pd_true = jnp.log(eta_base_true) + sigma_log_eta_pd_true * eta_off_true  # (P,2)
    D_idx = D.astype(jnp.int32)
    eta_for_obs_true = jnp.exp(log_eta_pd_true[project_id, D_idx])                   # (N,)
    coefTrue['eta_base'] = float(eta_base_true)
    coefTrue['sigma_log_eta_pd'] = float(sigma_log_eta_pd_true)

    # --- true NB mean
    logits_true = jnp.sum(Z * gamma_for_obs_true, axis=1) + omega_for_obs_true * D
    lam_true = jnp.exp(jnp.clip(logits_true, -20., 20.))
    T_counts = dist.NegativeBinomial2(lam_true, eta_for_obs_true).sample(key=kT)

    # --- scaled version of T for tau (as in your model)
    T_counts_f = T_counts.astype(jnp.float32)
    T_max = jnp.max(T_counts_f)
    T_cont = jnp.where(T_max > 0, T_counts_f / T_max, T_counts_f)
    T_leg = 2.0 * T_cont - 1.0

    # --- true tau (without random effects)
    if funcForm == "homogenous":
        b1 = 0.5
        tau_true = b1 * D
        coefTrue.update({'b1': b1})
    elif funcForm == "linear":
        b1 = 0.8
        tau_true = b1 * T_cont
        coefTrue.update({'b1': b1})
    elif funcForm == "poly2":
        b1, b2 = 1.8, -0.9
        tau_true = T_cont*b1 + b2*T_cont**2
        coefTrue.update({'b1': b1, 'b2': b2})
    elif funcForm == "poly2_legendre":
        b_true = -0.6
        P1, P2 = T_leg, 0.5*(3*T_leg**2 - 1)
        tau_true = b_true * (P1 + P2)
        coefTrue.update({'b': b_true})
    elif funcForm == "poly3":
        b1, b2, b3 = -0.5, 1.7, -1.0
        tau_true = T_cont*b1 + b2*T_cont**2 + b3*T_cont**3
        coefTrue.update({'b1': b1, 'b2': b2, 'b3': b3})
    elif funcForm == "poly3_legendre":
        a1, a2 = 0.53, -0.3
        P1 = T_leg
        P2 = 0.5 * (3.0 * T_leg**2 - 1.0)
        P3 = 0.5 * (5.0 * T_leg**3 - 3.0 * T_leg)
        tau_true = a1 * (P1 + P2) + a2 * (P2 + P3)
        coefTrue.update({'a1': a1, 'a2': a2})
    elif funcForm == "exponential":
        b1 = 0.6
        b2 = 1.0
        tau_true = b1 * (1 - jnp.exp(-b2 * T_cont))
        coefTrue.update({'b1': b1, 'b2': b2})
    elif funcForm == "sigmoid":
        b1, b2, b3 = 0.6, 50.0, 0.4
        tau_true = b1 * (1/(1 + jnp.exp(-b2*(T_cont - b3))))
        coefTrue.update({'b1': b1, 'b2': b2, 'b3': b3})
    else:
        raise ValueError("Invalid funcForm")

    # --- baseline outcome
    alpha_true = jnp.ones(K)
    Y0 = jnp.dot(X, alpha_true)

    # --- observed Y (unscaled), then standardized (as you fit the model)
    sigma_Y_true = 0.1
    Y_unscaled = Y0 + tau_true + u_effect + dist.Normal(0., sigma_Y_true).sample(rng_key, (N,))
    Y_mean, Y_std = Y_unscaled.mean(), Y_unscaled.std()
    Y_scaled = (Y_unscaled - Y_mean)/Y_std

    # --- record truths
    coefTrue.update({
        'gamma_c_true': np.array(gamma_c_true),
        'omega_p_true': np.array(omega_p_true),
        'log_eta_pd_true': np.array(log_eta_pd_true),
        'sigma_Y': float(sigma_Y_true)
    })

    # --- kwargs for model (fit on standardized Y; pass group sizes to activate plates)
    model_kwargs = {
        "Z": Z, "X": X, "T": T_counts, "D": D, "Y": Y_scaled,
        "village_id": village_id, "project_id": project_id, "country_id": country_id,
        "V": int(V), "P": int(P), "C": int(C),
        "funcForm": funcForm
    }

    return (Y_scaled, Y_unscaled, Y0, T_counts, T_cont, D, Z, X,
            village_id, project_id, country_id,
            tau_true, T_leg, model_kwargs, coefTrue, Y_mean, Y_std)


# %% Parameter recovery

# You can change/extend this list freely
func_forms = ['homogenous', 'linear', 'exponential', 'sigmoid', 'poly2', 'poly3']

# Parameters data generation
N = 2500
K = 5
V = 250
P = 50
C = 10

# parameters modelling/plotting
num_tau_draws = 200
n_warmup = 100
n_samples = 200
n_chains = 1

parameter_recovery = []

def tau_true_on_grid(T_plot_raw, coefTrue, funcForm):
    """
    Compute the true tau(T) on a grid of raw T values (integer counts).
    Rescales T internally to [0,1] (T_cont) and [-1,1] (T_leg),
    consistent with the model.
    """
    T_cont = T_plot_raw / T_plot_raw.max()
    T_leg = 2.0 * T_cont - 1.0

    if funcForm == "homogenous":
        return coefTrue['b1'] * jnp.ones_like(T_cont)

    elif funcForm == "linear":
        return coefTrue['b1'] * T_cont

    elif funcForm == "poly2":
        return (coefTrue['b1'] * T_cont
                + coefTrue['b2'] * T_cont**2)

    elif funcForm == "poly2_legendre":
        P1 = T_leg
        P2 = 0.5 * (3.0 * T_leg**2 - 1.0)
        return coefTrue['b'] * (P1 + P2)

    elif funcForm == "poly3":
        return (coefTrue['b1'] * T_cont
                + coefTrue['b2'] * T_cont**2
                + coefTrue['b3'] * T_cont**3)

    elif funcForm == "poly3_legendre":
        P1 = T_leg
        P2 = 0.5 * (3.0 * T_leg**2 - 1.0)
        P3 = 0.5 * (5.0 * T_leg**3 - 3.0 * T_leg)
        return (coefTrue['a1'] * (P1 + P2)
                + coefTrue['a2'] * (P2 + P3))

    elif funcForm == "exponential":
        return coefTrue['b1'] * (1 - jnp.exp(-coefTrue['b2'] * T_cont))

    elif funcForm == "sigmoid":
        return coefTrue['b1'] * (
            1 / (1 + jnp.exp(-coefTrue['b2'] * (T_cont - coefTrue['b3'])))
        )

    else:
        raise ValueError("Invalid funcForm")

# %% Run parameter recovery for each function form

# --- loop over function forms
func_forms = ['linear', 'exponential', 'sigmoid', 'poly2', 'poly3']
parameter_recovery = []

for funcForm in func_forms:
    rng_key, rng_data = random.split(rng_key)

    # --- generate data
    (Y_scaled, Y_unscaled, Y0, T_counts, T_cont, D, Z, X,
            village_id, project_id, country_id,
            tau_true, T_leg, model_kwargs, coefTrue, Y_mean, Y_std) = \
        data_generating(rng_data, funcForm=funcForm, N=N, K=K, V=V, P=P, C=C)

    # --- run MCMC
    kernel = NUTS(model_POF)
    mcmc = MCMC(kernel, num_warmup=n_warmup, num_samples=n_samples, num_chains=n_chains)
    rng_key, rng_mcmc = random.split(rng_key)
    mcmc.run(rng_mcmc, **model_kwargs)
    samples = mcmc.get_samples()

    # --- parameter recovery printout
    print(f"\nPosterior summary for {funcForm}:")
    recovered = {}

    # handle scalar params
    for k, v_true in coefTrue.items():
        # --- check if it's a random-effect sigma ---
        if k.startswith("sigma_"):
            # posterior samples are positive; compare posterior mean
            v_post_mean = jnp.mean(samples[k])
            print(f"{k}: true={v_true:.3f}, posterior mean={float(v_post_mean):.3f}")
            recovered[k] = (v_true, float(v_post_mean))
        else:
            if k in samples:
                v_post_mean = jnp.mean(samples[k])
                if funcForm == "exponential" and k == "b1":
                    # rescale amplitude back to original Y scale
                    v_post_mean_display = float(v_post_mean * Y_std)
                else:
                    v_post_mean_display = float(v_post_mean)
                print(f"{k}: true={v_true}, posterior mean={v_post_mean_display:.3f}")
                recovered[k] = (v_true, v_post_mean_display)
            elif k == 'gamma' and 'gamma' in samples:
                post_mean = jnp.mean(samples['gamma'], axis=0)
                print(f"gamma: true={np.array(v_true)}, posterior mean={np.array(post_mean)}")
                recovered['gamma'] = (np.array(v_true), np.array(post_mean))
            else:
                print(f"{k}: true={v_true} (no direct sample found)")
                recovered[k] = (v_true, None)

    parameter_recovery.append((funcForm, recovered))

    # --- T grid
    #T_plot = jnp.linspace(0, 1, 100)
    #T_leg_plot = 2.0 * T_plot - 1.0

    # --- posterior predictive samples (zero cluster effects)
    pps = Predictive(model_POF, posterior_samples=samples)

    # For τ(T) plotting
    x_percentile = np.percentile(T_counts, q=[0.1, 99])
    x_range = np.linspace(x_percentile[0], x_percentile[1], 100)
    
    ppc_args = {
        'Z': jnp.repeat(jnp.mean(Z, axis=0, keepdims=True), 100, axis=0),
        'D': jnp.ones(100),
        'X': jnp.repeat(jnp.mean(X, axis=0, keepdims=True), 100, axis=0),
        'T': jnp.asarray(x_range),   # raw counts
        'village_id': None, 'project_id': None, 'country_id': None,
        'V': 0, 'P': 0, 'C': 0,
        'funcForm': funcForm
    }
    
    rng_key, rng_key_tau = random.split(rng_key)
    post_predict = pps(rng_key_tau, **ppc_args)
    tau_post_samples = post_predict['tau']
    # tau_post_samples has shape [num_samples, len(x_range)]
    tau_post_samples_rescaled = tau_post_samples * Y_std

    # --- plot posterior predictive tau draws
    plt.figure(figsize=(7,4))
    num_draws = min(num_tau_draws, int(tau_post_samples_rescaled.shape[0]))
    for i in np.random.choice(int(tau_post_samples_rescaled.shape[0]), num_draws, replace=False):
        plt.plot(x_range, np.asarray(tau_post_samples_rescaled[i]), alpha=0.05, color='blue')

    # --- true tau without cluster effects
    tau_true_grid = tau_true_on_grid(jnp.asarray(x_range), coefTrue, funcForm)
    plt.plot(x_range, np.asarray(tau_true_grid), color='red', lw=2, label='True tau')

    plt.title(f"Predicted posterior tau (zero clusters) vs True tau ({funcForm})")
    plt.xlabel("T (scaled to [0,1])")
    plt.ylabel("tau (standardized Y-scale)")
    plt.legend()
    plt.show()


# %%
predictivePosterior = Predictive(model_POF, posterior_samples=samples)

# Inspect MCMC chains
rng_key, rng_key_ppc = random.split(rng_key)
ppc_loo = predictivePosterior(rng_key_ppc, **model_kwargs)
idata = az.from_numpyro(
    mcmc,
    posterior_predictive=ppc_loo,
    coords={"obs_id": np.arange(len(Y_scaled))},
    dims={"Y_obs": ["obs_id"]}
)

az.plot_trace(idata)
# %% Dignose Funnel
# assume `idata` is your InferenceData from az.from_numpyro
posterior = idata.posterior

# pick one random effect, e.g. first village effect "u[0]"
u_level = 'u_v'
sigma_level = 'sigma_v'

u0 = posterior[u_level][:,:,0].values.flatten()     # shape: (chains*samples,)
sigma_u = posterior[sigma_level].values.flatten()

plt.figure(figsize=(6,5))
plt.scatter(u0, sigma_u, alpha=0.3)
plt.ylabel("sigma (population sd)")
plt.xlabel("u[0] (village effect)")
plt.title("Centered parameterization diagnostic")
plt.axhline(0, color="gray", ls="--")
plt.show()

# %%
# plot trace of u components only
u_components = ['u_p', 'u_v', 'u_c']
az.plot_trace(idata, var_names=u_components)
# %%
prior_predictive = Predictive(model_POF, num_samples=1000)
prior_samples = prior_predictive(random.PRNGKey(1))

u0_prior = prior_samples["u_v"][:,0]     # first village
sigma_v_prior = prior_samples["sigma_v"]

plt.figure(figsize=(6,5))
plt.scatter(u0_prior, sigma_v_prior, alpha=0.3)
plt.ylabel("sigma_v")
plt.xlabel("u_v[0]")
plt.title("Prior funnel diagnostic")
plt.axhline(0, color="gray", ls="--")
plt.show()
# %%
# run with real data


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
project_column = "Project_code"
country_column = "Country_code"

# make unique project id as country X project
df1[project_column] = df1[country_column].astype(str) + "_" + df1[project_column].astype(str)

# first standardize X, Z, and T within df1
df1[X_names] = (df1[X_names] - df1[X_names].mean()) / df1[X_names].std()
df1[Z_names] = (df1[Z_names] - df1[Z_names].mean()) / df1[Z_names].std()

# prepare for modeling
X = df1[X_names].to_numpy()
Z = df1[Z_names].to_numpy()
D = df1[D_name].to_numpy()
Y = df1[Y_name].to_numpy()
T_counts = df1[T_name].to_numpy().astype(int)
T_cont = (T_counts.astype(float) - T_counts.min()) / max(1, (T_counts.max() - T_counts.min()))

# Add constant column
X = np.column_stack((X, np.ones(X.shape[0])))
Z = np.column_stack((Z, np.ones(Z.shape[0])))

# prepare village FE
village_id = df1[village_column].astype('category').cat.codes
project_id = df1[project_column].astype('category').cat.codes
country_id = df1[country_column].astype('category').cat.codes
V = village_id.nunique()
P = project_id.nunique()
C = country_id.nunique()
village_id = village_id.to_numpy()
project_id = project_id.to_numpy()
country_id = country_id.to_numpy()

n_obs = len(Y)
print(f"Number of countries: {C}")
print(f"Number of projects: {P}")
print(f"Number of villages: {V}")
print(f"Number of observations: {n_obs}")


# %% Model Setup

# --- Parameters ---
n_plot = 200 # posterior spaghetti lines per subplot
n_warmup = 1000
n_sample = 2000
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

rng_key = random.PRNGKey(0)

# Use half of the 'rocket' colormap
full_cmap = sns.color_palette("rocket", as_cmap=True)
colors = full_cmap(np.linspace(0, 1, 256))
half_colors = colors[256//2:]
inverted_half_colors = half_colors[::-1]
cmap = plt.cm.colors.ListedColormap(inverted_half_colors)

# %% Plot real data posteriors

fig, axes = plt.subplots(1, len(inference_model_configs), figsize=(20, 5), tight_layout=True)
fig.suptitle('Dose-Response Function by Inference Model (One True DGP)', fontsize=24, y=1.02)

posterior_summaries = []
diagnostic_summaries = []
loo_values = []

for j, (inf_funcForm, figTitle) in enumerate(inference_model_configs):
    ax = axes[j]
    print(f'Estimating {inf_funcForm} model for true data...')

    mcmc_args = {
        'Z': Z,
        'D': D,
        'X': X,
        'Y': Y,
        'T': T_counts,
        'village_id': village_id,
        'project_id': project_id,
        'country_id': country_id,
        'V': V,
        'P': P,
        'C': C,
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
    x_percentile = np.percentile(T_counts, q=[0.1, 99])
    x_range = np.linspace(x_percentile[0], x_percentile[1], 100)
    x_mean = jnp.mean(X, axis=0)
    x_plot = jnp.repeat(x_mean.reshape(1, -1), 100, axis=0)
    z_mean = jnp.mean(Z, axis=0)
    z_plot = jnp.repeat(z_mean.reshape(1, -1), 100, axis=0)
    D_plot = jnp.ones(100)
    
    ppc_args = {
        'Z': z_plot,
        'D': D_plot,
        'X': x_plot,
        'T': x_range,
        'village_id': None,
        'project_id': None,
        'country_id': None,
        'V': 0, 'P': 0, 'C': 0,
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
    ppc_loo = predictivePosterior(rng_key_ppc, **mcmc_args)
    idata = az.from_numpyro(
        mcmc,
        posterior_predictive=ppc_loo,
        coords={"obs_id": np.arange(len(Y))},
        dims={"Y_obs": ["obs_id"]}
    )

    posterior_summary_tmp = az.summary(idata, round_to=4).reset_index()
    posterior_summary_tmp.insert(0, 'DGP_Form', 'Real data')
    posterior_summary_tmp.insert(1, 'Inference_Form', inf_funcForm)
    posterior_summaries.append(posterior_summary_tmp)

    # PSIS-LOO & WAIC
    loo_res = az.loo(idata, pointwise=True, var_name='Y_obs')
    waic_res = az.waic(idata, pointwise=True, var_name='Y_obs')
    diagnostic_summaries.append({
        'DGP_Form': 'Real data',
        'Inference_Form': inf_funcForm,
        "LOO": round(loo_res.elpd_loo.item(), 4),
        "LOO_SE": round(loo_res.se.item(), 4),
        "WAIC": round(waic_res.elpd_waic.item(), 4),
        "WAIC_SE": round(waic_res.se.item(), 4)
    })
    loo_values.append(loo_res.elpd_loo.item())

    # Axis formatting
    ax.set_xlim([x_range.min(), x_range.max()])
    ax.set_ylim([-0.25, 0.5])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)        
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title(figTitle, fontsize=14)
    ax.text(0.05, 0.95, f"PSIS-LOO: {loo_res.elpd_loo:.2f}\nSE: {loo_res.se:.2f}",
            transform=ax.transAxes, ha='left', va='top', fontsize=8,
            bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8))

# --- Color subplots by PSIS-LOO rank
loo_values_array = np.array(loo_values)
ranks = np.argsort(-loo_values_array)
for j, _ in enumerate(inference_model_configs):
    ax = axes[j]
    rank = np.where(ranks == j)[0][0]
    normalized_rank = rank / (len(inference_model_configs) - 1)
    color = cmap(normalized_rank)
    ax.set_facecolor(color)

plt.show()
fig.savefig("POF_MCMC_v7_diagnostics_truedata.png", dpi=300)

# Save summaries
posterior_summary_df = pd.concat(posterior_summaries, ignore_index=True)
diagnostic_summary_df = pd.DataFrame(diagnostic_summaries)

posterior_summary_df.to_csv("POF_MCMC_v7_posteriors_truedata.csv", index=False)
diagnostic_summary_df.to_csv("POF_MCMC_v7_diagnostics_truedata.csv", index=False)
# %% Check summary
print(diagnostic_summary_df)
str_exclude = 'Y0|T_cont|alpha|gamma|u_|tau|lam|sigma|eta|omega'
print(posterior_summary_df[posterior_summary_df['index'].str.contains(str_exclude) == False])

# %%
# Inspect MCMC chains

az.plot_trace(idata)
# %% Dignose Funnel
# assume `idata` is your InferenceData from az.from_numpyro
posterior = idata.posterior

# pick one random effect, e.g. first village effect "u[0]"
u_level = 'u_v'
sigma_level = 'sigma_v'

u0 = posterior[u_level][:,:,0].values.flatten()     # shape: (chains*samples,)
sigma_u = posterior[sigma_level].values.flatten()

plt.figure(figsize=(6,5))
plt.scatter(u0, sigma_u, alpha=0.3)
plt.ylabel("sigma (population sd)")
plt.xlabel("u[0] (village effect)")
plt.title("Centered parameterization diagnostic")
plt.axhline(0, color="gray", ls="--")
plt.show()
