# Version 9
# Problems version 8:
# - Performance: very slow (1 sec/sample)
# - MCMC got trapped due to high model coplexity

# solution:
# - reduce model complexity (removed country/project NB parameters)
# - rewrite as functions for better flexibility
# - implement loop over true data subsets

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

# %%
def model_POF(Y=None, T=None, D=None, Z=None, X=None,
              village_id=None, project_id=None, country_id=None,
              V=0, P=0, C=0, funcForm="linear"):

    N, Kz = Z.shape
    _, Kx = X.shape

    # ------------------------------
    # Non-centered random effects
    # ------------------------------
    if V and (village_id is not None):
        sigma_v = numpyro.sample("sigma_v", dist.HalfNormal(.2))
        with numpyro.plate("village", V):
            u_v_raw = numpyro.sample("u_v_raw", dist.Normal(0., .2))
        u_v = sigma_v * u_v_raw
        v_eff = u_v[village_id]
    else:
        sigma_v = 0.0
        u_v_raw = jnp.zeros((0,))
        u_v = jnp.zeros((0,))
        v_eff = jnp.zeros(N)

    if P and (project_id is not None):
        sigma_p = numpyro.sample("sigma_p", dist.HalfNormal(.2))
        with numpyro.plate("project", P):
            u_p_raw = numpyro.sample("u_p_raw", dist.Normal(0., .2))
        u_p = sigma_p * u_p_raw
        p_eff = u_p[project_id]
    else:
        sigma_p = 0.0
        u_p_raw = jnp.zeros((0,))
        u_p = jnp.zeros((0,))
        p_eff = jnp.zeros(N)

    if C and (country_id is not None):
        sigma_c = numpyro.sample("sigma_c", dist.HalfNormal(.2))
        with numpyro.plate("country", C):
            u_c_raw = numpyro.sample("u_c_raw", dist.Normal(0., 0.2))
        u_c = sigma_c * u_c_raw
        c_eff = u_c[country_id]
    else:
        sigma_c = 0.0
        u_c_raw = jnp.zeros((0,))
        u_c = jnp.zeros((0,))
        c_eff = jnp.zeros(N)

    numpyro.deterministic("u_v", u_v)
    numpyro.deterministic("u_p", u_p)
    numpyro.deterministic("u_c", u_c)

    u_effect = v_eff + p_eff + c_eff

    # ------------------------------
    # Baseline outcome
    # ------------------------------
    alpha = numpyro.sample("alpha", dist.Normal(0., 1.).expand([Kx]))
    Y0 = jnp.dot(X, alpha)

    # ------------------------------
    # Placement indicator
    # ------------------------------
    with numpyro.plate("obs", N):
        numpyro.sample("D_obs", dist.Bernoulli(0.5), obs=D)

    # ------------------------------
    # Simplified NB2 dosage model
    # ------------------------------
    gamma_global = numpyro.sample("gamma_global", dist.Normal(0., 1.).expand([Kz]))
    gamma_for_obs = jnp.tile(gamma_global, (N, 1))

    omega_global = numpyro.sample("omega_global", dist.Normal(0., 1.))
    omega_for_obs = jnp.repeat(omega_global, N)

    mu_log_eta = numpyro.sample("mu_log_eta", dist.Normal(0., 1.))
    eta_for_obs = jnp.exp(mu_log_eta) * jnp.ones(N)

    # Linear predictor for NB mean
    logits = jnp.sum(Z * gamma_for_obs, axis=1) + omega_for_obs * D
    lam = jnp.exp(jnp.clip(logits, -20., 20.))

    # ------------------------------
    # Dose-response τ(T)
    # ------------------------------
    T = jnp.asarray(T).astype(jnp.float32)
    T_max = jnp.max(T)
    T_cont = jnp.where(T_max > 0, T / T_max, T)
    T_leg = 2.0 * T_cont - 1.0

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
    elif funcForm == "poly3":
        b1 = numpyro.sample("b1", dist.Normal(0., 2.))
        b2 = numpyro.sample("b2", dist.Normal(0., 2.))
        b3 = numpyro.sample("b3", dist.Normal(0., 2.))
        tau = b1 * T_cont + b2 * T_cont**2 + b3 * T_cont**3
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
        raise ValueError("Invalid funcForm")

    # ------------------------------
    # Observation-level plate for T_obs and Y_obs
    # ------------------------------
    sigma_Y = numpyro.sample("sigma_Y", dist.Exponential(1.0))
    with numpyro.plate("obs", N):
        numpyro.sample("T_obs", dist.NegativeBinomial2(lam, eta_for_obs), obs=T)
        numpyro.sample("Y_obs", dist.Normal(Y0 + tau + u_effect, sigma_Y), obs=Y)

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

    # --- Simplified NB dosage model truths (no country/project random effects)
    gamma_global_true = jnp.array([0.9, -0.05, 0.04, 0.03, -0.02])
    omega_global_true = 1.0
    mu_log_eta_true = 1.0  # log(2.718) ~ 1.0, so eta_base ~ 2.7

    coefTrue['gamma_global'] = np.array(gamma_global_true)
    coefTrue['omega_global'] = float(omega_global_true)
    coefTrue['mu_log_eta'] = float(mu_log_eta_true)

    gamma_for_obs_true = jnp.tile(gamma_global_true, (N, 1))  # (N, K)
    omega_for_obs_true = jnp.repeat(omega_global_true, N)     # (N,)
    eta_for_obs_true = jnp.exp(mu_log_eta_true) * jnp.ones(N) # (N,)

    # --- true NB mean
    logits_true = jnp.sum(Z * gamma_for_obs_true, axis=1) + omega_for_obs_true * D  # (N,)
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
        b1 = 0.6
        tau_true = b1 * T_cont
        coefTrue.update({'b1': b1})
    elif funcForm == "poly2":
        b1, b2 = 1.8, -1.3
        tau_true = T_cont*b1 + b2*T_cont**2
        coefTrue.update({'b1': b1, 'b2': b2})
    elif funcForm == "poly2_legendre":
        b_true = -0.6
        P1, P2 = T_leg, 0.5*(3*T_leg**2 - 1)
        tau_true = b_true * (P1 + P2)
        coefTrue.update({'b': b_true})
    elif funcForm == "poly3":
        b1, b2, b3 = -2.3, 7.5, -4.8
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
    coefTrue.update({'sigma_Y': float(sigma_Y_true)})

    # --- kwargs for model (fit on standardized Y; pass group sizes to activate plates)
    model_kwargs = {
        "Z": Z, "X": X, "T": T_counts, "D": D, "Y": Y_unscaled,
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
N = 3080
K = 5
V = 115
P = 17
C = 6

# parameters modelling/plotting
num_tau_draws = 200
n_warmup = 200
n_samples = 500
n_chains = 2

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
func_forms = ['homogenous', 'linear', 'exponential', 'sigmoid', 'poly2', 'poly3']
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
                    v_post_mean_display = float(v_post_mean)
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
    tau_post_samples_rescaled = tau_post_samples

    # --- plot posterior predictive tau draws
    plt.figure(figsize=(7,4))
    num_draws = min(num_tau_draws, int(tau_post_samples_rescaled.shape[0]))
    for i in np.random.choice(int(tau_post_samples_rescaled.shape[0]), num_draws, replace=False):
        plt.plot(x_range, np.asarray(tau_post_samples_rescaled[i]), alpha=0.05, color='blue')

    # --- true tau without cluster effects
    tau_true_grid = tau_true_on_grid(jnp.asarray(x_range), coefTrue, funcForm)
    plt.plot(x_range, np.asarray(tau_true_grid), color='red', lw=2, label='True tau')

    plt.title(f"Predicted posterior tau (zero clusters) vs True tau ({funcForm})")
    plt.xlabel("T")
    plt.ylabel("tau (Y-scale)")
    plt.legend()
    plt.show()



print(diagnostic_summary_df)
str_exclude = 'Y0|T_cont|alpha|gamma|u_|tau|lam|sigma|eta|omega'
print(posterior_summary_df[posterior_summary_df['index'].str.contains(str_exclude) == False])




# %% Functions
def load_and_merge_data(short_path, medium_path):
    df_short = pd.read_csv(short_path)
    df_medium = pd.read_csv(medium_path)

    # tags
    df_short["dataset_tag"] = "short_full"
    df_medium["dataset_tag"] = "medium"
    df_short.loc[df_short["Project_code_desc"].isin(df_medium["Project_code_desc"]), "dataset_tag"] = "short_present_in_medium"

    # merge
    df_all = pd.concat([df_short, df_medium], ignore_index=True)

    return df_all

def preprocess(df, X_names, Z_names, Y_name, D_name, T_name,
               village_col, project_col, country_col):
    relevant_cols = [Y_name] + X_names + Z_names + [D_name, T_name]
    df = df.dropna(subset=relevant_cols).copy()

    # Standardize X
    X = df[X_names].to_numpy()
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    X = np.column_stack((X, np.ones(X.shape[0])))

    # Standardize Z (safe against zero variance)
    Z = df[Z_names].to_numpy()
    Z_std = Z.std(axis=0)
    Z = (Z - Z.mean(axis=0)) / np.where(Z_std == 0, 1, Z_std)
    Z = np.column_stack((Z, np.ones(Z.shape[0])))

    Y = df[Y_name].to_numpy()
    D = df[D_name].to_numpy()
    T = df[T_name].to_numpy().astype(int)

    village_id = df[village_col].astype('category').cat.codes.to_numpy().astype(np.int32)
    project_id = df[project_col].astype('category').cat.codes.to_numpy().astype(np.int32)
    country_id = df[country_col].astype('category').cat.codes.to_numpy().astype(np.int32)

    return {
        "X": X, "Z": Z, "Y": Y, "D": D, "T": T,
        "village_id": village_id,
        "project_id": project_id,
        "country_id": country_id,
        "V": len(np.unique(village_id)),
        "P": len(np.unique(project_id)),
        "C": len(np.unique(country_id)),
    }


def run_mcmc(data_dict, funcForm, rng_key, n_warmup, n_sample, n_chains):
    mcmc_args = {**data_dict, "funcForm": funcForm}

    kernel = NUTS(model_POF)
    mcmc = MCMC(kernel, num_warmup=n_warmup, num_samples=n_sample, 
                num_chains=n_chains, progress_bar=True)

    rng_key, rng_key_mcmc = random.split(rng_key)
    mcmc.run(rng_key_mcmc, **mcmc_args)
    samples = mcmc.get_samples()

    # Posterior predictive for tau
    predictivePosterior = Predictive(model_POF, posterior_samples=samples)
    x_range = np.linspace(np.percentile(data_dict["T"], 0.1),
                          np.percentile(data_dict["T"], 99), 100)
    ppc_args = {
        "Z": jnp.repeat(jnp.mean(data_dict["Z"], axis=0, keepdims=True), 100, axis=0),
        "D": jnp.ones(100),
        "X": jnp.repeat(jnp.mean(data_dict["X"], axis=0, keepdims=True), 100, axis=0),
        "T": jnp.asarray(x_range),
        "village_id": None, "project_id": None, "country_id": None,
        "V": 0, "P": 0, "C": 0, "funcForm": funcForm
    }
    rng_key, rng_key_ppc = random.split(rng_key)
    post_predict = predictivePosterior(rng_key_ppc, **ppc_args)

    # arviz conversion
    ppc_obs = predictivePosterior(rng_key, **mcmc_args)
    idata = az.from_numpyro(
        mcmc,
        posterior_predictive=ppc_obs,
        coords={"obs_id": np.arange(len(data_dict["Y"]))},
        dims={"Y_obs": ["obs_id"]}
    )

    return {
        "samples": samples,
        "idata": idata,
        "tau_post_samples": post_predict["tau"],
        "x_range": x_range
    }, rng_key

def run_analysis(df_all, X_names, Z_names, outcomes, D_name, T_name,
                 village_col, project_col, country_col, split_by="Country_code",
                 rng_key=None):

    inference_models = [
        ('homogenous', 'Homogenous'),
        #('linear', 'Linear'),
        #('exponential', 'Exponential'),
        #('sigmoid', 'Sigmoid'),
        #('poly2', 'Poly2'),
        ('poly3', 'Poly3 (Standard)')
    ]

    results = {}
    for Y_name in outcomes:
        print(f"Analyzing outcome: {Y_name}")
        grouped = df_all.groupby(split_by)
        results[Y_name] = {}

        for group, df_subset in grouped:
            print(f" Processing group: {group} with {len(df_subset)} observations")
            data_dict = preprocess(df_subset, X_names, Z_names, Y_name, D_name, T_name,
                                   village_col, project_col, country_col)
            if len(data_dict["Y"]) < 50 or data_dict["V"] < 5:
                continue

            group_res = {}
            for inf_func, title in inference_models:
                print(f"  Running model: {title}")
                res, rng_key = run_mcmc(data_dict, inf_func, rng_key, n_warmup, n_sample, n_chains)
                group_res[inf_func] = {**res, "title": title}
            results[Y_name][group] = group_res

    return results
# %%
def plot_results(results, split_by="Country_code", n_draws=50, ylim=(-0.25, 0.5)):
    for outcome, groups in results.items():
        n_groups = len(groups)
        n_models = len(next(iter(groups.values())))  # same models for all groups

        fig, axes = plt.subplots(n_groups, n_models, 
                                 figsize=(4*n_models, 3*n_groups),
                                 squeeze=False, tight_layout=True)
        fig.suptitle(f"Dose-Response Function — Outcome: {outcome}", fontsize=18, y=1.02)

        for i, (group, models) in enumerate(groups.items()):
            for j, (funcForm, res) in enumerate(models.items()):
                ax = axes[i, j]

                # Posterior spaghetti
                tau_post = res["tau_post_samples"]
                x_range = res["x_range"]
                draws = min(n_draws, tau_post.shape[0])
                for idx in range(draws):
                    ax.plot(x_range, tau_post[idx], color="k", alpha=0.05)

                # Diagnostics
                loo = az.loo(res["idata"], pointwise=False, var_name="Y_obs")
                waic = az.waic(res["idata"], pointwise=False, var_name="Y_obs")

                ax.set_xlim([x_range.min(), x_range.max()])
                ax.set_ylim(ylim)
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)

                # Add red dashed line at y=0
                ax.axhline(0, color='red', linestyle='--', linewidth=1)

                if i == n_groups - 1:
                    ax.set_xlabel("T")
                if j == 0:
                    ax.set_ylabel(f"{group}\nTau (Y-scale)")
                if i == 0:
                    ax.set_title(res["title"], fontsize=12)

                ax.text(0.05, 0.95, f"LOO: {loo.elpd_loo:.2f}\nSE: {loo.se:.2f}",
                        transform=ax.transAxes, ha="left", va="top", fontsize=8,
                        bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8))

        plt.show()
        fig.savefig(f"POF_grid_{outcome}.png", dpi=300)

# %% MCMC parameters
n_warmup = 500
n_sample = 1000
n_chains = 2

X_names = ["forest_share"]
Z_names = ["Year_formed","Hh_ethnic","Hh_own_out", "hh_head_age",
             "hh_dependency_ratio","land_tot","total_asset_value","agric_salary",
             "forest_share","int_nonREDD_ongoing_sum","int_nonREDD_ended_sum"]

# %% Real data
df_all = load_and_merge_data(
    "C:/Users/DaSchulz/OneDrive - European Forest Institute/Dokumente/research/redd_intensity/data//pd_short_ALL.csv",
    "C:/Users/DaSchulz/OneDrive - European Forest Institute/Dokumente/research/redd_intensity/data/pd_medium_ALL.csv"
)

# subset to medium
df_medium = df_all[df_all["dataset_tag"] == "medium"].copy()

results = run_analysis(
    df_medium,
    X_names=X_names,
    Z_names=Z_names,
    outcomes=["DY_forest_share"],
    D_name="Village_Type",
    T_name="int_cat_tot_involALL",
    village_col="Village",
    project_col="Project_code",
    country_col="Country_code_desc",
    split_by="Country_code_desc",
    rng_key=rng_key
)

plot_results(results, split_by="Country_code_desc")


# %%

# subset to short (all)
df_short = df_all[df_all["dataset_tag"] == "short_full"].copy()

results = run_analysis(
    df_short,
    X_names=X_names,
    Z_names=Z_names,
    outcomes=["DY_forest_share"],
    D_name="Village_Type",
    T_name="int_cat_tot_involALL",
    village_col="Village",
    project_col="Project_code",
    country_col="Country_code_desc",
    split_by="Country_code_desc",
    rng_key=rng_key
)

plot_results(results, split_by="Country_code_desc")

# %%
# subset to short (also in medium)
df_short_restr = df_all[df_all["dataset_tag"] == "short_present_in_medium"].copy()

results = run_analysis(
    df_short_restr,
    X_names=X_names,
    Z_names=Z_names,
    outcomes=["DY_forest_share"],
    D_name="Village_Type",
    T_name="int_cat_tot_involALL",
    village_col="Village",
    project_col="Project_code",
    country_col="Country_code_desc",
    split_by="Country_code_desc",
    rng_key=rng_key
)

plot_results(results, split_by="Country_code_desc")

# %%
# subset to short (also in medium) but by project
df_short_restr = df_all[df_all["dataset_tag"] == "short_present_in_medium"].copy()

results = run_analysis(
    df_short_restr,
    X_names=X_names,
    Z_names=Z_names,
    outcomes=["DY_forest_share"],
    D_name="Village_Type",
    T_name="int_cat_tot_involALL",
    village_col="Village",
    project_col="Project_code",
    country_col="Country_code_desc",
    split_by="Project_code_desc",
    rng_key=rng_key
)

plot_results(results, split_by="Project_code_desc")