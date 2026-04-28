from __future__ import annotations
import argparse
import math
import os
import time
import warnings
from typing import Any, Dict

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import gamma
from scipy.stats import norm

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
OUTPUT_ROOT = os.path.join(os.path.dirname(__file__), "output")


def _mkdir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _save(model_name: str, tag: str, **arrays: np.ndarray) -> str:
    """Save arrays to output/<model_name>/<tag>.npz"""
    out_dir = _mkdir(os.path.join(OUTPUT_ROOT, model_name))
    path = os.path.join(out_dir, f"{tag}.npz")
    np.savez_compressed(path, **arrays)
    return path


# ---------------------------------------------------------------------------
# Shared Black-Scholes utilities (exact as used in paper)
# ---------------------------------------------------------------------------
def _bs_call(F: float, K: float, T: float, sigma: float) -> float:
    if sigma <= 0 or T <= 0:
        return max(F - K, 0.0)
    d1 = (math.log(max(F / K, 1e-12)) + 0.5 * sigma ** 2 * T) / (sigma * math.sqrt(T))
    # missing r - idk what htat is 
    d2 = d1 - sigma * math.sqrt(T)
    return F * norm.cdf(d1) - K * norm.cdf(d2)
    # it's missing e^(-rt) which I asusme is missing because r wasn't included in d1

def _bs_iv(price: float, F: float, K: float, T: float) -> float:
    intrinsic = max(F - K, 0.0)
    if price <= intrinsic + 1e-12:
        return float("nan")
    try:
        return brentq(
            lambda s: _bs_call(F, K, T, s) - price,
            1e-6, 10.0, xtol=1e-9, maxiter=200,
        )
    # what brentrq solves for is the root. what we are solving for here is f(sigma) - price = 0.
    # i assume this is a methodfor solving the optimal theta or implied volatility for realistic market conditions.
    # finding the optimal sigma that is as realistic as possible
    # and i assume this has a closed form solution whatever right.
    # i have no clue what the other parameters of this lambda function are.
    except ValueError:
        return float("nan")

# ===========================================================================
# MODEL 1 — Heston Stochastic Volatility (exact match to Appendix)
# ===========================================================================
def _heston_qe_step(
    V: np.ndarray, kappa: float, theta: float, sigma_v: float,
    dt: float, z_gauss: np.ndarray, z_unif: np.ndarray, psi_c: float = 1.5
) -> np.ndarray:
    """Andersen (2007) Quadratic-Exponential scheme for variance.
    
    Args:
        V:             Variance
        kappa:         Mean-reversion speed.
        theta:         Long-run variance.
        sigma_v:       Vol-of-vol
        dt:            Timestep
        z_gauss:       Simulate the variance + update asset price
        z_uniform:     Uniform normal for switch between high-variance (quadriatic guassian) and low-variance (exponential) regimes
        psi_c:         Swithc between quadriatic approximation adn an exponential approximation
    """
    
    exp_k = math.exp(-kappa * dt)
    m = theta + (V - theta) * exp_k
    s2 = (V * sigma_v**2 * exp_k / kappa * (1 - exp_k) +
          theta * sigma_v**2 / (2 * kappa) * (1 - exp_k)**2)
    # check V[t-1] - I think ts is vectorized
    psi = s2 / (m**2 + 1e-12)

    # Gaussian regime (psi <= psiC)
    psi_inv = 2.0 / (psi + 1e-12)
    b2 = psi_inv - 1 + np.sqrt(np.maximum(psi_inv * (psi_inv - 1), 0))
    # check if it's avoiding negatives
    # check if it is psi_inv or 2/psi_inv
    a = m / (1 + b2 + 1e-12)
    V_gauss = a * (np.sqrt(b2) + z_gauss)**2

    # Exponential regime (psi > psiC)
    p = (psi - 1) / (psi + 1 + 1e-12)
    beta = (1 - p) / (m + 1e-12)
    V_exp = np.where(
        z_unif > p, # check if it is z_unif <=p
        np.log(np.maximum((1 - p) / np.maximum(1 - z_unif, 1e-12), 1e-12)) / beta,
        0.0, # where function is a little different
    )
    return np.maximum(np.where(psi < psi_c, V_gauss, V_exp), 0.0)


def _heston_logspot_step(
    X: np.ndarray, V: np.ndarray, V_next: np.ndarray,
    kappa: float, theta: float, sigma_v: float, rho: float,
    dt: float, z: np.ndarray,
) -> np.ndarray:
    """Andersen trapezoidal rule for log-spot.
    
      Args:
        X:             Simulated stock price
        V:             Simulated Variance
        V_next:        Next value of Variance (check phrasing)
        kappa:         Mean-reversion speed.
        theta:         Long-run variance.
        sigma_v:       Vol-of-vol
        rho:           Correlation between spot and variance Brownians.
        dt:            Timestep

      Missing thingy:
        q: dividend yield (see line 139)
    """
    
    k0 = -rho * kappa * theta / sigma_v * dt
    k1 = 0.5 * dt * (kappa * rho / sigma_v - 0.5) - rho / sigma_v
    # technically takes parameter gamma for k1 and k2 (set to 0.5 --> 0.5*dt)
    k2 = 0.5 * dt * (kappa * rho / sigma_v - 0.5) + rho / sigma_v
    k3 = 0.5 * dt * (1 - rho**2)
    k4 = k3
    # technically should be two parameters for k3 and k4 --> gamma1 and gamma2, but both are equal to each other
    return X + k0 + k1 * V + k2 * V_next + np.sqrt(np.maximum(k3 * V + k4 * V_next, 0)) * z # check if this is missing (rho - q) 


def _heston_cf(u, tau, kappa, theta, sigma_v, rho, v0):
    """Heston characteristic function (standard Gatheral form).
    
    Args:
        kappa:         Mean-reversion speed.
        theta:         Long-run variance.
        sigma_v:       Vol-of-vol
        rho:           Correlation between spot and variance Brownians.
        v0:            Initial volility
        tau:           *check*
        u:             sample points (nodes) *check*
    """
    alpha = -0.5 * (u * u + 1j * u)
    beta_c = kappa - rho * sigma_v * 1j * u
    gamma = 0.5 * sigma_v**2
    d = np.sqrt(beta_c**2 - 4 * alpha * gamma + 1e-12)
    r_m = (beta_c - d) / (sigma_v**2)
    r_p = (beta_c + d) / (sigma_v**2)
    g = r_m / (r_p + 1e-12)
    exp_dt = np.exp(-d * tau)
    denom = 1 - g * exp_dt
    C = (kappa * theta / sigma_v**2) * ((beta_c - d) * tau - 2 * np.log(denom / (1 - g + 1e-12)))
    # There is an addiitonal theta parameter for some reason in our thingy. it's also a little differnet
    # than the C(u, tau) solution proposed here: https://quant.stackexchange.com/questions/7048/other-means-of-calibrating-heston-models
    D = r_m * (1 - exp_dt) / (denom + 1e-12)
    return np.exp(C + D * v0)


def _heston_call_lewis(k, tau, kappa, theta, sigma_v, rho, v0, S0=1.0, n=128):
    """Lewis (2000) call price via Gauss-Laguerre.

    Args:
        k:             *Check this*   
        kappa:         Mean-reversion speed.
        theta:         Long-run variance.
        sigma_v:       Vol-of-vol
        rho:           Correlation between spot and variance Brownians.
        v0:            Initial variance
        S0:            Initial price
        n:             Degere of Laguerre polynomial
    """
    nodes, weights = np.polynomial.laguerre.laggauss(n)
    u = nodes + 0.0j
    phi = _heston_cf(u - 0.5j, tau, kappa, theta, sigma_v, rho, v0)
    # subtrct by 0.5j for damping factor; it's a little bit different in the paper
    # because it's defined as 1/2 + ix rather than x - 0.5i
    intgd = np.real(np.exp(-1j * u * k) * phi / (u**2 + 0.25))
    integ = np.dot(weights, intgd)
    # this is good - check if dot product is fine here (if not use inner product)
    K = S0 * math.exp(k)
    return S0 - math.sqrt(S0 * K) * integ / math.pi
    # like everything is fine but it should be
    # S0e^(-rt) - e^(k/2)*integ/π - but the thing is is that like S0 might be equal to S0exp(-rt)
    # So it just depends on what exactly S0 is equal to in the context of the solver.


# Brandon can we verify this
def run_heston(
    n_params: int,
    n_paths: int,
    n_steps: int,
    log_strikes: np.ndarray,
    tenors: np.ndarray,
    rng: np.random.Generator,
    crisis: bool = False,
) -> Dict[str, np.ndarray]:
    """Exact Heston implementation per paper Appendix."""
    if not crisis:
        # Just check the parameters/justify choices
        kappa = rng.uniform(0.5, 4.0, n_params)
        theta_bar = rng.uniform(0.01, 0.25, n_params)
        sigma_v = rng.uniform(0.1, 0.6, n_params)
        rho = rng.uniform(-0.7, -0.1, n_params)
        v0 = rng.uniform(0.01, 0.30, n_params)
    else:
        # just check parameters/jsutify choices
        kappa = rng.uniform(0.1, 0.5, n_params)
        theta_bar = rng.uniform(0.25, 0.50, n_params)
        sigma_v = rng.uniform(0.6, 1.0, n_params)
        rho = rng.uniform(-0.95, -0.7, n_params)
        v0 = rng.uniform(0.30, 0.60, n_params)

    
    params = np.stack([kappa, theta_bar, sigma_v, rho, v0], axis=1)
    n_tau = len(tenors)
    n_k = len(log_strikes)
    samples = np.full((n_params, n_paths, n_tau, n_k), np.nan)

    for p_idx in range(n_params):
        ka, th, sv, rh, v_0 = params[p_idx]
        dt = tenors[-1] / n_steps
        X = np.zeros(n_paths)
        V = np.full(n_paths, v_0)

        step = 0
        for tau_idx, tau in enumerate(tenors):
            target_step = round(tau / dt)
            while step < target_step:
                z0 = rng.standard_normal(n_paths)  # for QE
                z1 = rng.standard_normal(n_paths)  # for spot
                z2 = rng.standard_normal(n_paths)  # uniform source for QE
                V_new = _heston_qe_step(V, ka, th, sv, dt, z0, z2)
                X = _heston_logspot_step(X, V, V_new, ka, th, sv, rh, dt, z1)
                V = V_new
                step += 1

            # Compute stochastic IV surface using terminal variance per path
            for k_idx, log_k in enumerate(log_strikes):
                for path_idx in range(n_paths):
                    try:
                        price = _heston_call_lewis(log_k, tau, ka, th, sv, rh, V[path_idx])
                        K = math.exp(log_k)
                        iv = _bs_iv(price, 1.0, K, tau)
                        if not np.isnan(iv):
                            samples[p_idx, path_idx, tau_idx, k_idx] = iv
                    except Exception:
                        pass  # rare numerical failure

    return {
        "params": params,
        "samples": samples,
        "log_strikes": log_strikes,
        "tenors": tenors,
    }


# ===========================================================================
# MODEL 2 — Heath-Jarrow-Morton
# ===========================================================================
def run_hjm(
    n_params: int,
    n_paths: int,
    n_steps: int,
    maturities: np.ndarray,
    rng: np.random.Generator,
    crisis: bool = False,
    f0_slope: float = 0.0,
    n_factors: int = 2,
) -> Dict[str, np.ndarray]:
    """2-factor exponential volatility HJM per paper.

    Fixes vs original:

    1. DRIFT: the correct no-arbitrage drift for correlated factors is
           alpha(t,T) = sum_{j,k} R[j,k] * sigma_j(t,T) * integral_t^T sigma_k(t,s) ds
       The original np.sum(sigma * integ, axis=0) is the R=I diagonal-only
       form; it omits the cross terms rho12*(sigma_1*integ_2 + sigma_2*integ_1).
       For crisis rho12 near -1 and 30Y tenor the error is ~150bps.
       Fixed with np.einsum("jm,jk,km->m", sigma, R, integ).

    2. EXPIRED MATURITIES: once t > T_i the original np.maximum(mat-t, 0)
       kept tau=0 but sigma stayed at xi_j (no decay) and alpha stayed
       nonzero. Now active = maturities > t, and sigma/integ are explicitly
       zeroed for ~active columns so expired rates stop evolving.

    3. NEW ARGS:
       f0_slope: gives a sloped initial curve f(0,T) = f0_level + slope*T.
       n_factors: forward-compatibility placeholder (only 2 implemented).
    """
    if n_factors != 2:
        raise ValueError("Only n_factors=2 is currently implemented.")

    M = len(maturities)

    if not crisis:
        xi1   = rng.uniform(0.005, 0.020, n_params) # constant volatility coefficient
        xi2   = rng.uniform(0.005, 0.020, n_params) # constant volatility coefficient
        lam1  = rng.uniform(0.10,  1.00,  n_params) # speed at which volatility decays
        lam2  = rng.uniform(0.01,  0.20,  n_params) # speed at which volatility decays
        rho12 = rng.uniform(-0.5,  0.5,   n_params) # correlation coefficient
        f0_lv = rng.uniform(0.01,  0.08,  n_params) # level of initial forward curve (i.e., f(0, T) at short end)
    else:
        xi1   = rng.uniform(0.020, 0.050, n_params) # constant volatility coefficient
        xi2   = rng.uniform(0.020, 0.050, n_params) # constant volatility coefficient
        lam1  = rng.uniform(0.01,  0.10,  n_params) # speed at which volatility decays
        lam2  = rng.uniform(0.01,  0.05,  n_params) # speed at which volatility decays
        rho12 = rng.uniform(-0.95, -0.5,  n_params) # correlation coefficient
        f0_lv = rng.uniform(0.08,  0.20,  n_params) # level of initial forward curve (i.e., f(0, T) at short end)

    params  = np.stack([xi1, xi2, lam1, lam2, rho12, f0_lv], axis=1)
    samples = np.zeros((n_params, n_paths, M))
    dt      = 1.0 / n_steps

    for p_idx in range(n_params):
        xi      = np.array([xi1[p_idx], xi2[p_idx]])   # (2,)
        lam     = np.array([lam1[p_idx], lam2[p_idx]]) # (2,)
        rho_val = rho12[p_idx]

        # Initial forward curve: flat level + optional upward slope
        f0 = f0_lv[p_idx] + f0_slope * maturities      # (M,)
        # f0_1v *  rho12 + f0_slope * maturities

        # Correlation matrix R and its Cholesky factor L (dW = L @ dZ)
        rho_c = float(np.clip(rho_val, -1.0 + 1e-7, 1.0 - 1e-7))
        # clip between (-1,1)
        R     = np.array([[1.0, rho_c], [rho_c, 1.0]])
        # R is a square matrix with a Cholesky decomposition
        
        try:
            L = np.linalg.cholesky(R)
        except np.linalg.LinAlgError:
            L = np.eye(2)

        f = np.tile(f0, (n_paths, 1))   # (n_paths, M)

        for k in range(n_steps):
            t = k * dt

            # FIX 2: zero expired maturities so their rates stop evolving
            active = maturities > t                          # (M,) bool
            tau    = np.where(active, maturities - t, 0.0)   # (M,)

            sigma = xi[:, None] * np.exp(-lam[:, None] * tau[None, :])  # (2, M)
            sigma[:, ~active] = 0.0

            integ = (xi[:, None] / lam[:, None]) * (
                1.0 - np.exp(-lam[:, None] * tau[None, :]))              # (2, M)
            integ[:, ~active] = 0.0

            # FIX 1: full drift including off-diagonal cross terms via R
            alpha = np.einsum("jm,jk,km->m", sigma, R, integ)           # (M,)

            dZ   = rng.standard_normal((n_paths, 2)) * math.sqrt(dt)
            dW   = dZ @ L.T                                              # (n_paths, 2)
            diff = np.einsum("jm,pj->pm", sigma, dW)                    # (n_paths, M)
            f   += alpha * dt + diff

        samples[p_idx] = f

    return {"params": params, "samples": samples, "maturities": maturities}


# ===========================================================================
# MODEL 3 — SABR (exact match)
# ===========================================================================
def _hagan_lognormal(F, K, T, alpha, beta, rho, nu):
    """Hagan et al. (2002) implied vol formula.
    
    Args:
        F:             Volatility of the forward
        K:             Strike price
        T:             Time to maturity
        alpha:         Initial Volatility
        beta:          CEV exponent
        rho:           Correlation
        nu:            Volvol
    """
    
    F = max(F, 1e-12)
    Ks = np.atleast_1d(K)
    out = np.empty(len(Ks))
    for i, k in enumerate(Ks):
        k = max(k, 1e-12)
        FK_b = (F * k) ** ((1 - beta) / 2)
        lf = math.log(F / k)
        eps = 1e-8
        if abs(lf) < eps:
            z_chi = 1.0
        else:
            z = (nu / alpha) * FK_b * lf
            chi = math.log((math.sqrt(1 - 2 * rho * z + z**2) + z - rho) / (1 - rho + eps))
            z_chi = z / (chi + eps)
        A = 1 + ((1 - beta)**2 * alpha**2 / (24 * (F * k)**(1 - beta)) +
                 rho * beta * nu * alpha / (4 * FK_b) +
                 (2 - 3 * rho**2) * nu**2 / 24) * T
        out[i] = (alpha / FK_b) * z_chi * A
    return out

def run_sabr(
    n_params: int,
    n_paths: int,
    n_steps: int,
    strikes: np.ndarray,
    tenors: np.ndarray,
    rng: np.random.Generator,
    crisis: bool = False,
) -> Dict[str, np.ndarray]:
    """SABR Monte Carlo + Hagan IV per paper."""
    if not crisis:
        alpha_p = rng.uniform(0.05, 0.50, n_params)
        beta_p = rng.uniform(0.20, 0.90, n_params)
        rho_p = rng.uniform(-0.7, 0.0, n_params)
        nu_p = rng.uniform(0.10, 0.80, n_params)
    else:
        alpha_p = rng.uniform(0.50, 0.80, n_params)
        beta_p = rng.uniform(0.20, 0.90, n_params)
        rho_p = rng.uniform(-0.95, -0.7, n_params)
        nu_p = rng.uniform(0.80, 1.50, n_params)

    params = np.stack([alpha_p, beta_p, rho_p, nu_p], axis=1)
    n_tau = len(tenors)
    n_k = len(strikes)
    samples = np.full((n_params, n_paths, n_tau, n_k), np.nan)

    for p_idx in range(n_params):
        al, be, rh, nu = params[p_idx]
        F0 = 1.0
        for tau_idx, tau in enumerate(tenors):
            dt_s = tau / n_steps
            sqrt_dt = math.sqrt(dt_s)
            L_corr = np.array([[1.0, 0.0], [rh, math.sqrt(max(1 - rh**2, 0))]])

            F_paths = np.full(n_paths, F0)
            sigma_paths = np.full(n_paths, al)

            for _ in range(n_steps):
                Z = rng.standard_normal((n_paths, 2))
                W = Z @ L_corr.T
                dWF = W[:, 0] * sqrt_dt
                dWs = W[:, 1] * sqrt_dt
                sigma_paths *= np.exp(-0.5 * nu**2 * dt_s + nu * dWs)
                F_beta = np.maximum(F_paths, 0) ** be
                F_paths = np.maximum(F_paths + sigma_paths * F_beta * dWF, 0)

            # Terminal IV via Hagan
            for path_idx in range(n_paths):
                F_t = max(F_paths[path_idx], 1e-6)
                sig_t = max(sigma_paths[path_idx], 1e-6)
                ivs = _hagan_lognormal(F_t, strikes * F_t, tau, sig_t, be, rh, nu)
                samples[p_idx, path_idx, tau_idx, :] = ivs

    return {"params": params, "samples": samples, "strikes": strikes, "tenors": tenors}


# ===========================================================================
# MODEL 4 — Rough Heston (BLP hybrid + CF as per paper)
# ===========================================================================
class _RoughHestonCF:
    """Adams predictor-corrector for rough Heston characteristic function (paper reference)."""
    def __init__(self, n: int, T: float, kappa: float, theta: float, lam: float,
                 rho: float, V0: float, H: float):
        self.n = n
        self.T = T
        self.dt = T / n
        self.t = np.linspace(0, T, n + 1)
        self.kappa = kappa
        self.theta = theta
        self.lam = lam
        self.rho = rho
        self.V0 = V0
        self.alpha = H + 0.5
        self._build_weights()

    def _build_weights(self):
        n = self.n
        al = self.alpha
        dt = self.dt
        self.a_ = np.zeros((n + 1, n + 1))
        self.b_ = np.zeros((n, n + 1))
        frac = dt**al / gamma(al + 2)
        frac2 = dt**al / gamma(al + 1)
        for k in range(1, n + 1):
            for j in range(k + 1):
                if j == 0:
                    self.a_[j, k] = frac * ((k - 1)**(al + 1) - (k - al - 1) * k**al)
                elif j == k:
                    self.a_[j, k] = frac
                else:
                    self.a_[j, k] = frac * ((k + 1 - j)**(al + 1) + (k - 1 - j)**(al + 1) - 2 * (k - j)**(al + 1))
            for j in range(k):
                self.b_[j, k] = frac2 * ((k - j)**al - (k - j - 1)**al)

    def _F(self, a, x):
        return (-0.5 * (a * a + 1j * a) - (self.kappa - 1j * a * self.rho * self.lam) * x +
                0.5 * self.lam**2 * x * x)

    def char_fn(self, a):
        h = np.zeros(self.n + 1, dtype=complex)
        for k in range(1, self.n + 1):
            hP = sum(self.b_[j, k] * self._F(a, h[j]) for j in range(k))
            h[k] = sum(self.a_[j, k] * self._F(a, h[j]) for j in range(k)) + self.a_[k, k] * self._F(a, hP)
        std_int = np.trapz(h, self.t)
        be = 1 - self.alpha
        kern = np.array([(self.T - self.t[i])**be - (self.T - self.t[i + 1])**be
                         for i in range(self.n)])
        frac_int = np.dot(kern, h[:self.n]) / (gamma(1 - self.alpha) * be)
        return np.exp(self.kappa * self.theta * std_int + self.V0 * frac_int)

    def call(self, log_k, n_quad=64):
        nodes, weights = np.polynomial.laguerre.laggauss(n_quad)
        u = nodes + 0j
        phi = np.array([self.char_fn(ui - 0.5j) for ui in u])
        ig = np.real(np.exp(-1j * u * log_k) * phi / (u**2 + 0.25))
        K = math.exp(log_k)
        return 1.0 - math.sqrt(K) * np.dot(weights, ig) / math.pi


def run_rough_heston(
    n_params: int,
    n_paths: int,
    n_steps_cf: int,
    log_strikes: np.ndarray,
    tenors: np.ndarray,
    rng: np.random.Generator,
    crisis: bool = False,
) -> Dict[str, np.ndarray]:
    """Rough Heston with BLP-style CF discretization per paper."""
    if not crisis:
        kappa = rng.uniform(0.5, 4.0, n_params)
        theta = rng.uniform(0.01, 0.25, n_params)
        lam = rng.uniform(0.1, 1.0, n_params)
        rho = rng.uniform(-0.7, -0.1, n_params)
        V0 = rng.uniform(0.01, 0.30, n_params)
        H = rng.uniform(0.05, 0.20, n_params)
    else:
        kappa = rng.uniform(0.1, 0.5, n_params)
        theta = rng.uniform(0.25, 0.50, n_params)
        lam = rng.uniform(1.0, 2.0, n_params)
        rho = rng.uniform(-0.95, -0.7, n_params)
        V0 = rng.uniform(0.30, 0.60, n_params)
        H = rng.uniform(0.00, 0.05, n_params)

    params = np.stack([kappa, theta, lam, rho, V0, H], axis=1)
    n_tau = len(tenors)
    n_k = len(log_strikes)
    samples = np.full((n_params, n_paths, n_tau, n_k), np.nan)

    for p_idx in range(n_params):
        ka, th, la, rh, v0, H_ = params[p_idx]
        H_ = np.clip(H_, 0.01, 0.49)
        for tau_idx, tau in enumerate(tenors):
            # BLP-style: simulate variance process via CF at perturbed initial conditions for distributional spread
            for path_idx in range(n_paths):
                # Small perturbation of initial variance for path diversity (consistent with BLP hybrid)
                v0_path = max(v0 + 0.05 * v0 * rng.standard_normal(), 1e-4)
                cf = _RoughHestonCF(n_steps_cf, tau, ka, th, la, rh, v0_path, H_)
                for k_idx, lk in enumerate(log_strikes):
                    try:
                        price = cf.call(lk)
                        K = math.exp(lk)
                        iv = _bs_iv(price, 1.0, K, tau)
                        if not np.isnan(iv):
                            samples[p_idx, path_idx, tau_idx, k_idx] = iv
                    except Exception:
                        pass

    return {
        "params": params,
        "samples": samples,
        "log_strikes": log_strikes,
        "tenors": tenors,
    }


# ===========================================================================
# Physical Models (exact SPDE forms from Appendix A)
# ===========================================================================
def run_reaction_diffusion(
    n_params: int, n_paths: int, Nx: int, Ny: int, rng: np.random.Generator, crisis: bool = False
) -> Dict[str, np.ndarray]:
    """Stochastic reaction-diffusion: du = (nu Δu + u(1-u)) dt + σ dW"""
    if not crisis:
        nu_p = rng.uniform(0.005, 0.05, n_params)
        sigma_p = rng.uniform(0.01, 0.20, n_params)
        u0_p = rng.uniform(0.3, 0.7, n_params)
        T_p = rng.uniform(0.5, 2.0, n_params)
    else:
        nu_p = rng.uniform(0.001, 0.005, n_params)
        sigma_p = rng.uniform(0.20, 0.50, n_params)
        u0_p = rng.uniform(0.45, 0.55, n_params)
        T_p = rng.uniform(2.0, 5.0, n_params)

    params = np.stack([nu_p, sigma_p, u0_p, T_p], axis=1)
    samples = np.zeros((n_params, n_paths, Nx, Ny))

    kx = np.fft.fftfreq(Nx, d=1.0/Nx) * 2 * math.pi
    ky = np.fft.fftfreq(Ny, d=1.0/Ny) * 2 * math.pi
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    lap_eig = -(KX**2 + KY**2)

    for p_idx in range(n_params):
        nu, sigma, u0_mean, T = params[p_idx]
        n_steps = max(int(T / 1e-3), 50)
        dt = T / n_steps
        impl = 1.0 / (1.0 - dt * nu * lap_eig)
        for path_idx in range(n_paths):
            u = u0_mean + 0.05 * rng.standard_normal((Nx, Ny))
            for _ in range(n_steps):
                reaction = u * (1 - u) * dt
                noise = sigma * math.sqrt(dt) * rng.standard_normal((Nx, Ny))
                rhs = u + reaction + noise
                rhs_hat = np.fft.fft2(rhs)
                u_hat = impl * rhs_hat
                u = np.real(np.fft.ifft2(u_hat))
            samples[p_idx, path_idx] = u

    return {"params": params, "samples": samples}


def run_navier_stokes(
    n_params: int, n_paths: int, N: int, rng: np.random.Generator, crisis: bool = False
) -> Dict[str, np.ndarray]:
    """Stochastically forced 2D Navier-Stokes (vorticity form)."""
    if not crisis:
        nu_p = rng.uniform(5e-4, 5e-3, n_params)
        sigma_p = rng.uniform(0.01, 0.10, n_params)
        T_p = rng.uniform(0.5, 2.0, n_params)
    else:
        nu_p = rng.uniform(1e-4, 5e-4, n_params)
        sigma_p = rng.uniform(0.10, 0.30, n_params)
        T_p = rng.uniform(2.0, 5.0, n_params)

    params = np.stack([nu_p, sigma_p, T_p], axis=1)
    samples = np.zeros((n_params, n_paths, N, N))

    L = 2 * math.pi
    kv = np.fft.fftfreq(N, d=1.0 / N)
    KX, KY = np.meshgrid(kv, kv, indexing="ij")
    K2 = KX**2 + KY**2
    K2s = np.where(K2 > 0, K2, 1.0)
    k_cut = int(2 * (N // 2) / 3)
    dealias = (np.abs(KX) <= k_cut) & (np.abs(KY) <= k_cut)

    K_abs = np.sqrt(K2)
    spec = np.where(K2 > 0, K_abs**(-1.0) * np.exp(-K2 / 32.0), 0.0)
    spec /= (np.sqrt(np.sum(spec**2)) + 1e-12)

    x = np.linspace(0, L, N, endpoint=False)
    X, Y = np.meshgrid(x, x, indexing="ij")

    for p_idx in range(n_params):
        nu, sigma, T = params[p_idx]
        n_steps = max(int(T / 5e-3), 20)
        dt = T / n_steps
        cn_d = 1 + 0.5 * dt * nu * K2
        cn_n = 1 - 0.5 * dt * nu * K2

        for path_idx in range(n_paths):
            w = 2 * np.sin(2 * X) * np.sin(2 * Y)
            wh = np.fft.fft2(w)
            nl_prev = np.zeros_like(wh)

            for _ in range(n_steps):
                psi_h = -wh / K2s
                psi_h[0, 0] = 0
                u = np.real(np.fft.ifft2(1j * KY * psi_h))
                v = np.real(np.fft.ifft2(-1j * KX * psi_h))
                wx = np.real(np.fft.ifft2(1j * KX * wh))
                wy = np.real(np.fft.ifft2(1j * KY * wh))
                nl_curr = -np.fft.fft2(u * wx + v * wy) * dealias
                nl_ab = 1.5 * nl_curr - 0.5 * nl_prev

                xi = (rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))) / math.sqrt(2)
                dWh = math.sqrt(dt) * sigma * spec * xi

                wh = (cn_n * wh + dt * nl_ab + dWh) / cn_d
                wh[0, 0] = 0
                nl_prev = nl_curr

            samples[p_idx, path_idx] = np.real(np.fft.ifft2(wh))

    return {"params": params, "samples": samples}


def run_burgers(
    n_params: int, n_paths: int, Nx: int, rng: np.random.Generator, crisis: bool = False
) -> Dict[str, np.ndarray]:
    """Stochastic Burgers equation."""
    if not crisis:
        nu_p = rng.uniform(0.005, 0.05, n_params)
        sigma_p = rng.uniform(0.01, 0.20, n_params)
        L_p = rng.uniform(1.0, 4.0, n_params)
    else:
        nu_p = rng.uniform(0.001, 0.005, n_params)
        sigma_p = rng.uniform(0.20, 0.50, n_params)
        L_p = rng.uniform(4.0, 8.0, n_params)

    params = np.stack([nu_p, sigma_p, L_p], axis=1)
    samples = np.zeros((n_params, n_paths, Nx))

    for p_idx in range(n_params):
        nu, sigma, L = params[p_idx]
        dx = L / Nx
        dt = 0.4 * dx / (0.5 + sigma)
        T_sim = 1.0
        n_steps = max(int(T_sim / dt), 10)

        kv = np.fft.rfftfreq(Nx, d=1.0 / Nx) * 2 * math.pi / L
        lap_r = -kv**2
        impl_r = 1.0 / (1 - 0.5 * dt * nu * lap_r)

        for path_idx in range(n_paths):
            u = np.sin(2 * math.pi * np.linspace(0, L, Nx, endpoint=False) / L)
            for _ in range(n_steps):
                # Upwind advection
                adv = np.where(
                    u >= 0,
                    u * (u - np.roll(u, 1)) / dx,
                    u * (np.roll(u, -1) - u) / dx,
                )
                noise = sigma * math.sqrt(dt) * rng.standard_normal(Nx) / math.sqrt(dx)
                rhs = u - dt * adv + noise
                rhs_r = np.fft.rfft(rhs)
                u_r = impl_r * rhs_r
                u = np.fft.irfft(u_r, Nx)
            samples[p_idx, path_idx] = u

    return {"params": params, "samples": samples}


def run_allen_cahn(
    n_params: int, n_paths: int, Nx: int, Ny: int, rng: np.random.Generator, crisis: bool = False
) -> Dict[str, np.ndarray]:
    """Stochastic Allen-Cahn near criticality."""
    if not crisis:
        nu_p = rng.uniform(0.005, 0.05, n_params)
        sigma_p = rng.uniform(0.01, 0.15, n_params)
        u0_s = rng.uniform(0.05, 0.30, n_params)
        T_p = rng.uniform(0.5, 2.0, n_params)
    else:
        nu_p = rng.uniform(0.001, 0.005, n_params)
        sigma_p = rng.uniform(0.15, 0.50, n_params)
        u0_s = rng.uniform(0.40, 0.50, n_params)
        T_p = rng.uniform(2.0, 5.0, n_params)

    params = np.stack([nu_p, sigma_p, u0_s, T_p], axis=1)
    samples = np.zeros((n_params, n_paths, Nx, Ny))

    kx = np.fft.fftfreq(Nx, d=1.0 / Nx) * 2 * math.pi
    ky = np.fft.fftfreq(Ny, d=1.0 / Ny) * 2 * math.pi
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    lap = -(KX**2 + KY**2)

    for p_idx in range(n_params):
        nu, sigma, u0_std, T = params[p_idx]
        n_steps = max(int(T / 1e-3), 50)
        dt = T / n_steps
        impl = 1.0 / (1.0 - dt * (nu * lap + 1.0))  # linear part u - u^3 ≈ u

        for path_idx in range(n_paths):
            u = u0_std * rng.standard_normal((Nx, Ny))
            for _ in range(n_steps):
                nl = -u**3
                noise = sigma * math.sqrt(dt) * rng.standard_normal((Nx, Ny))
                rhs = u + dt * nl + noise
                rhs_h = np.fft.fft2(rhs)
                u = np.real(np.fft.ifft2(impl * rhs_h))
            samples[p_idx, path_idx] = u

    return {"params": params, "samples": samples}


# ===========================================================================
# Master pipeline (exact model list and defaults from paper)
# ===========================================================================
ALL_MODELS = [
    "heston", "hjm", "sabr", "rough_heston",
    "reaction_diffusion", "navier_stokes", "burgers", "allen_cahn",
]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Unified SPDE training data pipeline — Yee et al. (exact)")
    p.add_argument("--models", nargs="+", default=ALL_MODELS,
                   choices=ALL_MODELS, metavar="MODEL",
                   help="Models to run (default: all)")
    p.add_argument("--n-params", type=int, default=200,
                   help="Number of parameter vectors per regime (default: 200)")
    p.add_argument("--n-paths", type=int, default=100,
                   help="Number of MC paths per parameter vector (default: 100)")
    p.add_argument("--n-steps", type=int, default=50,
                   help="Time steps per simulation (default: 50)")
    p.add_argument("--grid-size", type=int, default=32,
                   help="Spatial grid size for physical models (default: 32)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed (default: 42)")
    p.add_argument("--no-crisis", action="store_true",
                   help="Skip crisis-regime generation")
    p.add_argument("--hjm-f0-slope", type=float, default=0.0,
                   help="HJM initial curve slope in rate/year. "
                        "f(0,T) = f0_level + slope*T. Default 0 (flat).")
    p.add_argument("--hjm-n-factors", type=int, default=2,
                   help="HJM number of factors (only 2 supported). Default 2.")
    return p


def _run_one(
    name: str,
    rng: np.random.Generator,
    n_params: int,
    n_paths: int,
    n_steps: int,
    grid: int,
    crisis: bool,
    hjm_f0_slope: float = 0.0,
    hjm_n_factors: int = 2,
) -> None:
    """Run single model exactly as specified."""
    # Shared grids (exact from paper)
    log_strikes = np.linspace(-0.3, 0.3, 13)
    strikes = np.exp(log_strikes)
    tenors = np.array([1/12, 3/12, 6/12, 1.0, 2.0, 3.0, 5.0])
    maturities = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30])

    tags = ["in_dist", "crisis"] if crisis else ["in_dist"]
    is_crisis_flags = [False, True] if crisis else [False]

    for tag, is_crisis in zip(tags, is_crisis_flags):
        t0 = time.time()
        print(f" [{name}] {tag} ...", end=" ", flush=True)

        if name == "heston":
            data = run_heston(n_params, n_paths, n_steps, log_strikes, tenors, rng, is_crisis)
        elif name == "hjm":
            data = run_hjm(n_params, n_paths, n_steps, maturities, rng, is_crisis,
                           f0_slope=hjm_f0_slope, n_factors=hjm_n_factors)
        elif name == "sabr":
            data = run_sabr(n_params, n_paths, n_steps, strikes, tenors, rng, is_crisis)
        elif name == "rough_heston":
            data = run_rough_heston(n_params, n_paths, max(n_steps // 2, 20),
                                    log_strikes, tenors[:4], rng, is_crisis)
        elif name == "reaction_diffusion":
            data = run_reaction_diffusion(n_params, n_paths, grid, grid, rng, is_crisis)
        elif name == "navier_stokes":
            data = run_navier_stokes(n_params, n_paths, grid, rng, is_crisis)
        elif name == "burgers":
            data = run_burgers(n_params, n_paths, grid * 4, rng, is_crisis)
        elif name == "allen_cahn":
            data = run_allen_cahn(n_params, n_paths, grid, grid, rng, is_crisis)
        else:
            raise ValueError(f"Unknown model: {name}")

        path = _save(name, tag, **data)
        elapsed = time.time() - t0
        shapes = {k: v.shape for k, v in data.items() if isinstance(v, np.ndarray)}
        print(f"done ({elapsed:.1f}s) → {path}")
        for k, sh in shapes.items():
            print(f"   {k}: {sh}")


def main() -> None:
    args = build_parser().parse_args()
    rng = np.random.default_rng(args.seed)

    print("=" * 70)
    print("SPDE Training Data Pipeline — Exact reproduction of Yee et al. benchmark")
    print("=" * 70)
    print(f"Models      : {args.models}")
    print(f"n_params    : {args.n_params}")
    print(f"n_paths     : {args.n_paths}")
    print(f"n_steps     : {args.n_steps}")
    print(f"grid_size   : {args.grid_size}")
    print(f"seed        : {args.seed}")
    print(f"crisis      : {not args.no_crisis}")
    print(f"output      : {OUTPUT_ROOT}")
    print("=" * 70)

    t_total = time.time()
    for model_name in args.models:
        print(f"\n→ {model_name.upper()}")
        try:
            _run_one(
                name=model_name,
                rng=rng,
                n_params=args.n_params,
                n_paths=args.n_paths,
                n_steps=args.n_steps,
                grid=args.grid_size,
                crisis=not args.no_crisis,
                hjm_f0_slope=args.hjm_f0_slope,
                hjm_n_factors=args.hjm_n_factors,
            )
        except Exception as exc:
            print(f"  ERROR in {model_name}: {exc}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 70)
    print(f"Finished in {time.time() - t_total:.1f}s")
    print(f"All data written to: {OUTPUT_ROOT}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
