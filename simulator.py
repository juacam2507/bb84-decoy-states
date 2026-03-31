from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, Dict
from tqdm import tqdm
from scipy.special import factorial

# =========================
# Parámetros
# =========================

@dataclass
class QKDParams:
    alpha: float
    eta_det: float
    bob_loss: float
    e_d: float
    Y0: float
    e0: float


# =========================
# Canal
# =========================

def channel_eta(params: QKDParams, distance: float) -> float:
    return 10 ** (-params.alpha * distance / 10) * params.eta_det * params.bob_loss


# =========================
# Tabla de verdad (modelo base)
# =========================

def detection_probs(n: np.ndarray, eta: float, e_d: float, Y0: float, e0: float):
    p_s = 1 - (1 - eta) ** n

    P0 = (1 - e_d) * p_s + (1 - e0) * Y0
    P1 = e_d * p_s + e0 * Y0


    qx = (1 - P0) * (1 - P1)
    qy = P0 * (1 - P1)
    qz = (1 - P0) * P1
    qw = P0 * P1

    return qx, qy, qz, qw


# =========================
# MONTE CARLO
# =========================

def simulate_bb84(N, mu, nu, params: QKDParams, distance, rng):
    eta = channel_eta(params, distance)

    alice_bits = rng.integers(0, 2, N)
    alice_basis = rng.integers(0, 2, N)

    states = rng.choice([2,1,0], size=N, p=[0.75,0.125,0.125])

    pulses = np.zeros(N, dtype=int)
    pulses[states==2] = rng.poisson(mu, np.sum(states==2))
    pulses[states==1] = rng.poisson(nu, np.sum(states==1))

    bob_basis = rng.integers(0,2,N)
    sift = alice_basis == bob_basis

    qx,qy,qz,qw = detection_probs(pulses, eta, params.e_d, params.Y0, params.e0)

    probs = np.vstack([qx,qy,qz,qw]).T

    events = np.array([0,1,2,3])
    sampled = np.array([rng.choice(events, p=p/np.sum(p)) for p in probs])

    bob_bits = -np.ones(N)

    correct = sampled==1
    error = sampled==2
    double = sampled==3

    bob_bits[correct] = alice_bits[correct]
    bob_bits[error] = 1 - alice_bits[error]
    bob_bits[double] = rng.integers(0,2,np.sum(double))

    valid = sift & (bob_bits!=-1)

    Q = np.sum(valid)/N
    QBER = np.mean(bob_bits[valid] != alice_bits[valid]) if np.sum(valid)>0 else 0

    return Q, QBER


# =========================
# ANALÍTICO
# =========================

def analytic_gain_error(mu: float, eta: float, params: QKDParams, max_n=50):
    """
    Calcula Q_mu y E_mu analíticamente
    """
    probs = [np.exp(-mu) * mu**n / factorial(n) for n in range(max_n)]
    probs = np.array(probs)

    n_vals = np.arange(max_n)

    qx,qy,qz,qw = detection_probs(n_vals, eta, params.e_d, params.Y0, params.e0)

    Y_n = 1 - qx
    e_n = (qz + 0.5*qw) / Y_n
    e_n[Y_n == 0] = 0

    Q = np.sum(probs * Y_n)
    E = np.sum(probs * Y_n * e_n) / Q if Q>0 else 0

    return Q, E


# =========================
# DECOY STATES
# =========================

def decoy_estimation(mu, nu, Q_mu, Q_nu, Q_0, E_mu, E_nu):
    Y0 = Q_0

    Y1 = (mu/(mu*nu - nu**2)) * (
        Q_nu*np.exp(nu) - (nu**2/mu**2)*Q_mu*np.exp(mu) - (1-nu**2/mu**2)*Y0
    )

    e1 = (E_nu*Q_nu*np.exp(nu) - 0.5*Y0) / (nu*Y1)

    return max(Y1,0), max(e1,0)


# =========================
# KEY RATE
# =========================

def H2(x):
    if x<=0 or x>=1: return 0
    return -x*np.log2(x)-(1-x)*np.log2(1-x)


def key_rate(mu, Q_mu, E_mu, Y1, e1):
    return Q_mu*(-H2(E_mu)) + mu*np.exp(-mu)*Y1*(1-H2(e1))


# =========================
# SIMULACIÓN COMPLETA
# =========================

def run_simulation():
    params = QKDParams(
        alpha=0.35,
        eta_det=0.5,
        bob_loss=0.5,
        e_d=0.02,
        Y0=1e-6,
        e0=0.7
    )

    distances = np.linspace(0,150,100)

    mu, nu = 1.0, 0.5
    N = 10000
    iterations = 5

    rng = np.random.default_rng()

    rates_mc = []
    rates_an = []

    for d in tqdm(distances, desc="Distance sweep"):

        # Monte Carlo promedio
        R_accum = 0
        for _ in range(iterations):
            Q_mu, E_mu = simulate_bb84(N, mu, nu, params, d, rng)
            Q_nu, E_nu = simulate_bb84(N, nu, nu, params, d, rng)
            Q_0, _ = simulate_bb84(N, 0.0, nu, params, d, rng)

            Y1, e1 = decoy_estimation(mu, nu, Q_mu, Q_nu, Q_0, E_mu, E_nu)
            R = key_rate(mu, Q_mu, E_mu, Y1, e1)

            R_accum += R

        rates_mc.append(max(R_accum/iterations,1e-15))

        # Analítico
        eta = channel_eta(params, d)

        Q_mu_a, E_mu_a = analytic_gain_error(mu, eta, params)
        Q_nu_a, E_nu_a = analytic_gain_error(nu, eta, params)
        Q_0_a, _ = analytic_gain_error(0.0, eta, params)

        Y1_a, e1_a = decoy_estimation(mu, nu, Q_mu_a, Q_nu_a, Q_0_a, E_mu_a, E_nu_a)
        R_a = key_rate(mu, Q_mu_a, E_mu_a, Y1_a, e1_a)

        rates_an.append(max(R_a,1e-15))

    # Plot
    plt.figure()
    plt.semilogy(distances, rates_mc, 'o-', label="Monte Carlo")
    plt.semilogy(distances, rates_an, '-', label="Analítico")
    plt.xlabel("Distance (km)")
    plt.ylabel("Key Rate")
    plt.legend()
    plt.grid(True, which="both")
    plt.title("QKD: Monte Carlo vs Analítico")
    plt.savefig("key_rate_vs_distance.png")
    plt.show()


if __name__ == "__main__":
    run_simulation()