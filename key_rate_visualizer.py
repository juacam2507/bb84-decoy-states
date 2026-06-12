import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.optimize import minimize

# =========================
# Physical/model functions
# =========================


def transmittance(l, beta):
    return 10 ** (-beta * l / 10)


def eta_channel(t_ab, t_bob, eta_detector):
    return t_ab * t_bob * eta_detector


def Q(eta, y_0, mu):
    return y_0 + 1 - np.exp(-eta * mu)


def E(eta, e_d, y_0, e_0, mu):
    return (e_0 * y_0 + e_d * (1 - np.exp(-eta * mu))) / (y_0 + 1 - np.exp(-eta * mu))


def shannon_entropy(x):
    x = np.asarray(x)
    with np.errstate(divide="ignore", invalid="ignore"):
        H = np.where((x > 0) & (x < 1), -x * np.log2(x) - (1 - x) * np.log2(1 - x), 0.0)
    return H


def Q_1_L(mu, nu, eta, y_0):
    denom = nu * mu - nu**2

    Q_mu = Q(eta, y_0, mu)
    Q_nu = Q(eta, y_0, nu)

    numer = (
        mu**2
        * np.exp(-mu)
        * (
            Q_nu * np.exp(nu)
            - ((mu**2 - nu**2) / mu**2) * y_0
            - (nu**2 / mu**2) * Q_mu * np.exp(mu)
        )
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        q_1 = np.where(denom > 0, numer / denom, 0.0)

    return q_1


def e_1_u(mu, nu, eta, e_d, y_0, e_0):
    q1 = Q_1_L(mu, nu, eta, y_0)

    with np.errstate(divide="ignore", invalid="ignore"):
        y_1_L = np.where(mu > 0, q1 / (mu * np.exp(-nu)), 0.0)

    Q_nu = Q(eta, y_0, nu)
    E_nu = E(eta, e_d, y_0, e_0, nu)

    numer = E_nu * Q_nu * np.exp(nu) - e_0 * y_0
    denom = nu * y_1_L

    with np.errstate(divide="ignore", invalid="ignore"):
        e_1 = np.where(denom > 0, numer / denom, 0.0)

    return e_1


def key_rate(mu, nu, eta, e_d, y_0, e_0, q, f):
    q_1 = Q_1_L(mu, nu, eta, y_0)
    e_1 = e_1_u(mu, nu, eta, e_d, y_0, e_0)

    Q_mu = Q(eta, y_0, mu)
    E_mu = E(eta, e_d, y_0, e_0, mu)

    i_ab = q_1 * (1 - shannon_entropy(e_1))
    i_ae = Q_mu * f * shannon_entropy(E_mu)

    R = np.where(i_ab > i_ae, q * (i_ab - i_ae), 0.0)
    return R


# =========================
# Parameters
# =========================

beta = 0.21
l = 140

t_bob = 0.225
eta_detector = 0.200
e_d = 0.033
y_0 = 1.7e-6
e_0 = 0.5

f = 1.22
q = 0.5

t_ab = transmittance(l, beta)
eta = eta_channel(t_ab, t_bob, eta_detector)




