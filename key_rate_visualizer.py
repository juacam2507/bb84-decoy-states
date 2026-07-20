import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

os.makedirs("visualization", exist_ok=True)


plt.rcParams.update(
    {
        "figure.dpi": 180,
        "savefig.dpi": 150,
        "font.size": 26,  # was 21
        "axes.labelsize": 27,  # was 22
        "axes.titlesize": 28,  # was 23
        "axes.titleweight": "bold",
        "xtick.labelsize": 25,  # was 20
        "ytick.labelsize": 25,  # was 20
        "legend.fontsize": 25,  # was 20
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": False,
        "grid.alpha": 0.18,
        "grid.linestyle": "--",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "mathtext.fontset": "stix",
        "font.family": "STIXGeneral",
    }
)


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
    q_1 = np.where(np.isfinite(q_1), q_1, 0.0)
    return np.maximum(q_1, 0.0)


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
    e_1 = np.where(np.isfinite(e_1), e_1, 0.0)
    return np.clip(e_1, 0.0, 0.5)


def key_rate(mu, nu, eta, e_d, y_0, e_0, q, f):
    q_1 = Q_1_L(mu, nu, eta, y_0)
    e_1 = e_1_u(mu, nu, eta, e_d, y_0, e_0)
    Q_mu = Q(eta, y_0, mu)
    E_mu = E(eta, e_d, y_0, e_0, mu)
    i_ab = q_1 * (1 - shannon_entropy(e_1))
    i_ae = Q_mu * f * shannon_entropy(E_mu)
    R = np.where(i_ab > i_ae, q * (i_ab - i_ae), 0.0)
    return np.maximum(R, 0.0)


# =========================
# Parameters
# =========================


beta = 0.21
l = 20
t_bob = 0.225
eta_detector = 0.200
e_d = 0.033
y_0 = 1.7e-6
e_0 = 0.5
f = 1.22
q = 0.5


t_ab = transmittance(l, beta)
eta = eta_channel(t_ab, t_bob, eta_detector)


# =========================
# Parameter grid  (nu < mu enforced)
# =========================


mu_vals = np.linspace(0.05, 1.20, 320)
nu_vals = np.linspace(0.005, 0.80, 280)
MU, NU = np.meshgrid(mu_vals, nu_vals)
mask = NU < MU


signal_qber = np.where(mask, E(eta, e_d, y_0, e_0, MU), np.nan)
single_photon_gain = np.where(mask, Q_1_L(MU, NU, eta, y_0), np.nan)
single_photon_error = np.where(mask, e_1_u(MU, NU, eta, e_d, y_0, e_0), np.nan)
signal_gain = np.where(mask, Q(eta, y_0, MU), np.nan)
secret_key_rate = np.where(mask, key_rate(MU, NU, eta, e_d, y_0, e_0, q, f), np.nan)


plots = [
    ("signal_qber", signal_qber, "Signal QBER", r"$E_{\mu}$", "viridis"),
    (
        "single_photon_gain",
        single_photon_gain,
        "Single-Photon Gain",
        r"$Q_1^L$",
        "magma",
    ),
    (
        "single_photon_error",
        single_photon_error,
        "Single-Photon Error",
        r"$e_1^U$",
        "cividis",
    ),
    ("signal_gain", signal_gain, "Signal Gain", r"$Q_{\mu}$", "plasma"),
    ("key_rate", secret_key_rate, "Secret Key Rate", r"$R$", "inferno"),
]


outdir = "visualization"


def add_parameter_box(fig):
    txt = (
        r"$\beta=0.21\,\mathrm{dB/km},\; l=20\,\mathrm{km},"
        r"\; t_{\mathrm{Bob}}=0.225,\; \eta_d=0.200$" + "\n"
        r"$e_d=0.033,\; Y_0=1.7\times10^{-6},\; e_0=0.5,\; f=1.22,\; q=0.5$"
    )
    fig.text(
        0.5,
        0.03,
        txt,
        ha="center",
        va="bottom",
        fontsize=25,  # was 20
        bbox=dict(boxstyle="round,pad=0.35", fc="#f6f6f6", ec="#cccccc", lw=0.8),
    )


for fname, Z, title, zlabel, cmap in plots:
    valid = Z[np.isfinite(Z)]
    zmin, zmax = float(np.nanmin(valid)), float(np.nanmax(valid))

    fig = plt.figure(figsize=(15.5, 10.5), constrained_layout=False)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.18, 1.0], wspace=0.18)

    # ---- 3D surface ----
    # ax3d = fig.add_subplot(gs[0, 0], projection="3d")
    # ax3d.plot_surface(
    #     MU,
    #     NU,
    #     Z,
    #     cmap=cmap,
    #     linewidth=0,
    #     antialiased=True,
    #     rcount=180,
    #     ccount=180,
    #     shade=True,
    # )
    # ax3d.set_xlabel(r"Signal intensity $\mu$", labelpad=10)
    # ax3d.set_ylabel(r"Decoy intensity $\nu$", labelpad=10)
    # ax3d.set_zlabel(zlabel, labelpad=8)
    # ax3d.view_init(elev=28, azim=-128)
    # ax3d.xaxis.pane.set_alpha(0.0)
    # ax3d.yaxis.pane.set_alpha(0.0)
    # ax3d.zaxis.pane.set_alpha(0.0)
    # ax3d.grid(True, alpha=0.15)
    # ax3d.set_xlim(mu_vals.min(), mu_vals.max())
    # ax3d.set_ylim(nu_vals.min(), nu_vals.max())
    # ax3d.set_zlim(zmin, zmax)

    # ---- 2D heatmap ----
    ax2d = fig.add_subplot(gs[:, :])
    pcm = ax2d.pcolormesh(MU, NU, Z, shading="auto", cmap=cmap)
    cs = ax2d.contour(MU, NU, Z, levels=8, colors="white", linewidths=0.5, alpha=0.75)
    ax2d.clabel(
        cs, inline=True, fmt="%.2e" if zmax < 1e-2 else "%.3f", fontsize=23
    )  # was 18

    ax2d.plot(
        mu_vals,
        mu_vals,
        ls="--",
        lw=1.2,
        color="black",
        alpha=0.85,
        label=r"$\nu = \mu$ (boundary)",
    )
    ax2d.fill_between(mu_vals, mu_vals, nu_vals.max(), color="white", alpha=0.92)

    ax2d.set_xlim(mu_vals.min(), mu_vals.max())
    ax2d.set_ylim(nu_vals.min(), nu_vals.max())
    ax2d.set_xlabel(r"Signal intensity $\mu$", labelpad=14)
    ax2d.set_ylabel(r"Decoy intensity $\nu$", labelpad=14)
    ax2d.tick_params(axis="both", pad=7, labelsize=25)
    ax2d.legend(loc="upper right", frameon=True, prop={"size": 25})

    cbar = fig.colorbar(pcm, ax=ax2d, fraction=0.065, pad=0.035)
    cbar.set_label(zlabel, fontsize=27, labelpad=14)  # was 22
    cbar.ax.tick_params(labelsize=25)

    if zmax < 1e-2:
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((-2, 2))
        cbar.formatter = formatter
        cbar.update_ticks()
        cbar.ax.tick_params(labelsize=25)

    fig.suptitle(
        f"{title} vs signal and decoy intensities",
        fontsize=31,  # was 26
        fontweight="bold",
        y=0.965,
    )

    # add_parameter_box(fig)

    fig.subplots_adjust(left=0.12, right=0.85, bottom=0.15, top=0.87)

    fig.savefig(f"{outdir}/{fname}.png", bbox_inches=None)
    plt.close(fig)
    print(f"Saved {outdir}/{fname}.png")
