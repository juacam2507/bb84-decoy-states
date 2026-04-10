import numpy as np
import matplotlib.pyplot as plt

# Leer archivo CSV como array 2D
data = np.genfromtxt('data/data_20260410_124020_100_1000000.csv', delimiter=',', skip_header=1)

distances = data[:, 0]        # primera columna
key_rates = data[:, 1]   # segunda columna

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 18,
    "axes.labelsize": 20,
    "axes.titlesize": 22,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
    "axes.linewidth": 1.5,
    "xtick.major.width": 1.5,
    "ytick.major.width": 1.5,
    "lines.linewidth": 2.5,
    "lines.markersize": 8,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

fig, ax = plt.subplots(figsize=(8, 6))

color_mc = "#b11932"  
ax.semilogy(
    distances, key_rates,
    marker='o',
    linestyle='-',     
    color=color_mc,
    linewidth=1.5,
    markersize=5,
    label="MonteCarlo simulation",
    markeredgecolor="white",
    markeredgewidth=1.0,
    
)

ax.set_xlabel("Distance (km)", fontsize=20, labelpad=10)
ax.set_ylabel("Secret key rate (R)", fontsize=20, labelpad=10)
ax.set_title("Secret key rate vs. distance", fontsize=22, pad=20)


#ax.grid(True, which="both", axis="y", alpha=0.5, color="gray", linewidth=1.0)

ax.legend(
    loc="upper right",
    frameon=True,
    fancybox=False,
    edgecolor="0.8",
    facecolor="white",
    framealpha=1.0,
)

# Opcional: ajustar márgenes para que no quede tan apretado
fig.tight_layout()

# Guardar para papel o presentación (sin pérdida de calidad)
fig.savefig("secure_key_rate_vs_distance.png", dpi=300, bbox_inches="tight")
fig.savefig("secure_key_rate_vs_distance.pdf", bbox_inches="tight")  # útil para artículos

plt.show()