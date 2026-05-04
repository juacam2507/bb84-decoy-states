import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from dataHandler import DataHandler
import os



class Plotter:
    def __init__(self) -> None:

        self.fig_dir = "figures"
        os.makedirs(self.fig_dir, exist_ok=True)

    def R_vs_D_plot(self, filename):

        data_handler = DataHandler("key_rate_vs_distance")

        filepath = os.path.join(data_handler.dirpath, filename)

        df = data_handler.read_data(filepath=filepath)

        distances = df.iloc[:, 0].to_numpy(dtype=float)
        theoretical = df.iloc[:, -1].to_numpy(dtype=float)
        experimental = df.iloc[:, 1:-1].to_numpy(dtype=float)

        n_exp = len(experimental[0])

        box_data = [experimental[i, :] for i in range(len(distances))]

        display_floor = 1e-10
        plot_data = [np.where(row == 0, display_floor, row) for row in box_data]

        fig, ax = plt.subplots(figsize=(10, 6))

        bp = ax.boxplot(
            plot_data,
            positions=distances,
            widths=np.diff(np.append(distances, distances[-1] + 5)).min() * 0.4,
            patch_artist=True,
            manage_ticks=False,
            boxprops=dict(facecolor="steelblue", alpha=0.55),
            medianprops=dict(color="navy", linewidth=2),
            whiskerprops=dict(color="steelblue", linewidth=1.5),
            flierprops=dict(
                marker="o",
                markerfacecolor="steelblue",
                markeredgecolor="steelblue",
                markersize=4,
                alpha=0.6,
            ),
        )

        ax.plot(
            distances,
            theoretical,
            color="crimson",
            linewidth=2,
            marker="o",
            markersize=5,
            label="Theoretical Key rate",
        )

        ax.set_yscale("log")
        ax.set_xlabel("Distance (Km)", fontsize=12)
        ax.set_ylabel("Key Rate (bits/pulse)", fontsize=12)
        ax.set_title(
            f"Experimental key rates ({n_exp} runs) vs Theoretical Key Rate",
            fontsize=13,
        )
        ax.set_xticks(distances)
        ax.set_xticklabels([f"{d:.1f}" for d in distances], rotation=30, ha="center")
        ax.grid(True, which="both", linestyle="--", alpha=0.4)

        legend_elements = [
            Patch(
                facecolor="steelblue",
                alpha=0.55,
                label=f"Experimental key rates ({n_exp} runs/distance)",
            ),
            Line2D(
                [0],
                [0],
                color="crimson",
                linewidth=2,
                marker="o",
                markersize=5,
                label="Theoretical key rate",
            ),
        ]

        ax.legend(handles=legend_elements, fontsize=10)

        plt.tight_layout()
        plt.show()
