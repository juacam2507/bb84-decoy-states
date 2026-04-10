import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from bb88_simulator import Simulator
from datetime import datetime

simulation_parameters = {
    "Iterations": 75,
    "N": 30_000_000,  # Number of generated pulses
    "mu": 0.5,  # Signal intensity
    "decoy_intensities": [0.1, 0.0],  # Decoy intensities
    "decoy_rate": 0.3,  # Decoy probability
    "channel_properties": {
        "beta": 0.2,  # Loss coefficient (dB/Km)
    },
    "detector_properties": {
        "receiver_transmit": 0.013,  # Receiver transmittance
        "detector_efficiency": 0.2,  # Detector Efficiency
        "detector_error": 0.03,  # Probability of a pulse measured in the correct basis to trigger the wrong detector
        "dark_count_rate": 4e-9,  # Probability of dark counts
        "dark_count_error": 0.5,  # Probability of dark counts triggering the wrong detector
    },
    "error_correction_efficiency": 1.0,
    "debug": False,
}
rng = np.random.default_rng()
iter = simulation_parameters["Iterations"]

simulator = Simulator(simulation_parameters=simulation_parameters, rng=rng)

d_min = 10.0
d_max = 120
sample = 60
alpha = 0.5  # Controls the concentration of points

t = np.linspace(0.0, 1.0, sample)

distances = d_min + (d_max - d_min) * (t**alpha)
key_rates = np.array([], dtype=float)

print(f"Distance (Km), Key rate (bit/s)")
for d in tqdm(distances, desc="Distances"):

    R = simulator.run(l=d)
    key_rates = np.append(key_rates, R)
    print(f"{d}, {R}")


data = np.column_stack((distances, key_rates))
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

filename = f"data_{timestamp}_{iter}_{simulation_parameters["N"]}.csv"

np.savetxt(
    "data.csv",
    data,
    delimiter=",",
    header="Distance (Km), Key rate (bit/s)",
    comments="",
    fmt="%.8f",
)
