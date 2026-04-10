import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from bb88_simulator import Simulator
from datetime import datetime
import json
import os

simulation_parameters = {
    "Iterations": 1,
    "N": 1_000_000,  # Number of generated pulses
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
d_max = 140
d_sample = 60
alpha = 0.4  # Controls the concentration of points

iter_max = 120
iter_min = 20
gamma = 0.4

t = np.linspace(0.0, 1.0, d_sample)

iterations = (iter_min + (iter_max - iter_min) * (t**gamma)).astype(int)
distances = d_min + (d_max - d_min) * (t**alpha)
key_rates = np.array([], dtype=float)
print(f"Iterations: {iterations}")
print(f"Distances: {distances}" )

print(f"Distance (Km), Key rate (bit/s)")
i = 0
for d in tqdm(distances, desc="Distances"):

    R = simulator.run(l=d, iter=iterations[i])
    key_rates = np.append(key_rates, R)
    print(f"{d}, {R}")
    i+=1

data = np.column_stack((distances, key_rates))
data_dir = "data"
os.makedirs(data_dir, exist_ok=True)

timestamp = datetime.now()

meta = simulation_parameters.copy()
meta["time"] = timestamp.strftime("%Y/%m/%d - %H:%M:%S")

filename = f"data_{timestamp.strftime('%Y%m%d_%H%M%S')}_{iter}_{simulation_parameters['N']}.csv"
filepath = os.path.join(data_dir, filename)


with open(filepath, "w", encoding="utf-8") as f:
    meta_json = json.dumps(meta, indent=2, ensure_ascii=False)
    for line in meta_json.splitlines():
        f.write(f"#{line}\n")
    f.write("#---\n")

    header = "Distance (Km), Key rate (1/s)\n"
    f.write(header)

    np.savetxt(
        f,
        data,
        delimiter=",",
        fmt="%.8f",
    )
