import numpy as np
from tqdm import tqdm
from bb88_simulator import Simulator
from channelAnalysis import ChannelAnalysis
from datetime import datetime
import json
import os

simulation_parameters = {
    "Iterations": 10,
    "N": 100_000_000,  # Number of generated pulses
    "mu": 0.55,  # Signal intensity
    "decoy_intensities": [0.10, 0.0],  # Decoy intensities
    "state_probs": [0.80, 0.16 ,0.04],  # State probability
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
    "error_correction_efficiency": 1.2,
    "debug": True,
}
rng = np.random.default_rng()

simulator = Simulator(simulation_parameters=simulation_parameters, rng=rng)
analysis = ChannelAnalysis(simulation_parameters=simulation_parameters)

d_min = 50
d_max = 50
d_sample = 1
alpha = 0.4  # Controls the concentration of distances sampled

iter_max = 1
iter_min = 1
gamma = 0.4 # Controls the concentration of iterations sampled

t = np.linspace(0.0, 1.0, d_sample)

iterations = (iter_min + (iter_max - iter_min) * (t**gamma)).astype(int)
distances = d_min + (d_max - d_min) * (t**alpha)
key_rates = np.array([], dtype=float)
key_rates_teo = np.array([], dtype=float)
print(iterations)
i = 0
for d in tqdm(distances, desc="Distances"):
    
    Q_exp, E_exp, eta = simulator.run(l=d, iterations=iterations[i])
    R_exp = analysis.compute_key_rate(gains=Q_exp, qbers=E_exp)
    
    Q_teo = analysis.compute_theoretical_gains(eta=eta)
    E_teo = analysis.compute_theoretical_qbers(eta=eta, Q_teo=Q_teo)
    R_teo = analysis.compute_key_rate(gains=Q_teo, qbers=E_exp)
    
    key_rates = np.append(key_rates, R_exp)
    key_rates_teo = np.append(key_rates_teo, R_teo)
    print(f"{d}, {R_exp}, {R_teo}")
    i += 1

data = np.column_stack([distances, key_rates, key_rates_teo])
data_dir = "data"
os.makedirs(data_dir, exist_ok=True)

timestamp = datetime.now()

meta = simulation_parameters.copy()
meta["time"] = timestamp.strftime("%Y/%m/%d - %H:%M:%S")

filename = f"data_{timestamp.strftime('%Y%m%d_%H%M%S')}_{simulation_parameters['N']}.csv"
filepath = os.path.join(data_dir, filename)


with open(filepath, "w", encoding="utf-8") as f:
    meta_json = json.dumps(meta, indent=2, ensure_ascii=False)
    for line in meta_json.splitlines():
        f.write(f"#{line}\n")
    f.write("#---\n")

    header = "Distance (Km), Key rate (1/pulse), Theoretical key rate(1/pulse)\n"

    np.savetxt(
        f,
        data,
        delimiter=",",
        fmt="%.8f",
    )
