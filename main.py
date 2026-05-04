import numpy as np
from distanceSweep import DistanceSweep
from dataHandler import DataHandler

simulation_parameters = {
    "Iterations": 50,
    "N": 50_000_000,  # Number of generated pulses
    "mu": 0.55,  # Signal intensity
    "decoy_intensities": [0.10, 0.0],  # Decoy intensities
    "state_probs": [0.90, 0.08, 0.02],  # State probability
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
    "attack_properties": {
        "execute_attack": False,  # Bool value to determine if the attack is performed
        "attack_type": "BS",  # Type of attack. "PNS" or "BS".
        "efficiency_loss": 0.9,  # Efficiency loss for the BS attack
    },
    "error_correction_efficiency": 1.2,
    "debug": False,
}

distance_sweep_params = {
    "n_sample": 60,
    "distance_control": {
        "d_min": 10,
        "d_max": 140,
        "alpha_dist": 0.4,  # Controls the concentration of distances sampled
    },
    "iteration_control": {
        "iter_min": simulation_parameters["Iterations"],
        "iter_max": simulation_parameters["Iterations"],
        "alpha_iter": 0.4,  # Controls the concentration of iterations sampled
    },
}

iter = simulation_parameters["Iterations"]
rng = np.random.default_rng()

distance_sweep = DistanceSweep(
    simulation_parameters=simulation_parameters,
    distance_sweep_params=distance_sweep_params,
    rng=rng,
)

R_exp = distance_sweep.run_experimental()
R_teo = distance_sweep.run_theoretical()

data_handler = DataHandler(dir="key_rate_vs_distance")
first_column = "Distance (Km)"
last_column = "Theoretical Key rates (bits/pulse)"

middle_columns = []
for i in range(iter):
    middle_columns.append(f"Key rate {i+1} (bits/pulse)")

R_data_header = [first_column] + middle_columns + [last_column]

data_handler.write_data(
    distance_sweep.distances,
    R_exp,
    R_teo,
    header=R_data_header,
    filename="key_rate",
    simulation_parameters=simulation_parameters,
)
