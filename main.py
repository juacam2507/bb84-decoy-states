import numpy as np
from distanceSweep import DistanceSweep
from yieldAnalysis import YieldAnalysis
from dataHandler import DataHandler

simulation_parameters = {
    "iterations": 50,
    "N": 100_000_000,  # Number of generated pulses
    "mu": 0.48,  # Signal intensity
    "decoy_intensities": [0.05, 0.0],  # Decoy intensities
    "state_probs": [0.75, 0.15, 0.10],  # State probabilities
    "channel_properties": {
        "beta": 0.21,  # Loss coefficient (dB/Km)
    },
    "detector_properties": {
        "receiver_transmit": 0.225,  # Receiver transmittance
        "detector_efficiency": 0.20,  # Detector Efficiency
        "detector_error": 0.033,  # Probability of a pulse measured in the correct basis to trigger the wrong detector
        "dark_count_rate": 1.7e-6,  # Probability of dark counts
        "dark_count_error": 0.5,  # Probability of dark counts triggering the wrong detector
    },
    "attack_properties": {
        "execute_attack": False,  # Bool value to determine if the attack is performed
        "attack_type": "BS",  # Type of attack. "PNS" or "BS".
        "efficiency_loss": 0.9,  # Efficiency loss for the BS attack
    },
    "error_correction_efficiency": 1.22,
    "debug": True,
}

distance_sweep_params = {
    "run_sweep": False,
    "n_sample": 5,
    "distance_control": {
        "d_min": 10,
        "d_max": 50,
        "alpha_dist": 1,  # Controls the concentration of distances sampled
    },
    "iteration_control": {
        "iter_min": simulation_parameters["iterations"],
        "iter_max": simulation_parameters["iterations"],
        "alpha_iter": 1.0,  # Controls the concentration of iterations sampled
    },
}

yield_sweep_params = {
    "run_yields": True,
    "photon_nums": [1, 2, 3],
    "distance": 20,
}

distance_sweep_switch = distance_sweep_params["run_sweep"]
yield_sweep_switch = yield_sweep_params["run_yields"]

iter = simulation_parameters["iterations"]
rng = np.random.default_rng()

if distance_sweep_switch:
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

if yield_sweep_switch:

    yield_analysis = YieldAnalysis(
        simulation_parameters=simulation_parameters,
        yield_sweep_params=yield_sweep_params,
        rng=rng,
    )

    signal_data, decoy_data, theoretical_yield = yield_analysis.run_channel()

    yield_data_handler = DataHandler(dir="yield_assessment")

    first_column = "n"
    second_column = "Theoretical yield"
    signal_columns = []
    decoy_columns = []

    for i in range(iter):
        signal_columns.append(f"Signal Yield {i + 1}")
        decoy_columns.append(f"Decoy Yield {i + 1}")

    yield_data_header = [first_column] + [second_column] + signal_columns + decoy_columns

    yield_data_handler.write_data(
        yield_analysis.photon_nums,
        theoretical_yield,
        signal_data,
        decoy_data,
        header=yield_data_header,
        filename="yields",
        simulation_parameters=simulation_parameters,
    )
