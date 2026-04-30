import numpy as np
from tqdm import tqdm
from datetime import datetime
from bb88_simulator import Simulator
from channelAnalysis import ChannelAnalysis
from quantumChannel import QuantumChannel
import json
import os


class DistanceAnalysis:
    def __init__(self, simulation_parameters : dict, distance_analysis_params: dict, ):
        self.sim_params = simulation_parameters

        self.n_sample = distance_analysis_params["n_sample"]

        self.d_min = distance_analysis_params["distance_control"]["d_min"]
        self.d_max = distance_analysis_params["distance_control"]["d_max"]
        self.alpha_dist = distance_analysis_params["distance_control"]["alpha_dist"]

        self.iter_min = distance_analysis_params["iteration_control"]["iter_min"]
        self.iter_max = distance_analysis_params["iteration_control"]["iter_max"]
        self.alpha_iter = distance_analysis_params["iteration_control"]["alpha_iter"]

        self.iterations = self.generate_array(
            min=self.iter_min, max=self.iter_max, alpha=self.alpha_iter, type=int
        )
        self.distances = self.generate_array(
            min=self.d_min, max=self.d_max, alpha=self.alpha_dist, type=float
        )

    def generate_array(
        self, min: float, max: float, alpha: float, type: type[int] | type[float]
    ) -> np.ndarray:

        t = np.linspace(0.0, 1.0, self.n_sample)

        array = (min + (max - min) * (t**alpha)).astype(type)

        return array

    def run_distance_sweep(
        self, simulator: Simulator, channel_analysis: ChannelAnalysis, analytical: bool
    ):
        key_rates = np.array([], dtype=float)

        i = 0

        for d in tqdm(self.distances, desc="Distances"):
            quantum_channel = QuantumChannel(source=alice, detector= bob, postProcess=post_process,l=d)
            eta = quantum_channel.eta

            Q_exp, E_exp = simulator.run(iterations=self.iterations[i])
            R_exp = channel_analysis.compute_key_rate(gains=Q_exp, qbers=E_exp)
            i += 1

            key_rates = np.append(key_rates, R_exp)
        return key_rates
