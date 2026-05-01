import numpy as np
from tqdm import tqdm
from datetime import datetime
from bb88_simulator import Simulator
from securityAnalysis import SecurityAnalysis
from quantumChannel import QuantumChannel
from classicalChannel import ClassicalChannel
import json
import os


class DistanceSweep:
    def __init__(
        self,
        simulation_parameters: dict,
        distance_sweep_params: dict,
        rng: np.random.Generator,
    ):
        self.rng = rng
        self.simulation_parameters = simulation_parameters
        self.debug = simulation_parameters["debug"]

        self.n_sample = distance_sweep_params["n_sample"]

        self.d_min = distance_sweep_params["distance_control"]["d_min"]
        self.d_max = distance_sweep_params["distance_control"]["d_max"]
        self.alpha_dist = distance_sweep_params["distance_control"]["alpha_dist"]
        self.iter_min = distance_sweep_params["iteration_control"]["iter_min"]
        self.iter_max = distance_sweep_params["iteration_control"]["iter_max"]
        self.alpha_iter = distance_sweep_params["iteration_control"]["alpha_iter"]

        self.iterations = self.generate_array(
            min=self.iter_min, max=self.iter_max, alpha=self.alpha_iter, type=int
        )
        self.distances = self.generate_array(
            min=self.d_min, max=self.d_max, alpha=self.alpha_dist, type=float
        )
        
        if self.debug:
            print(f"[DEBUG] Distances: {self.distances}")
            print(f"[DEBUG] Iterations: {self.iterations}")

    def generate_array(
        self, min: float, max: float, alpha: float, type: type[int] | type[float]
    ) -> np.ndarray:

        t = np.linspace(0.0, 1.0, self.n_sample)

        array = (min + (max - min) * (t**alpha)).astype(type)

        return array

    def run_experimental(self):

        key_rates = np.array([], dtype=float)

        i = 0

        for d in tqdm(self.distances, desc="Distances"):
            quantum_channel = QuantumChannel(self.simulation_parameters, self.rng, d)
            classical_channel = ClassicalChannel(self.simulation_parameters)
            simulator = Simulator(
                quantum_channel=quantum_channel, classical_channel=classical_channel
            )
            security_analysis = SecurityAnalysis(quantum_channel=quantum_channel)

            Q_exp, E_exp = simulator.run(iterations=self.iterations[i])
            R_exp = security_analysis.compute_key_rate(gains=Q_exp, qbers=E_exp)

            key_rates = np.append(key_rates, R_exp)
        
        if self.debug:    
            print(f"[DEBUG] Experimental Key rates: {key_rates}")

        return key_rates

    def run_theoretical(self):
        
        key_rates = np.array([], dtype=float)

        i = 0

        for d in tqdm(self.distances, desc="Distances"):
            quantum_channel = QuantumChannel(self.simulation_parameters, self.rng, d)
            security_analysis = SecurityAnalysis(quantum_channel=quantum_channel)

            Q_teo = security_analysis.compute_theoretical_gains()
            E_teo = security_analysis.compute_theoretical_qbers(Q_teo=Q_teo)
            R_teo = security_analysis.compute_key_rate(gains=Q_teo, qbers=E_teo)

            key_rates = np.append(key_rates, R_teo)
            
        if self.debug:    
            print(f"[DEBUG] Theoretical Key rates: {key_rates}")

        return key_rates
