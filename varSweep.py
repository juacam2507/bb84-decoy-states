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
        sweep_params: dict,
        rng: np.random.Generator,
    ):
        self.rng = rng
        self.simulation_parameters = simulation_parameters
        self.debug = simulation_parameters["debug"]

        self.var_name = sweep_params["var_name"]
        self.n_sample = sweep_params["n_sample"]

        self.var_min = sweep_params["var_control"]["var_min"]
        self.var_max = sweep_params["var_control"]["var_max"]
        self.alpha_var = sweep_params["var_control"]["alpha_var"]
        self.iter_min = sweep_params["iteration_control"]["iter_min"]
        self.iter_max = sweep_params["iteration_control"]["iter_max"]
        self.alpha_iter = sweep_params["iteration_control"]["alpha_iter"]

        self.iterations = self.generate_array(
            min=self.iter_min, max=self.iter_max, alpha=self.alpha_iter, type=int
        )
        self.values = self.generate_array(
            min=self.var_min, max=self.var_max, alpha=self.alpha_var, type=float
        )

        if self.debug:
            print(f"[DEBUG] Distances: {self.values}")
            print(f"[DEBUG] Iterations: {self.iterations}")

    def generate_array(
        self, min: float, max: float, alpha: float, type: type[int] | type[float]
    ) -> np.ndarray:

        t = np.linspace(0.0, 1.0, self.n_sample)

        array = (min + (max - min) * (t**alpha)).astype(type)

        return array

    def run_experimental(self):

        key_rates = []

        i = 0

        for val in tqdm(self.values, desc=self.var_name):
            self.simulation_parameters[self.var_name] = val
            quantum_channel = QuantumChannel(self.simulation_parameters, self.rng)
            classical_channel = ClassicalChannel(self.simulation_parameters)
            simulator = Simulator(
                quantum_channel=quantum_channel, classical_channel=classical_channel
            )
            security_analysis = SecurityAnalysis(quantum_channel=quantum_channel)

            state_gains, state_errors = simulator.run(iterations=self.iterations[i])
            R_exp = security_analysis.compute_key_rate(
                state_gains=state_gains, state_errors=state_errors
            )

            key_rates.append([R_exp])

        key_rates = np.vstack(key_rates)

        if self.debug:
            print(f"[DEBUG] Experimental Key rates: {key_rates}")
            print("----------------------------------------------------------------")

        return key_rates

    def run_theoretical(self):

        key_rates = []

        i = 0

        for val in tqdm(self.values, desc=self.var_name):
            quantum_channel = QuantumChannel(self.simulation_parameters, self.rng)
            security_analysis = SecurityAnalysis(quantum_channel=quantum_channel)

            Q_teo = security_analysis.compute_theoretical_gains()
            E_teo = security_analysis.compute_theoretical_qbers(Q_teo=Q_teo)
            R_teo = security_analysis.compute_key_rate(
                state_gains=Q_teo, state_errors=E_teo
            )

            key_rates = np.append(key_rates, [R_teo])

        if self.debug:
            print(f"[DEBUG] Theoretical Key rates: {key_rates}")
            print("----------------------------------------------------------------")

        return key_rates
