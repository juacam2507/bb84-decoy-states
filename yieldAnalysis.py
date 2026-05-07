import numpy as np
from quantumChannel import QuantumChannel
from classicalChannel import ClassicalChannel
from bb88_simulator import Simulator
from securityAnalysis import SecurityAnalysis


class YieldAnalysis:
    def __init__(
        self,
        simulation_parameters: dict,
        yield_sweep_params: dict,
        rng: np.random.Generator,
    ) -> None:

        self.simulation_parameters = simulation_parameters
        self.yield_sweep_params = yield_sweep_params
        self.rng = rng

        self.iterations = self.simulation_parameters["iterations"]
        self.debug = self.simulation_parameters["debug"]

        self.photon_nums = self.yield_sweep_params["photon_nums"]
        self.d = self.yield_sweep_params["distance"]

    def run_channel(self):

        quantum_channel = QuantumChannel(self.simulation_parameters, self.rng, l=self.d)
        classical_channel = ClassicalChannel(self.simulation_parameters)
        simulator = Simulator(
            quantum_channel=quantum_channel, classical_channel=classical_channel
        )
        security_analysis = SecurityAnalysis(quantum_channel=quantum_channel)

        state_gains, _ = simulator.run(iterations=self.iterations)

        yields_mu, yields_nu, yields_teo = security_analysis.compute_state_yields(
            photon_nums=self.photon_nums, state_gains=state_gains
        )

        signal_data = np.vstack(yields_mu)
        decoy_data = np.vstack(yields_nu)
        theoretical_yield = np.vstack(yields_teo)
         
        if self.debug:
            print(f"[DEBUG] Signal data: {signal_data}")
            print(f"[DEBUG] Decoy data: {decoy_data}")
            print(f"[DEBUG] Theoretical data: {theoretical_yield}")
            print("----------------------------------------------------------------")

        return yields_mu, yields_nu, theoretical_yield
