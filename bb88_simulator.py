import numpy as np
from source import Source
from detector import Detector
from postProcess import PostProcess
from quantumChannel import QuantumChannel
from tqdm import tqdm


class Simulator:
    def __init__(self, simulation_parameters: dict, rng: np.random.Generator):
        """
        Initialize the QKD simulation environment.

        This constructor sets up the parameters and random number generator to be used
        during the quantum key distribution (QKD) simulation. The total number of states
        (signal, decoy, vacuum, etc.) is derived from the specified decoy intensities.
        A debug mode may optionally provide detailed console output during simulation.

        Parameters
        ----------
        simulation_parameters : dict
            Dictionary containing the simulation configuration. Expected keys include:
            - "decoy_intensities": list of floats representing decoy and signal states.
            - "debug": bool flag to enable or disable debug output.
            Other protocol-specific parameters (e.g., basis probabilities) may also be included.

        rng : np.random.Generator
            Instance of NumPy's random number generator for reproducibility in stochastic processes.

        Attributes
        ----------
        simulation_parameters : dict
            Stores all configuration parameters used in the simulation.

        rng : np.random.Generator
            Random number generator used for all probabilistic events in the simulation.

        state_num : int
            Number of source states, derived from `len(decoy_intensities) + 1`.

        debug : bool
            Flag to enable verbose output for debugging or performance analysis.
        """

        self.simulation_parameters = simulation_parameters
        self.rng = rng
        self.state_num = len(simulation_parameters["decoy_intensities"]) + 1
        self.debug = simulation_parameters["debug"]

    def run(self, iterations: int, quantum_channel: QuantumChannel, post_process : PostProcess) -> tuple:
        """
        Execute the quantum key distribution (QKD) simulation for a given channel length.

        At each iteration, this method simulates Alice’s pulse generation, channel transmission,
        and Bob’s detection process. It then performs post-processing steps such as basis
        reconciliation, QBER computation, and secure key rate estimation over multiple
        simulation rounds.

        Parameters
        ----------
        l : float
            Channel length (typically in kilometers) representing optical fiber distance
            between Alice and Bob. Affects transmission efficiency and overall key rate.

        iter : int
            Number of Monte Carlo iterations to average over.

        Returns
        -------
        R : float
            Estimated secure key rate (in bits per pulse or bits per signal state), averaged
            over the specified number of iterations and adjusted for statistical bounds.

        Notes
        -----
        - The simulation follows a decoy-state BB84-like model.
        - Intermediate results such as average gains (Q) and QBER (E) are accumulated and
          averaged across all iterations.
        - When debug mode is enabled, detailed per-iteration statistics are printed.
        """

        # Declare objects
        Q_cum = np.zeros(self.state_num, dtype=float)
        E_cum = np.zeros(self.state_num, dtype=float)
        Q_av = np.zeros(self.state_num, dtype=float)
        E_av = np.zeros(self.state_num, dtype=float)

        # Compute the channel efficiency
        
        for iter in tqdm(range(iterations), desc="Iterations"):

            alice_bits, alice_basis, state_choice, bob_basis, bob_bits = quantum_channel.send_pulses()

            # Compute gains for each state (Signal, weak, vacuum)
            gains = post_process.compute_gains(bob_bits, state_choice)

            # Perform basis reconciliation
            matching_basis_mask = post_process.basis_reconciliation(
                alice_basis, bob_basis
            )
            sifted_alice_bits = alice_bits[matching_basis_mask]
            sifted_bob_bits = bob_bits[matching_basis_mask]
            sifted_state_choice = state_choice[matching_basis_mask]

            # Compute QBER for each state
            qbers = post_process.compute_qbers(
                sifted_source_bits=sifted_alice_bits,
                sifted_receptor_bits=sifted_bob_bits,
                sifted_state_choice=sifted_state_choice,
            )
            Q_cum += gains
            E_cum += qbers

        Q_av = Q_cum / iterations
        E_av = E_cum / iterations

        if self.debug:
            print(f"[DEBUG] Average gains after {iterations} iterations: {Q_av}")
            print(f"[DEBUG] Average QBER after {iterations} iterations: {E_av}")

        return Q_av, E_av
