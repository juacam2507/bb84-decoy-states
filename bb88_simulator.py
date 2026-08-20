import numpy as np
from classicalChannel import ClassicalChannel
from quantumChannel import QuantumChannel
from tqdm import tqdm


class Simulator:
    def __init__(
        self, quantum_channel: QuantumChannel, classical_channel: ClassicalChannel
    ):
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
        self.quantum_channel = quantum_channel
        self.classical_channel = classical_channel
        self.simulation_parameters = quantum_channel.simulation_parameters
        self.N = self.simulation_parameters["N"]
        self.protocol = self.simulation_parameters["protocol"]
        self.signal_intensity = self.simulation_parameters["mu"]

        if self.protocol == "bb84":
            self.intensities = [self.signal_intensity]
        else:
            self.decoy_intensities = self.simulation_parameters["decoy_intensities"]
            self.intensities = np.array(
                [self.signal_intensity] + self.decoy_intensities, dtype=np.float64
            )
        self.state_num = len(self.intensities)
        self.debug = self.simulation_parameters["debug"]

    def run(self, iterations: int) -> tuple:
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
        state_gains = np.zeros((self.state_num, iterations), dtype=float)
        state_errors = np.zeros((self.state_num, iterations), dtype=float)

        signal_gains = []
        decoy_gains = []
        vacuum_gains = []

        signal_errors = []
        decoy_errors = []
        vacuum_errors = []

        # Compute the channel efficiency

        for iter in tqdm(range(iterations), desc="Iterations"):

            alice_bits, alice_basis, state_choice, bob_basis, bob_bits = (
                self.quantum_channel.send_pulses()
            )
            if self.protocol == "rfi-bb84":
                gains = self.classical_channel.compute_gains_rfi(
                    receptor_bits=bob_bits,
                    source_basis=alice_basis,
                    receptor_basis=bob_basis,
                    state_choice=state_choice,
                )

            else:
                gains = self.classical_channel.compute_gains(bob_bits, state_choice)

            if self.protocol == "bb84":
                signal_gains.append(gains[0])
            else:
                signal_gains.append(gains[0])
                decoy_gains.append(gains[1])
                vacuum_gains.append(gains[2])

            # Perform basis reconciliation
            matching_basis_mask = self.classical_channel.basis_reconciliation(
                alice_basis, bob_basis
            )
            sifted_alice_bits = alice_bits[matching_basis_mask]
            sifted_bob_bits = bob_bits[matching_basis_mask]
            sifted_state_choice = state_choice[matching_basis_mask]

            # Compute QBER for each state
            qbers = self.classical_channel.compute_qbers(
                sifted_source_bits=sifted_alice_bits,
                sifted_receptor_bits=sifted_bob_bits,
                sifted_state_choice=sifted_state_choice,
            )
            if self.protocol == "bb84":
                signal_errors.append(qbers[0])
            else:
                signal_errors.append(qbers[0])
                decoy_errors.append(qbers[1])
                vacuum_errors.append(qbers[2])

        if self.protocol == "bb84":
            state_gains = np.vstack([signal_gains])
            state_errors = np.vstack([signal_errors])
        else:
            state_gains = np.vstack([signal_gains, decoy_gains, vacuum_gains])
            state_errors = np.vstack([signal_errors, decoy_errors, vacuum_errors])

        if self.debug:
            print(f"[DEBUG] Gains after {iterations} iterations:\n {state_gains}")
            print(f"[DEBUG] QBERs after {iterations} iterations:\n {state_errors}")
            print("----------------------------------------------------------------")

        return state_gains, state_errors
