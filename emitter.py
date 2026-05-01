import numpy as np

class Emitter:
    def __init__(self, simulation_parameters: dict, rng: np.random.Generator):
        """
        Initialize the photon source for the BB84 decoy-state protocol.

        The source is responsible for generating the quantum signal pulses,
        including random bit values, basis choices, decoy/signal states, and
        photon-number distributions according to the simulation parameters.

        Parameters
        ----------
        simulation_parameters : dict
            Dictionary containing the simulation configuration. Expected keys include:
            - "N": int, number of transmitted pulses.
            - "mu": float, mean photon number for the signal state.
            - "decoy_intensities": list of floats, mean photon numbers for each decoy state.
            - "decoy_rate": float, fraction of decoy pulses among all pulses.
            - "debug": bool, enables verbose output for debugging.
        rng : np.random.Generator
            NumPy random generator for stochastic sampling and reproducibility.
        """

        self.N = simulation_parameters["N"]
        self.mu = simulation_parameters["mu"]
        self.decoy_intensities = simulation_parameters["decoy_intensities"]
        self.state_probs = simulation_parameters["state_probs"]
        self.debug = simulation_parameters["debug"]
        self.rng = rng

    def generate_bit_seq(self) -> np.ndarray:
        """
        Generate a random bit sequence representing Alice’s key bits.

        Each bit is chosen uniformly at random in {0, 1} and corresponds to
        a qubit polarization or phase value to be encoded in the optical pulse.

        Returns
        -------
        np.ndarray
            Array of shape (N,) containing random bits (0 or 1) for each pulse.
        """

        source_bits = self.rng.integers(0, 2, self.N)

        if self.debug:
            print(f"[DEBUG] Source bits: {source_bits}")
            print("----------------------------------------------------------------")

        return source_bits

    def generate_basis_seq(self) -> np.ndarray:
        """
        Generate the random basis sequence used for state preparation.

        Bases are represented as 0 for rectilinear (Z) and 1 for diagonal (X),
        each with equal probability unless specified otherwise.

        Returns
        -------
        np.ndarray
            Array of shape (N,) containing the basis choice for each bit
            (0 = rectilinear, 1 = diagonal).
        """

        source_basis = self.rng.integers(0, 2, self.N)

        if self.debug:
            print(f"[DEBUG] Source basis: {source_basis}")
            print("----------------------------------------------------------------")

        return source_basis

    def generate_state_seq(self) -> np.ndarray:
        """
        Select the signal or decoy state for each pulse.

        The method assigns an integer label to each state:
        0 for the signal state and 1, 2, ... for decoy states. Probabilities are
        determined by the decoy rate and normalized automatically.

        Returns
        -------
        np.ndarray
            Array of shape (N,) where each entry indicates the chosen pulse state
            (0 = signal, 1... = decoy states).
        """
        if not np.isclose(np.sum(self.state_probs), 1.0, atol=1e-10):
            raise ValueError(f"state probabilities should sum to ~ 1. Actual: {sum(np.array(self.state_probs)):.15g}")

        decoy_num = int(len(self.decoy_intensities))

        state_index = [0] + list(np.arange(1, decoy_num + 1))
        state_probs = self.state_probs
        state_probs = np.array(state_probs, dtype=np.float64) / np.sum(
            state_probs
        )  # Normalization of probabilities

        state_sequence = self.rng.choice(
            np.asarray(state_index), size=self.N, p=state_probs
        )

        if self.debug:
            print(f"[DEBUG] State choice: {state_sequence}")
            print("----------------------------------------------------------------")

        return state_sequence

    def generate_photon_number_seq(self, state_choice: np.ndarray) -> np.ndarray:
        """
        Sample the number of photons emitted in each pulse.

        For each pulse, the photon number is drawn from a Poisson distribution
        whose mean is determined by the selected state’s intensity (signal or decoy).

        Parameters
        ----------
        state_choice : np.ndarray
            Array of integers specifying which decoy or signal state was chosen
            for each pulse.

        Returns
        -------
        np.ndarray
            Photon number sequence of shape (N,) giving the number of photons 
            generated in each pulse.
        """

        intensities = [self.mu] + self.decoy_intensities  # Intensity list

        photon_nums = np.zeros(self.N, dtype=int)

        for i in range(len(intensities)):
            intensity_mask = state_choice == i
            photon_nums[intensity_mask] = self.rng.poisson(
                intensities[i], size=np.sum(intensity_mask)
            )

        if self.debug:
            print(f"[DEBUG] Photon numbers: {photon_nums}")
            print("----------------------------------------------------------------")

        return photon_nums

    def generate_pulses(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate the full set of source parameters for a round of transmission.

        This method combines all steps of state preparation, returning the
        quantum pulse information: key bits, encoding bases, signal/decoy state
        labels, and photon numbers per pulse.

        Returns
        -------
        tuple of np.ndarray
            A 4-tuple `(bits, basis, state, photons)` where:
            - bits : np.ndarray
                Random bit sequence for each pulse.
            - basis : np.ndarray
                Random basis (0 = Z, 1 = X) used for encoding.
            - state : np.ndarray
                Signal/decoy state index for each pulse.
            - photons : np.ndarray
                Sampled photon numbers from Poisson statistics.
        """
        source_bit_seq = self.generate_bit_seq()
        source_basis_seq = self.generate_basis_seq()
        source_state_seq = self.generate_state_seq()
        photon_nums = self.generate_photon_number_seq(source_state_seq)

        return source_bit_seq, source_basis_seq, source_state_seq, photon_nums
