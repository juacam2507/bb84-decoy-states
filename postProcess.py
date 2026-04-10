import numpy as np


class PostProcess:
    def __init__(self, simulation_parameters: dict, rng: np.random.Generator):
        """
        Initializes a post processing object that emulates error verification and statistical
        assesment of the quantum channel through the classical channel.

        Args:
            simulation_parameters (dict): Dictionary containing the simulation parameters
            rng (np.random.Generator): Random number generator

        """
        self.N = simulation_parameters["N"]
        self.rng = rng
        self.debug = simulation_parameters["debug"]

        self.signal_intensity = simulation_parameters["mu"]
        self.decoy_intensities = tuple(simulation_parameters["decoy_intensities"])
        self.state_num = 1 + len(self.decoy_intensities)

        self.error_correction_efficiency = simulation_parameters[
            "error_correction_efficiency"
        ]

    def compute_state_gain(
        self, receptor_bits: np.ndarray, state_choice: np.ndarray, state: int = 0
    ) -> float:
        """
        Computes the gain Q for a specific state (Vacuum, weak or decoy). The gain is defined as the probability of
        a detection event triggering at the receptor when alice sent with mean photon number mu. In this case, it is
        computed as the relative frequency of the successful detection events (bob_bits != -1) for each
        state choice (state_choice = 0, 1, 2, ...)

        Args:
            receptor_bits (np.ndarray): Array with the detection event decision
            state_choice (np.ndarray): Alice's choice of states.
            state (int, optional): State for which the gain is to be computed. Defaults to 0.

        Returns:
            float: Gain Q of a specific state.
        """

        state_mask = state_choice == state
        state_detection_mask = (receptor_bits != -1) & (state_mask)

        states_sent = np.sum(state_mask)
        states_detected = np.sum(state_detection_mask)

        if self.debug:
            print(f"[DEBUG] State {state} Detected: {states_detected}")
            print(f"[DEBUG] State {state} Sent: {states_sent}")
            print(f"[DEBUG] Gain of state {state} = {states_detected/states_sent}")

        return states_detected / states_sent

    def compute_gains(
        self, receptor_bits: np.ndarray, state_choice: np.ndarray
    ) -> list:
        """
        Computes the gains of all states in an output list.

        Args:
            receptor_bits (np.ndarray): Array with the detection event decision
            state_choice (np.ndarray): Alice's choice of states.

        Returns:
            list: A list with the gains for each state. The index of the state coincides
            with the index of the gain in the list.
        """
        gains = np.array([], dtype=float)

        for i in range(self.state_num):
            q = self.compute_state_gain(
                receptor_bits=receptor_bits, state_choice=state_choice, state=i
            )
            gains = np.append(gains, q)

        if self.debug:
            print(f"[DEBUG] Gains: {gains}")

        return gains

    def basis_reconciliation(
        self, source_basis: np.ndarray, receptor_basis: np.ndarray
    ) -> np.ndarray:
        """
        Obtains the indexes for which basis for source and receptor coincide

        Args:
            source_basis (np.ndarray): Alice's basis choice array
            receptor_basis (np.ndarray): Bob's basis choice array

        Returns:
            np.ndarray: Boolean array with true values on the indexes where basis match.
        """

        matching_basis_mask = source_basis == receptor_basis

        if self.debug:
            print(f"[DEBUG] Matching basis index: {matching_basis_mask}")
            print(
                f"[DEBUG] Basis coincidence rate: {np.sum(matching_basis_mask)/len(matching_basis_mask)}"
            )
        return matching_basis_mask

    def compute_state_qber(
        self,
        sifted_source_bits: np.ndarray,
        sifted_receptor_bits: np.ndarray,
        sifted_state_choice: np.ndarray,
        state: int = 0,
    ) -> float:
        """
        Computes the Quantum Bit Error Rate (QBER) associated to a specified state (Signal or decoy)
        as the relative frecuency of non coincident bits from detected pulses associated to such state and after
        sifting basis.

        Args:
            sifted_source_bits (np.ndarray): Source's bits after sifting basis
            sifted_receptor_bits (np.ndarray): Receptor's bits after sifting basis
            sifted_state_choice (np.ndarray): Choices of states after sifting basis
            state (int, optional): State of interest. Defaults to 0.

        Returns:
            state_qber (float): Quantum bit error rate
        """
        detected_mask = sifted_receptor_bits != -1  # True if the bit was detected
        sifted_detected_state_mask = (sifted_state_choice == state) & (
            detected_mask
        )  # True if the bit is from the state of interest and was detected
        state_error_mask = (sifted_source_bits != sifted_receptor_bits) & (
            sifted_detected_state_mask
        )  # True if the bit corresponds to the state of interest, was detected and the received bit differs from the prepared bit.

        num_state_err = np.sum(
            state_error_mask
        )  # Number of errors associated to the state (Signal or decoy)
        num_states_detected = max(
            1e-15, np.sum(sifted_detected_state_mask)
        )  # Number of detections associated to the state of interes after sifting basis

        if self.debug:
            print(
                f"[DEBUG] Non coincident detected bits associated to state {state} after sifting: {num_state_err}"
            )
            print(
                f"[DEBUG] Total detections of {state} after sifting: {num_states_detected}"
            )
            print(
                f"[DEBUG] Quantum Bit Error Rate associated to state {state}: {num_state_err/num_states_detected}"
            )
        return num_state_err / num_states_detected

    def compute_qbers(
        self,
        sifted_source_bits: np.ndarray,
        sifted_receptor_bits: np.ndarray,
        sifted_state_choice: np.ndarray,
    ) -> list:
        """
        Computes the qbers of all states in an output list.

        Args:
            sifted_source_bits (np.ndarray): Source's bits after sifting basis
            sifted_receptor_bits (np.ndarray): Receptor's bits after sifting basis
            sifted_state_choice (np.ndarray): Choices of states after sifting basis

        Returns:
            list: A list with the gains for each state. The index of the state coincides
            with the index of the gain in the list.
        """
        qbers = np.array([], dtype=float)

        for i in range(self.state_num):
            e = self.compute_state_qber(
                sifted_receptor_bits=sifted_receptor_bits,
                sifted_source_bits=sifted_source_bits,
                sifted_state_choice=sifted_state_choice,
                state=i,
            )
            qbers = np.append(qbers, e)

        if self.debug:
            print(f"[DEBUG] QBER: {qbers}")

        return qbers

    def background_yield_bound(self, gains: list) -> float:

        if len(self.decoy_intensities) != 2:
            return 0.0

        nu_1, nu_2 = self.decoy_intensities

        Q_d1, Q_d2 = gains[1], gains[2]

        denom = nu_1 - nu_2

        if denom <= 0:
            print(f"[DEBUG] Final Y0_L: {0.0}")
            return 0.0

        y_0_l = (nu_1 * Q_d2 * np.exp(nu_2) - nu_2 * Q_d1 * np.exp(nu_1)) / denom
        if self.debug:
            print(f"[DEBUG] Computed Y0_L: {y_0_l}")
            print(f"[DEBUG] Final Y0_L: {np.clip(y_0_l, 0.0, 1.0)}")

        return float(
            np.clip(y_0_l, 0.0, 1.0)
        )  # Bounds the yield to values between 0 and 1

    def single_photon_yield_bound(self, gains: list, y_0_l: float) -> float:
        if len(self.decoy_intensities) != 2:
            return 0.0

        mu = self.signal_intensity
        nu_1, nu_2 = self.decoy_intensities
        Q_s, Q_d1, Q_d2 = gains[0], gains[1], gains[2]

        denom = (nu_1 - nu_2) * (mu - (nu_1 + nu_2))

        if denom <= 0:
            return 0.0

        y_1_l = (mu / denom) * (
            Q_d1 * np.exp(nu_1)
            - Q_d2 * np.exp(nu_2)
            - ((nu_1**2 - nu_2**2) / mu**2) * (Q_s * np.exp(mu) - y_0_l)
        )
        if self.debug:
            print(f"[DEBUG] Computed Y1_L: {y_1_l}")
            print(f"[DEBUG] Final Y1_L: {y_1_l}")

        return float(np.clip(y_1_l, 0.0, 1.0))

    def single_photon_error_bound(
        self, gains: list, qbers: list, y_1_l: float
    ) -> float:
        if len(self.decoy_intensities) != 2 or y_1_l <= 0.0:
            if self.debug:
                print(f"[DEBUG] Final e1_u: {0.5}")
            return 0.5

        nu_1, nu_2 = self.decoy_intensities

        Q_d1, Q_d2 = gains[1], gains[2]

        E_d1, E_d2 = qbers[1], qbers[2]

        denom = (nu_1 - nu_2) * y_1_l

        if denom <= 0.0:
            if self.debug:
                print(f"[DEBUG] Final e1_u: {0.5}")
            return 0.5

        e_1_u = (E_d1 * Q_d1 * np.exp(nu_1) - E_d2 * Q_d2 * np.exp(nu_2)) / denom

        if self.debug:
            print(f"[DEBUG] Computed e1_u: {e_1_u}")
            print(f"[DEBUG] Final e1_u: {float(np.clip(e_1_u, 0.0, 0.5))}")

        return float(np.clip(e_1_u, 0.0, 0.5))

    def shannon_entropy(self, x: float):
        if x > 0 and x < 1:
            return -x * np.log2(x) - (1 - x) * np.log2(1 - x)
        else:
            return 0

    def secure_key_rate(
        self, gains: list, qbers: list, y_1_l: float, e_1_u: float
    ) -> float:
        mu = self.signal_intensity
        Q_s = gains[0]
        E_s = qbers[0]

        Q_1 = max(0.0, y_1_l * mu * np.exp(-mu))

        R = 0.5 * (
            -Q_s * self.error_correction_efficiency * self.shannon_entropy(E_s)
            + Q_1 * (1 - self.shannon_entropy(e_1_u))
        )
        if self.debug:
            print(f"[DEBUG] Computed Secure Key Rate: {R}")
            print(f"[DEBUG] Final Secure Key Rate: {max(0.0, float(R))}")

        return max(0.0, float(R))
