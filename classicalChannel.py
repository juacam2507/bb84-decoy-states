import numpy as np


class ClassicalChannel:
    def __init__(self, simulation_parameters: dict):
        """
        Initialize post-processing for decoy-state BB84 parameter estimation.

        Performs basis sifting, computes per-state gains/QBER, and extracts security
        bounds (Y₀ᴸ, Y₁ᴸ, e₁ᵘ) using two-decoy analysis. Computes asymptotic secure
        key rate via GLLP formula.

        Parameters
        ----------
        simulation_parameters : dict
            Required keys:
            - "N": int, number of pulses.
            - "debug": bool, verbose output.
            - "mu": float, signal state mean photon number.
            - "decoy_intensities": list[float], exactly 2 decoy intensities.
            - "error_correction_efficiency": float, f(EC) ∈ [1.0, 1.16].
        rng : np.random.Generator
            Random number generator for privacy amplification simulation.
        """
        self.N = simulation_parameters["N"]
        self.debug = simulation_parameters["debug"]

        self.signal_intensity = simulation_parameters["mu"]
        self.decoy_intensities = simulation_parameters["decoy_intensities"]
        self.intensities = np.array([self.signal_intensity] + self.decoy_intensities, dtype= np.float64)
        self.state_num = len(self.intensities)
        
        self.error_correction_efficiency = simulation_parameters[
            "error_correction_efficiency"
        ]

    def compute_state_gain(
        self, receptor_bits: np.ndarray, state_choice: np.ndarray, state: int = 0
    ) -> float:
        """
        Compute empirical gain Qᵢ for state i.

        Gain is the sifted detection probability: :math:`Q_i = \\frac{N_{\\text{det},i}}{N_i}`

        Parameters
        ----------
        receptor_bits : npt.NDArray[np.int_]
            Bob's bits, shape (N,) ∈ {-1, 0, 1}.
        state_choice : npt.NDArray[np.int_]
            Alice's state labels, shape (N,) ∈ {0, 1, 2, ...}.
        state : int, default=0
            State index (0=signal, 1/2=decoys).

        Returns
        -------
        float
            Gain Qᵢ ∈ [0, 1].
        """

        state_mask = state_choice == state
        state_detection_mask = (receptor_bits != -1) & (state_mask)

        states_sent = np.sum(state_mask)
        states_detected = np.sum(state_detection_mask)

        if self.debug:
            print(f"[DEBUG] State {state} Detected: {states_detected}")
            print(f"[DEBUG] State {state} Sent: {states_sent}")
            print(f"[DEBUG] Gain of state {state} = {states_detected/states_sent}")
            print("----------------------------------------------------------------")

        return states_detected / states_sent

    def compute_gains(
        self, receptor_bits: np.ndarray, state_choice: np.ndarray
    ) -> np.ndarray:
        """
        Compute gains Q = [Q₀, Q₁, Q₂, ...] for all states.

        Parameters
        ----------
        receptor_bits : npt.NDArray[np.int_]
            Bob's detection outcomes, shape (N,).
        state_choice : npt.NDArray[np.int_]
            Alice's state sequence, shape (N,).

        Returns
        -------
        npt.NDArray[np.float64]
            Array of shape (state_num,) containing gains Qᵢ for each state.
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
        Sift pulses with matching bases (Z↔Z, X↔X).

        Parameters
        ----------
        source_basis : npt.NDArray[np.int_]
            Alice's bases, shape (N,) ∈ {0, 1}.
        receptor_basis : npt.NDArray[np.int_]
            Bob's bases, shape (N,) ∈ {0, 1}.

        Returns
        -------
        npt.NDArray[np.bool_]
            Sifting mask, shape (N,) where True indicates basis match.
        """

        matching_basis_mask = source_basis == receptor_basis

        if self.debug:
            print(f"[DEBUG] Matching basis index: {matching_basis_mask}")
            print(
                f"[DEBUG] Basis coincidence rate: {np.sum(matching_basis_mask)/len(matching_basis_mask)}"
            )
            print("----------------------------------------------------------------")
        return matching_basis_mask

    def compute_state_qber(
        self,
        sifted_source_bits: np.ndarray,
        sifted_receptor_bits: np.ndarray,
        sifted_state_choice: np.ndarray,
        state: int = 0,
    ) -> float:
        """
        Compute QBER Eᵢ for state i among sifted, detected pulses.

        QBER: :math:`E_i = \\frac{\\text{Number of errors in state } i}{\\text{Number detections in state } i}`

        Parameters
        ----------
        sifted_source_bits : npt.NDArray[np.int_]
            Alice's sifted bits, shape (M,).
        sifted_receptor_bits : npt.NDArray[np.int_]
            Bob's sifted bits, shape (M,) ∈ {-1, 0, 1}.
        sifted_state_choice : npt.NDArray[np.int_]
            Sifted state labels, shape (M,).
        state : int, default=0
            State index to analyze.

        Returns
        -------
        float
            QBER Eᵢ ∈ [0, 0.5].
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
            print("----------------------------------------------------------------")
        return num_state_err / num_states_detected

    def compute_qbers(
        self,
        sifted_source_bits: np.ndarray,
        sifted_receptor_bits: np.ndarray,
        sifted_state_choice: np.ndarray,
    ) -> np.ndarray:
        """
        Compute QBERs E = [E₀, E₁, E₂, ...] for all states.

        Parameters
        ----------
        sifted_source_bits : npt.NDArray[np.int_]
            Sifted Alice bits.
        sifted_receptor_bits : npt.NDArray[np.int_]
            Sifted Bob bits.
        sifted_state_choice : npt.NDArray[np.int_]
            Sifted state labels.

        Returns
        -------
        npt.NDArray[np.float64]
            Array of shape (state_num,) containing QBERs Eᵢ.
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
            print("----------------------------------------------------------------")

        return qbers

    