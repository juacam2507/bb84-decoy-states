import numpy as np


class PostProcess:
    def __init__(self, simulation_parameters: dict, rng: np.random.Generator):
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

            QBER: :math:`E_i = \\frac{\\text{# errors in state } i}{\\text{# detections in state } i}`

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

        return qbers

    def background_yield_bound(self, gains: np.ndarray) -> float:
        """
            Estimate lower bound on vacuum yield Y₀ᴸ using two decoys.

            :math:`Y_0^L = \\frac{\\nu_1 Q_{d2} e^{\\nu_2} - \\nu_2 Q_{d1} e^{\\nu_1}}{\\nu_1 - \\nu_2}`

            Parameters
            ----------
            gains : npt.NDArray[np.float64]
                Observed gains Q = [Qₛ, Q_{d1}, Q_{d2}, ...].

            Returns
            -------
            float
                Lower bound Y₀ᴸ ∈ [0, 1] (clipped).
        """

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

    def single_photon_yield_bound(self, gains: np.ndarray, y_0_l: float) -> float:
        """
            Estimate lower bound on single-photon yield Y₁ᴸ.

            :math:`Y_1^L = \\frac{\\mu}{(\\nu_1-\\nu_2)(\\mu-(\\nu_1+\\nu_2))}\\left[Q_{d1}e^{\\nu_1}-Q_{d2}e^{\\nu_2}-\\frac{\\nu_1^2-\\nu_2^2}{\\mu^2}(Q_se^\\mu-Y_0^L)\\right]`

            Parameters
            ----------
            gains : npt.NDArray[np.float64]
                Observed gains.
            y_0_l : float
                Vacuum yield lower bound.

            Returns
            -------
            float
                Single-photon yield lower bound Y₁ᴸ ∈ [0, 1].
        """

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
        self, gains: np.ndarray, qbers: np.ndarray, y_1_l: float
    ) -> float:
        
        """
            Estimate upper bound on single-photon error rate e₁ᵘ.

            :math:`e_1^u = \\frac{E_{d1}Q_{d1}e^{\\nu_1}-E_{d2}Q_{d2}e^{\\nu_2}}{(\\nu_1-\\nu_2)Y_1^L}`

            Parameters
            ----------
            gains : npt.NDArray[np.float64]
                Observed gains.
            qbers : npt.NDArray[np.float64]
                Observed QBERs.
            y_1_l : float
                Single-photon yield lower bound.

            Returns
            -------
            float
                Single-photon phase error upper bound e₁ᵘ ∈ [0, 0.5].
        """
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

        """
            Binary Shannon entropy function H(x) = -xlog₂(x)-(1-x)log₂(1-x).

            Parameters
            ----------
            x : float
                Probability ∈ [0, 1].

            Returns
            -------
            float
                H(x), with H(0) = H(1) = 0 by continuity.
        """
        if x > 0 and x < 1:
            return -x * np.log2(x) - (1 - x) * np.log2(1 - x)
        else:
            return 0

    def secure_key_rate(
        self, gains: np.ndarray, qbers: np.ndarray, y_1_l: float, e_1_u: float
    ) -> float:
        """
            Compute asymptotic secure key rate (GLLP formula).

            :math:`R = \\frac{1}{2} \\left[ Q_1 \\left(1 - h(e_1^u)\\right) - Q_\\mu f(E_\\mu) h(E_\\mu) \\right]`

            where :math:`Q_1 = Y_1^L \\mu e^{-\\mu}`.

            Parameters
            ----------
            gains : npt.NDArray[np.float64]
                Observed gains [Qₛ, Q_{d1}, Q_{d2}, ...].
            qbers : npt.NDArray[np.float64]
                Observed QBERs [Eₛ, E_{d1}, E_{d2}, ...].
            y_1_l : float
                Single-photon yield lower bound.
            e_1_u : float
                Single-photon phase error upper bound.

            Returns
            -------
            float
                Secure key rate R ≥ 0 (bits per pulse).
        """
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
