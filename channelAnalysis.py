import numpy as np


class ChannelAnalysis:
    def __init__(self, simulation_parameters: dict):

        self.debug = simulation_parameters["debug"]

        self.simulation_parameters = simulation_parameters
        self.signal_intensity = simulation_parameters["mu"]
        self.decoy_intensities = simulation_parameters["decoy_intensities"]
        self.intensities = np.array(
            [self.signal_intensity] + self.decoy_intensities, dtype=np.float64
        )
        self.state_num = len(self.intensities)

        self.y_0 = simulation_parameters["detector_properties"]["dark_count_rate"]
        self.e_0 = simulation_parameters["detector_properties"]["dark_count_error"]
        self.e_d = simulation_parameters["detector_properties"]["detector_error"]

        self.error_correction_efficiency = simulation_parameters[
            "error_correction_efficiency"
        ]

    def compute_theoretical_gains(self, eta: float) -> np.ndarray:
        """
        Compute theoretical channel gains Q_μ for all intensities.

        Gain for mean photon number μ follows threshold detector model:

        .. math:: Q_\\mu = Y_0 + 1 - e^{-\\eta \\mu}

        where Y₀ is the dark count probability and ημ is the expected number of
        signal photons arriving at the detector.

        Parameters
        ----------
        eta : float
            Overall channel + detector efficiency ∈ [0, 1].

        Returns
        -------
        npt.NDArray[np.float64]
            Array of shape (state_num,) containing theoretical gains Q_μ
            for signal and decoy intensities, ordered as self.intensities.

        Notes
        -----
        - Includes dark counts via Y₀ (self.y_0).

        - Approximates P(click) = P(dark) + P(signal ≥ 1). Valid when

           .. math:: P(dark + signal) = Y_0 e^{-\\eta \\mu} ≪ 1

          which is typical for QKD parameters.

        - Matches empirical gains :math:`\\hat{Q}_\\mu` in infinite-statistics limit.

        - Used for validation against Monte Carlo simulation results.
        """
        Q_teo = self.y_0 + 1 - np.exp(-eta * self.intensities)

        if self.debug:
            print(f"[DEBUG] Theoretical gains of the channel: {Q_teo}")

        return Q_teo

    def compute_theoretical_qbers(self, eta: float, Q_teo: np.ndarray) -> np.ndarray:
        """
        Compute theoretical QBERs E_μ for all intensities.

        Theoretical QBER for threshold detectors:

        .. math:: E_\\mu = \\frac{e_0 Y_0 + e_d (1-e^{-\\eta \\mu})}{Q_\\mu}

        Parameters
        ----------
        eta : float
            Channel efficiency ∈ [0, 1].

        Returns
        -------
        npt.NDArray[np.float64]
            Array of shape (state_num,) with theoretical QBERs E_μ.
        """

        error_counts = self.e_0 * self.y_0 + self.e_d * (
            1 - np.exp(-eta * self.intensities)
        )

        E_teo = np.where(Q_teo > 1e-15, error_counts / Q_teo, 0.0)

        if self.debug:
            print(f"[DEBUG] Theoretical Errors of the channel: {E_teo}")

        return E_teo

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

    def compute_key_rate(self, gains: np.ndarray, qbers: np.ndarray) -> float:
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

        y_0_l = self.background_yield_bound(gains=gains)
        y_1_l = self.single_photon_yield_bound(gains=gains, y_0_l=y_0_l)
        e_1_u = self.single_photon_error_bound(gains=gains, qbers=qbers, y_1_l=y_1_l)

        Q_1 = max(0.0, y_1_l * mu * np.exp(-mu))

        R = 0.5 * (
            -Q_s * self.error_correction_efficiency * self.shannon_entropy(E_s)
            + Q_1 * (1 - self.shannon_entropy(e_1_u))
        )
        if self.debug:
            print(f"[DEBUG] Computed Secure Key Rate: {R}")
            print(f"[DEBUG] Final Secure Key Rate: {max(0.0, float(R))}")

        return max(0.0, float(R))

    def compute_state_eta(self, gains: np.ndarray) -> np.ndarray:
        """
            Compute the effective transmission parameter (:math:`\\eta`) for each state.

            This function estimates the transmission efficiency for each state by
            inverting the gain model, taking into account background counts and
            state intensities.

            The computation is defined as:

            .. math::

                \\eta = 
                \\begin{cases}
                -\\frac{\\ln\\left(1 - (G - Y_0)\\right)}{\\mu}, & \\text{if } \\mu > 10^{-15} \\\\
                0, & \\text{otherwise}
                \\end{cases}

            where:

            - :math:`G` is the observed gain,
            - :math:`Y_0` is the background (dark count) contribution,
            - :math:`\\mu` is the state intensity,
            - :math:`\\eta` is the effective transmission efficiency.

            Parameters
            ----------
            gains : np.ndarray
                Array of observed gains (:math:`G`) for each state. Must have the same
                shape as `self.intensities`.

            Returns
            -------
            np.ndarray
                Array of transmission efficiencies (:math:`\\eta`) for each state.

            Notes
            -----
            - Values of intensity :math:`\\mu \\leq 10^{-15}` are treated as zero to
            avoid numerical instability due to division by very small numbers.
            - The argument of the logarithm must remain positive:
            :math:`1 - (G - Y_0) > 0`.
        """

        eta_state = np.where(
            self.intensities > 1e-15,
            -np.log(1 - (gains - self.y_0)) / self.intensities,
            0.0,
        )
        if self.debug:
            print(f"[DEBUG] Computed Transmission Efficiencies: {eta_state}")

        return eta_state

    def compute_state_yield_n(
        self, photon_number: int, eta_state: np.ndarray
    ) -> np.ndarray:
        
        """
            Compute the yield for an n-photon state.

            This function calculates the probability that at least one photon is
            detected given a state with :math:`n` photons and transmission efficiency
            :math:`\\eta`.

            The yield is defined as:

            .. math::

                Y_n = 1 - (1 - \\eta)^n

            where:

            - :math:`n` is the photon number,
            - :math:`\\eta` is the transmission efficiency,
            - :math:`Y_n` is the probability that at least one photon is detected.

            This follows from the assumption that each photon is independently
            transmitted with probability :math:`\\eta`. The probability that none are
            detected is :math:`(1 - \\eta)^n`, so the yield is its complement.

            Parameters
            ----------
            photon_number : int
                Number of photons (:math:`n`). Must be a non-negative integer.

            eta_state : np.ndarray
                Array of transmission efficiencies (:math:`\\eta`) for each state.

            Returns
            -------
            np.ndarray
                Array of yields (:math:`Y_n`) for each state.

            Notes
            -----
            - Assumes independent transmission events for each photon.
            - Typically, :math:`0 \\leq \\eta \\leq 1`.
        """

        state_yields = 1 - (1 - eta_state) ** photon_number + self.y_0

        return state_yields

    def compute_effective_yields(self, photon_nums: list, gains: np.ndarray) -> np.ndarray:
        
        eta_state = self.compute_state_eta(gains=gains)
        for num in photon_nums: 
            state_yield_n = self.compute_state_yield_n(photon_number=num, eta_state=eta_state)

        return state_yield_n