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

    def compute_state_eta(self, gains: np.ndarray) -> tuple:
        """
            Compute the effective transmission efficiencies for the signal and decoy states.

            This function estimates the transmission efficiency for the signal and decoy
            states by inverting the gain model and accounting for the background
            contribution ``self.y_0``.

            The efficiencies are computed as:

            .. math::

                \\eta_\\mu = -\\frac{\\ln\\left(1 - (Q_\\mu - Y_0)\\right)}{\\mu}

            .. math::

                \\eta_\\nu = -\\frac{\\ln\\left(1 - (Q_\\nu - Y_0)\\right)}{\\nu}

            where:

            - :math:`Q_\\mu` is the observed gain for the signal state,
            - :math:`Q_\\nu` is the observed gain for the decoy state,
            - :math:`Y_0` is the background (dark count) contribution,
            - :math:`\\mu` is the signal-state intensity,
            - :math:`\\nu` is the decoy-state intensity,
            - :math:`\\eta_\\mu` and :math:`\\eta_\\nu` are the corresponding
            transmission efficiencies.

            Parameters
            ----------
            gains : np.ndarray
                Array containing the observed gains for the states. The function expects
                the first two entries to correspond to ``Q_mu`` and ``Q_nu``.

            Returns
            -------
            tuple
                A tuple containing:

                - eta_mu : float
                    Effective transmission efficiency for the signal state.
                - eta_nu : float
                    Effective transmission efficiency for the decoy state.

            Notes
            -----
            - The function currently unpacks ``gains`` as ``Q_mu, Q_nu, _``.
            - The function also unpacks ``self.intensities`` as ``mu, nu, _``.
            - If ``self.debug`` is ``True``, the computed efficiencies are printed.
            - The logarithm arguments must satisfy :math:`1 - (Q - Y_0) > 0`.
        """ 
        Q_mu, Q_nu, _ = gains
        mu, nu, _ = self.intensities

        eta_mu = -np.log(1 - (Q_mu - self.y_0)) / mu
        eta_nu = -np.log(1 - (Q_nu - self.y_0)) / nu

        if self.debug:
            print(f"[DEBUG] Efficiency of signal state: {eta_mu}")
            print(f"[DEBUG] Efficiency of decoy state:{eta_nu}")

        return eta_mu, eta_nu

    def compute_state_yields(self, photon_nums: list, gains: np.ndarray) -> tuple:
        """
        Compute the single‑photon yields for the signal and decoy states at given photon numbers.

        This function computes the expected yield for each state (signal and decoy)
        as a function of photon number, using the effective transmission efficiencies
        obtained from the gains.

        The single‑photon yield for each state is given by:

        .. math::

            Y_n^{\\mu} = Y_0 + 1 - (1 - \\eta_\\mu)^n

        .. math::

            Y_n^{\\nu} = Y_0 + 1 - (1 - \\eta_\\nu)^n

        where:

        - :math:`n` is the photon number,
        - :math:`Y_0` is the background (dark count) contribution,
        - :math:`\\eta_\\mu` is the transmission efficiency of the signal state,
        - :math:`\\eta_\\nu` is the transmission efficiency of the decoy state,
        - :math:`Y_n^{\\mu}` is the yield of the signal state for :math:`n` photons,
        - :math:`Y_n^{\\nu}` is the yield of the decoy state for :math:`n` photons.

        Parameters
        ----------
        photon_nums : list of int
            List of photon numbers :math:`n` for which yields are to be computed.
            Each element must be a non‑negative integer.

        gains : np.ndarray
            Array of observed gains used to compute transmission efficiencies
            via :meth:`compute_state_eta`. The first two entries are treated as
            :math:`Q_\\mu` (signal) and :math:`Q_\\nu` (decoy).

        Returns
        -------
        tuple
            A tuple containing:

            - yields_mu : list of float
                Yields for the signal state :math:`(Y_n^{\\mu})` at each photon number.
            - yields_nu : list of float
                Yields for the decoy state :math:`(Y_n^{\\nu})` at each photon number.

        Notes
        -----
        - The function calls :meth:`compute_state_eta` to obtain :math:`\\eta_\\mu`
        and :math:`\\eta_\\nu` from the input gains.
        - If :attr:`self.debug` is ``True``, the computed yield lists are printed.
        - The background term :math:`Y_0` is assumed to be stored in ``self.y_0``.
        """

        yields_mu = []
        yields_nu = []
        eta_mu, eta_nu = self.compute_state_eta(gains=gains)
        
        for num in photon_nums: 
            Y_n_mu = self.y_0 + 1 - (1 - eta_mu)**num
            Y_n_nu = self.y_0 + 1 - (1 - eta_nu)**num

            yields_mu.append(Y_n_mu)
            yields_nu.append(Y_n_nu)

        if self.debug:
            print(f"[DEBUG] Efficiency of signal state: {yields_mu}")
            print(f"[DEBUG] Efficiency of decoy state:{yields_nu}")

        return yields_mu, yields_nu