import numpy as np


class Detector:
    def __init__(self, simulation_parameters: dict, rng: np.random.Generator, l: float):
        """
        Initialize the photon detector for BB84 decoy-state QKD simulation.

        Models a two-detector system with channel loss, detector efficiency,
        dark counts, and intrinsic error rates. Computes overall channel efficiency
        based on fiber attenuation and receiver optics.

        Parameters
        ----------
        simulation_parameters : dict
            Configuration dictionary with required keys:
            - "N": int, number of pulses.
            - "debug": bool, enable verbose output.
            - "channel_properties": dict with "beta": float (dB/km loss coefficient).
            - "detector_properties": dict with:
                - "receiver_transmit": float, Bob's optical circuit transmittance.
                - "detector_efficiency": float, single-photon detector efficiency.
                - "detector_error": float, intrinsic detector error rate.
                - "dark_count_rate": float, dark count probability per pulse.
                - "dark_count_error": float, dark count bit error probability.
        rng : np.random.Generator
            Random number generator for reproducible detection events.
        l : float
            Channel length in kilometers.
        """

        self.N = simulation_parameters["N"]
        self.rng = rng
        self.debug = simulation_parameters["debug"]

        self.l = l
        self.beta = simulation_parameters["channel_properties"]["beta"]

        self.t_bob = simulation_parameters["detector_properties"]["receiver_transmit"]
        self.eta_d = simulation_parameters["detector_properties"]["detector_efficiency"]
        self.e_d = simulation_parameters["detector_properties"]["detector_error"]
        self.y_0 = simulation_parameters["detector_properties"]["dark_count_rate"]
        self.e_0 = simulation_parameters["detector_properties"]["dark_count_error"]

    def channel_efficiency(self) -> float:
        """
        Compute overall channel efficiency including fiber loss and receiver optics.

        Total efficiency is given by :math:`\\eta = t_{ab} \\cdot \\eta_{bob}` where:
        - Channel transmittance: :math:`t_{ab} = 10^{-\\beta l / 10}`
        - Receiver efficiency: :math:`\\eta_{bob} = t_{bob} \\cdot \\eta_d`
    
        Returns
        -------
        float
            Overall channel efficiency :math:`\\eta \\in [0, 1]`.
        """

        t_ab = 10 ** (-1.0 * self.beta * self.l / 10.0)  # Channel transmittance
        eta_bob = self.t_bob * self.eta_d  # Receiver efficiency

        if self.debug:
            print(f"[DEBUG] Channel efficiency {t_ab*eta_bob}")

        return t_ab * eta_bob

    def compute_detection_probabilities(
        self, eta: float, photon_nums: np.ndarray
    ) -> np.ndarray:
        """
        Compute multinomial detection probabilities for two-detector model.

        For each pulse, computes probabilities of four mutually exclusive events:
        - Column 0: No detection on either detector
        - Column 1: Correct detector click only
        - Column 2: Wrong detector click only  
        - Column 3: Both detectors click

        Parameters
        ----------
        eta : float
            Channel efficiency for this pulse batch.
        photon_nums : npt.NDArray[np.int_]
            Array of shape (N,) containing photon numbers per pulse.

        Returns
        -------
        npt.NDArray[np.float64]
            Array of shape (N, 4) where each row sums to 1.0, containing
            event probabilities clipped to [0, 1] range.
        """
        eta_n = 1 - (1 - eta) ** photon_nums

        p_none = (1 - (1 - self.e_d) * eta_n - (1 - self.e_0) * self.y_0) * (
            1 - self.e_d * eta_n - self.e_0 * self.y_0
        )
        p_corr = ((1 - self.e_d) * eta_n + (1 - self.e_0) * self.y_0) * (
            1 - self.e_d * eta_n - self.e_0 * self.y_0
        )
        p_errn = (1 - (1 - self.e_d) * eta_n - (1 - self.e_0) * self.y_0) * (
            self.e_d * eta_n + self.e_0 * self.y_0
        )
        p_both = ((1 - self.e_d) * eta_n + (1 - self.e_0) * self.y_0) * (
            self.e_d * eta_n + self.e_0 * self.y_0
        )

        # Stack into (N, 4) matrix

        detection_probs = np.column_stack([p_none, p_corr, p_errn, p_both])

         # Normalize each row to sum exactly to 1.0
        row_sums = detection_probs.sum(axis=1, keepdims=True)
        detection_probs /= np.maximum(row_sums, 1e-15)  # Avoid div-by-zero

        if self.debug:
            print(f"[DEBUG] Row sums before norm: {row_sums[:5]}")
            print(f"[DEBUG] Row sums after norm: {(detection_probs.sum(axis=1)[:5])}")
            print(f"[DEBUG] Probability matrix:\n{detection_probs[:3]}")

        return detection_probs

    def detect_pulse(self, eta: float, photon_nums: np.ndarray) -> np.ndarray:
        """
        Sample detection events from multinomial distribution.

        Uses inverse transform sampling on cumulative probability matrix to
        select one event per pulse: 0=no-click, 1=correct, 2=wrong, 3=both.

        Parameters
        ----------
        eta : float
            Channel efficiency.
        photon_nums : npt.NDArray[np.int_]
            Photon numbers per pulse, shape (N,).

        Returns
        -------
        npt.NDArray[np.int_]
            Detection decisions, shape (N,) with values in {0,1,2,3}.
        """

        detection_probs = self.compute_detection_probabilities(eta, photon_nums)

        cumulative_sum = detection_probs.cumsum(
            axis=1
        )  # Computes the Nx4 matrix with the cumulative sum of row elements.

        # We generate an auxiliary array of size Nx1 with random numbers between 0 and 1

        aux = self.rng.random(size=(len(detection_probs), 1))
        detection_event = (aux < cumulative_sum).argmax(axis=1)

        if self.debug:
            print(f"[DEBUG] Cumulative sum matrix: {cumulative_sum}")
            print(f"[DEBUG] Random detection choice vector: {aux}")
            print(f"[DEBUG] Choosen detection event: {detection_event}")

        return detection_event

    def generate_receptor_bits(
        self, eta: float, photon_nums: np.ndarray, source_bits
    ) -> np.ndarray:
        """
        Map detection events to Bob's bit values.

        Detection → bit mapping:
        * 0 (no-click): -1 (erasure)
        * 1 (correct): copy source bit
        * 2 (wrong): flip source bit
        * 3 (both): random bit (0 or 1)

        Parameters
        ----------
        eta : float
            Channel efficiency.
        photon_nums : npt.NDArray[np.int_]
            Photon numbers per pulse, shape (N,).
        source_bits : npt.NDArray[np.int_]
            Alice's transmitted bits, shape (N,) with values in {0,1}.

        Returns
        -------
        npt.NDArray[np.int_]
            Bob's received bits, shape (N,) with values in {-1,0,1}.
        """
        detection_event = self.detect_pulse(eta, photon_nums)

        receptor_bits = np.zeros(self.N, dtype=int)

        no_detection_mask = detection_event == 0
        corr_detection_mask = detection_event == 1
        errn_detection_mask = detection_event == 2
        both_detection_mask = detection_event == 3

        receptor_bits[no_detection_mask] = -1
        receptor_bits[corr_detection_mask] = source_bits[corr_detection_mask]
        receptor_bits[errn_detection_mask] = (source_bits[errn_detection_mask] + 1) % 2
        receptor_bits[both_detection_mask] = self.rng.integers(
            0, 2, size=np.sum(both_detection_mask)
        )

        if self.debug:
            print(f"[DEBUG] Receptor bit array after detection: {receptor_bits}")

        return receptor_bits

    def generate_basis_seq(self) -> np.ndarray:
        """
        Generate random measurement basis choices for Bob.

        Each basis chosen independently with equal probability:
        0 = rectilinear (Z-basis), 1 = diagonal (X-basis).

        Returns
        -------
        npt.NDArray[np.int_]
            Basis choices, shape (N,) with values in {0,1}.
        """
        basis_seq = self.rng.integers(0, 2, size=self.N)

        if self.debug:
            print(f"[DEBUG] Receptor basis choice: {basis_seq}")

        return basis_seq
