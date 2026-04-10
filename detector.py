import numpy as np


class Detector:
    def __init__(self, simulation_parameters: dict, rng: np.random.Generator, l: float):
        """
        Initialize a photon detector object for the BB84 simulation. The detector is modeled
        as a two detector model with the respective probability for the 4 possible events.

        Args:
            simulation_parameters (dict): Dictionary containing the simulation parameters
            rng (np.random.Generator): Random number generator
            l (float): Distance of the channel
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
        Computes the channel efficiency eta as:

            eta = t_ab*eta_bob

        Where the transmittance between a and b is given by:

            t_ab = 10^(-beta*l/10)

        Where beta is the loss coefficient of the channel and l is its lenght.
        T receptor efficiency is calculated as:

            eta_bob = t_bob*eta_d

        With t_bob being the receptors optical circuit transmittance and eta_d the detector
        efficiency.

        Returns:
            float: Channel Efficiency eta
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
        Computes the probabilities of the 4 possible events for a two detector model:

        p_none: Probability of no detection on either detector.
        p_corr: Probability of the signal being detected completely by the corresponding
        detector.
        p_errn: Probability of the signal being detected completely by the erroneous detector.
        p_both: Probability of the signal being detected by both detectors simultaneously.

        Args:
            eta (float): Channel efficiency
            photon_nums (np.ndarray): Array containing the photon numbers of each pulse

        Returns:
            np.ndarray: A (N, 4) matrix containing the probabilities of each event in its columns
            for each one of the pulses sent.
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

        # probabilities are [N,1] arrays so they are joined on a [N,4] matrix:

        detection_probs = np.column_stack([p_none, p_corr, p_errn, p_both])
        detection_probs = np.clip(detection_probs, 0.0, 1.0) #normalizes the range 

        if self.debug:
            print(f"[DEBUG] Probability matrix: {detection_probs}")

        return detection_probs

    def detect_pulse(self, eta: float, photon_nums: np.ndarray) -> np.ndarray:
        """
        Chooses a detection event based on the probability matrix for each pulse and returns
        an array with the corresponding decision:

            0: No detection (Signal lost)
            1: Correct detection
            2: Erroneous detection
            3: Detection on both

        Args:
            eta (float): Channel efficiency
            photon_nums (np.ndarray): Array containing the photon numbers of each pulse

        Returns:
            np.ndarray: Array containing 0, 1, 2 or 3 depending on the decision of the detection
            event
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
        Runs the detection event decision and computes bobs bits as follows:

            detected == 0 => Assigns bit to -1 indicating no detection
            detected == 1 => Assigns source_bit = receptor_bit
            detected == 2 => Assigns receptor_bit = (source_bit + 1) % 2 (Bit flip)
            detected == 3 => Tosses a coin and randomly assigns 0 or 1.

        Args:
            eta (np.ndarray): Channel efficiency
            photon_nums (np.ndarray): Array containing the number of photons in the pulse
            source_bits (_type_): Array containing Alice's bits

        Returns:
            receptor_bits (np.ndarray): Receptor's bit string
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
        Generates the random basis sequence for the receptor.

        Returns:
            basis_seq: Random array representing the base in which each bit will be encoded (0 = Rectilinear, 1 = Hadamard)
        """
        basis_seq = self.rng.integers(0, 2, size=self.N)

        if self.debug:
            print(f"[DEBUG] Receptor basis choice: {basis_seq}")

        return basis_seq
