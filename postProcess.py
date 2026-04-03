import numpy as np
from source import Source
from detector import Detector


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

    def compute_state_gain(
        self, bob_bits: np.ndarray, state_choice: np.ndarray, state: int = 0
    ) -> float:
        """
        Computes the gain Q for a specific state (Vacuum, weak or decoy). The gain is defined as the probability of
        a detection event triggering at the receptor when alice sent with mean photon number mu. In this case, it is
        computed as the relative frequency of the successful detection events (bob_bits != -1) for each
        state choice (state_choice = 0, 1, 2, ...)

        Args:
            detection_event (np.ndarray): Array with the detection event decision
            state_choice (np.ndarray): Alice's choice of states.
            state (int, optional): State for which the gain is to be computed. Defaults to 0.

        Returns:
            float: Gain Q of a specific state.
        """

        state_mask = state_choice == state
        state_detection_mask = (bob_bits != -1) & (state_mask)

        states_sent = np.sum(state_mask)
        states_detected = np.sum(state_detection_mask)

        if self.debug:
            print(f"[DEBUG] State {state} Detected: {states_detected}")
            print(f"[DEBUG] State {state} Sent: {states_sent}")
            print(f"[DEBUG] Gain of state {state} = {states_detected/states_sent}")

        return states_detected / states_sent

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


simulation_parameters = {
    "N": 1000000,  # Number of generated pulses
    "mu": 1.0,  # Signal intensity
    "decoy_intensities": [0.5, 0.0],  # Decoy intensities
    "decoy_rate": 0.2,  # Decoy probability
    "channel_properties": {
        "beta": 0.2,  # Loss coefficient (dB/Km)
        "l": 20,  # Channel lenght (Km)
        "channel_loss": 0.8,  # Attenuation coefficient
    },
    "detector_properties": {
        "receiver_transmit": 0.045,  # Receiver transmittance
        "detector_efficiency": 0.47,  # Detector Efficiency
        "detector_error": 0.033,  # Probability of a pulse measured in the correct basis to trigger the wrong detector
        "dark_count_rate": 1e-6,  # Probability of dark counts
        "dark_count_error": 0.5,  # Probability of dark counts triggering the wrong detector
    },
    "debug": True,
}
rng = np.random.default_rng()
alice = Source(simulation_parameters, rng)
alice_bits, alice_basis, state_choice, photon_nums = alice.generate_pulses()

bob = Detector(simulation_parameters, rng)
eta = bob.channel_efficiency()
bob_bits = bob.generate_receptor_bits(eta, photon_nums, alice_bits)
bob_basis = bob.generate_basis_seq()

post_process = PostProcess(simulation_parameters, rng)
signal_gain = post_process.compute_state_gain(bob_bits, state_choice, 0)
matching_basis_mask = post_process.basis_reconciliation(alice_basis, bob_basis)

sifted_alice_bits = alice_bits[matching_basis_mask]
sifted_bob_bits = bob_bits[matching_basis_mask]
sifted_state_choice = state_choice[matching_basis_mask]

signal_qber = post_process.compute_state_qber(
    sifted_source_bits=sifted_alice_bits,
    sifted_receptor_bits=sifted_bob_bits,
    sifted_state_choice=sifted_state_choice,
    state=0,
)
