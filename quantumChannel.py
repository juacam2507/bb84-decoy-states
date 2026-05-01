import numpy as np
from emitter import Emitter
from receiver import Receiver

class QuantumChannel:

    def __init__(self, simulation_parameters: dict, rng : np.random.Generator, l : float = 20.0):

        self.simulation_parameters = simulation_parameters
        self.debug = simulation_parameters["debug"]
        
        self.beta = simulation_parameters["channel_properties"]["beta"]
        self.t_bob = simulation_parameters["detector_properties"]["receiver_transmit"]
        self.eta_d = simulation_parameters["detector_properties"]["detector_efficiency"]
        
        self.alice = Emitter(simulation_parameters=simulation_parameters, rng=rng)
        self.bob = Receiver(simulation_parameters=simulation_parameters, rng=rng)
        
        self.eta = self.channel_efficiency(l)

    def channel_efficiency(self, l: float) -> float:
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

        t_ab = 10 ** (-1.0 * self.beta * l / 10.0)  # Channel transmittance
        eta_bob = self.t_bob * self.eta_d  # Receiver efficiency

        if self.debug:
            print(f"[DEBUG] Channel efficiency {t_ab*eta_bob}")
            print("----------------------------------------------------------------")
        
        return t_ab * eta_bob

    def send_pulses(self): 
    
    # Generate the photon pulse for Alice
        alice_bits, alice_basis, state_choice, photon_nums = self.alice.generate_pulses() 
        
        # Generate measurement basis and Bob's bits from the detection probabilities
        bob_basis = self.bob.generate_basis_seq()
        bob_bits = self.bob.generate_receptor_bits(self.eta, photon_nums, alice_bits)

        return alice_bits, alice_basis, state_choice, bob_basis, bob_bits