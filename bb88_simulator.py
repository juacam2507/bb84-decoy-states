import numpy as np 
from source import Source
from detector import Detector
from postProcess import PostProcess
from tqdm import tqdm

class Simulator:
    def __init__(self, simulation_parameters: dict, rng: np.random.Generator):
        
        self.simulation_parameters = simulation_parameters
        self.rng = rng
        self.state_num = len(simulation_parameters["decoy_intensities"]) + 1
        self.debug = simulation_parameters["debug"]

    
    def run(self, l: float, iter: int) -> float: 
        
        #Declare objects
        alice = Source(self.simulation_parameters, self.rng)
        bob = Detector(self.simulation_parameters, self.rng, l)
        post_process = PostProcess(self.simulation_parameters, self.rng)
        
        Q_cum = np.zeros(self.state_num, dtype=float)
        E_cum = np.zeros(self.state_num, dtype=float)
        Q_av =  np.zeros(self.state_num, dtype=float)
        E_av =  np.zeros(self.state_num, dtype=float)
            
        for iter in tqdm(range(iter), desc = "Iterations"):
            
            #Generate the photon pulse for Alice
            alice_bits, alice_basis, state_choice, photon_nums = alice.generate_pulses()

            #Compute the channel efficiency
            eta = bob.channel_efficiency()
            
            #Generate measurement basis and Bob's bits from the detection probabilities
            bob_basis = bob.generate_basis_seq()
            bob_bits = bob.generate_receptor_bits(eta, photon_nums, alice_bits)

            #Compute gains for each state (Signal, weak, vacuum)
            gains = post_process.compute_gains(bob_bits, state_choice)
            
            #Perform basis reconciliation 
            matching_basis_mask = post_process.basis_reconciliation(alice_basis, bob_basis)
            sifted_alice_bits = alice_bits[matching_basis_mask]
            sifted_bob_bits = bob_bits[matching_basis_mask]
            sifted_state_choice = state_choice[matching_basis_mask]

            #Compute QBER for each state
            qbers = post_process.compute_qbers(
                sifted_source_bits=sifted_alice_bits,
                sifted_receptor_bits=sifted_bob_bits,
                sifted_state_choice=sifted_state_choice,
            )
            Q_cum += gains
            E_cum += qbers 
            
        Q_av = Q_cum/iter
        E_av = E_cum/iter
        
        if self.debug: 
            print(f"[DEBUG] Average gains after {iter} iterations: {Q_av}")
            print(f"[DEBUG] Average QBER after {iter} iterations: {E_av}")
            
        #Compute the thresholds to calculate the secure Key rate. 
        y_0_l = post_process.background_yield_bound(gains=Q_av)
        y_1_l = post_process.single_photon_yield_bound(gains=Q_av, y_0_l=y_0_l)
        e_1_u = post_process.single_photon_error_bound(gains=Q_av, qbers=E_av, y_1_l=y_1_l)
        R = post_process.secure_key_rate(gains=Q_av, qbers=E_av, y_1_l=y_1_l, e_1_u=e_1_u)

        return R