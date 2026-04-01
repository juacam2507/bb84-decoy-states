import numpy as np

class Source:
    def __init__(self, simulation_parameters: dict, rng: np.random.Generator):
        
        self.N = simulation_parameters["N"]
        self.mu = simulation_parameters["mu"]
        self.decoy_intensities = simulation_parameters["decoy_intensities"]
        self.decoy_rate = simulation_parameters["decoy_rate"]
        self.debug = simulation_parameters["debug"]
        self.rng = rng

    def generate_bit_seq(self) -> np.ndarray:
        
        """
        Generate a random bit sequence for the source (uses self.N and self.rng)

        Returns:
            source_bits: Random array of representing the bits of the key 
        """
        
        source_bits = self.rng.integers(0, 2, self.N)
        
        if self.debug:
            print(f'[DEBUG] Source bits: {source_bits}')
            
        return source_bits

    def generate_basis_seq(self) -> np.ndarray: 
        """
        Generates the basis sequence in which the bits are to be encoded. 

        Returns:
            source_basis: Random array representing the base in which each bit will be encoded (0 = Rectilinear, 1 = Hadamard)
        """
        
        source_basis = self.rng.integers(0, 2, self.N)
        
        if self.debug:
            print(f'[DEBUG] Source basis: {source_basis}')
        
        return source_basis

    def generate_state_seq(self) -> np.ndarray:
        """
        Assigns a number to the signal state (0) and for the decoy states (1, 2, ...) and generates the sequence of states.  

        Returns:
            decoy_sequence: Sequence of numbers that indicate the state of the pulse.  
        
        """
        decoy_num = int(len(self.decoy_intensities))
        
        state_index = [0] + list(np.arange(1, decoy_num + 1))
        state_probs = [1 - self.decoy_rate] + [self.decoy_rate/decoy_num]*decoy_num  #Concatenation of arrays

        state_sequence = self.rng.choice(state_index, size = self.N, p = state_probs)
        
        if self.debug:
            print(f'[DEBUG] State choice: {state_sequence}')

        return state_sequence

    def generate_photon_number_seq(self, state_choice: np.ndarray) -> np.ndarray:
        """
        Computes the number of photons in each pulse from a poissonian distribution with a mean intensity based on the choice of state (Decoy or signal) from decoy_choice. 

        Args:
            state_choice (np.ndarray): Array containing the index associated to pulse and decoy intensities
   
        Returns:
            photon_nums: Sequence with the number of photons sent for each pulse. 
        """ 
        intensities = [self.mu] + self.decoy_intensities #Intensity list
        
        photon_nums = np.zeros(self.N, dtype = int)
        
        for i in range(len(intensities)):
            intensity_mask = state_choice == i
            photon_nums[intensity_mask] = self.rng.poisson(intensities[i], size = np.sum(intensity_mask))
        
        
        if self.debug:
            print(f'[DEBUG] Photon numbers: {photon_nums}')
        
        return photon_nums

    def generate_pulses(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
        """
        Returns a tuple with the bit sequence, the basis choice, the state choice and the photon number for each pulse. 

        Args:
            simulation_parameters (dict): Dictionary containing the simulation parameters
            rng (np.random.Generator): Random number generator

        Returns:
            tuple[source_bit_seq, source_basis_seq, source_state_seq, photon_nums]: Tuple with the information of the generated pulses. 
        """
        
        source_bit_seq = self.generate_bit_seq()
        source_basis_seq = self.generate_basis_seq()
        source_state_seq = self.generate_state_seq()
        photon_nums = self.generate_photon_number_seq(source_state_seq)
        
        return source_bit_seq, source_basis_seq, source_state_seq, photon_nums


