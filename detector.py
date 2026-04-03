import numpy as np
from source import Source


class Detector:
    def __init__(self, simulation_parameters: dict, rng):
        
        self.N = simulation_parameters["N"]
        self.rng = rng
        self.debug = simulation_parameters["debug"]
        
        self.l = simulation_parameters["channel_properties"]["l"]
        self.beta = simulation_parameters["channel_properties"]["beta"]
        
        self.t_bob = simulation_parameters["detector_properties"]["receiver_transmit"]
        self.eta_d = simulation_parameters["detector_properties"]["detector_efficiency"]
        self.e_d   = simulation_parameters["detector_properties"]["detector_error"]
        self.y_0   = simulation_parameters["detector_properties"]["dark_count_rate"]
        self.e_0   = simulation_parameters["detector_properties"]["dark_count_error"]
    
    def channel_efficiency(self) -> float:
        
        t_ab = 10**(-1.0*self.beta*self.l/10.0)         #Channel transmittance
        eta_bob = self.t_bob*self.eta_d                 #Receiver efficiency
        
        if self.debug:
            print(f"[DEBUG] Channel efficiency {t_ab*eta_bob}")
        
        return t_ab*eta_bob
    
    def compute_detection_probabilities(self, eta: float, photon_nums: np.ndarray) -> np.ndarray:
        
        eta_n = 1 - (1 - eta)**photon_nums 
        
        p_none = (1 - (1 - self.e_d)*eta_n - (1 - self.e_0)*self.y_0)*(1 - self.e_d*eta_n - self.e_0*self.y_0) 
        p_corr = ((1 - self.e_d)*eta_n + (1 - self.e_0)*self.y_0)*(1 - self.e_d*eta_n - self.e_0*self.y_0) 
        p_errn = (1 - (1 - self.e_d)*eta_n - (1 - self.e_0)*self.y_0)*(self.e_d*eta_n + self.e_0*self.y_0)
        p_both = ((1 - self.e_d)*eta_n + (1 - self.e_0)*self.y_0)*(self.e_d*eta_n + self.e_0*self.y_0)
        
        #probabilities are [N,1] arrays so they are joined on a [N,4] matrix:
        
        detection_probs = np.column_stack([p_none, p_corr, p_errn, p_both])
        
        if self.debug:
            print(f"[DEBUG] Probability matrix: {detection_probs}")
        
        return detection_probs
    
    def detect_pulse(self, eta: float, photon_nums: np.ndarray) -> np.ndarray:
        
        detection_probs = self.compute_detection_probabilities(eta, photon_nums)
        
        cumulative_sum = detection_probs.cumsum(axis=1) # Computes the Nx4 matrix with the cumulative sum of row elements.
        
        #We generate an auxiliary array of size Nx1 with random numbers between 0 and 1
         
        aux = rng.random(size = (len(detection_probs), 1))
        detection_event = (aux < cumulative_sum).argmax(axis = 1)
        
        if self.debug: 
            print(f"[DEBUG] Cumulative sum matrix: {cumulative_sum}")
            print(f"[DEBUG] Randomly generated vector: {aux}")
            print(f"[DEBUG] Choosen detection event: {detection_event}")
        
        return detection_event
    
    def generate_basis_seq(self) -> np.ndarray:
        
        basis_seq = self.rng.integers(0, 2, size = self.N) 
        
        return basis_seq

simulation_parameters = {
    "N"                 : 10,                       # Number of generated pulses
    "mu"                : 1.0,                      # Signal intensity
    "decoy_intensities" : [0.5, 0.0],               # Decoy intensities
    "decoy_rate"        : 0.25,                     # Decoy probability
    "channel_properties" : {                                   
        "beta"                  : 0.3,              # Loss coefficient (dB/Km)
        "l"                     : 0.5,              # Channel lenght (Km)
        "channel_loss"          : 0.8               # Attenuation coefficient
        },
    "detector_properties": {
        "receiver_transmit"     : 0.6,              # Receiver transmittance
        "detector_efficiency"   : 0.47,             # Detector Efficiency
        "detector_error"        : 0.3,              # Probability of a pulse measured in the correct basis to trigger the wrong detector
        "dark_count_rate"       : 1e-6,             # Probability of dark counts
        "dark_count_error"      : 0.5               # Probability of dark counts triggering the wrong detector
        
    },
    "debug": False
}
rng = np.random.default_rng()
alice = Source(simulation_parameters, rng)
_, _, _, photon_nums = alice.generate_pulses()

bob = Detector(simulation_parameters, rng)

eta = bob.channel_efficiency()

p = bob.detect_pulse(eta, photon_nums)
    
  
        
        