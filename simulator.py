import numpy as np
import random

channel_properties = {
    'alpha': 0.03,
    'l': 5, 
    'detector_efficiency': 0.5,
    'bob_transmittance': 0.8, 
    'detector_error': 0.02,
    'y0': 1e-6,
    'e0': 0.5
}

def print_relative_frequencies(arr: np.ndarray) -> None:
    """
    Prints the relative frequency of each unique element in the array.
    
    Args:
        arr: numpy array (any dtype, but typically numeric)
    """
    # Aplanar por si viene multidimensional
    arr = arr.flatten()
    
    total = arr.size
    
    # Obtener valores únicos y sus conteos
    values, counts = np.unique(arr, return_counts=True)
    
    print("=== Relative Frequencies ===")
    print(f"Total elements: {total}\n")
    
    for val, count in zip(values, counts):
        freq = count / total
        print(f"Value {val:>5}: {count:>8} ({freq:.6f})")
    
    print("============================")

def channel_transmittance(alpha : float, l: float) -> float:
    return 10 ** (-alpha * l / 10)

def eta(channel_properties: dict) -> float:
    alpha = channel_properties['alpha']
    l = channel_properties['l']
    bob_transmittance = channel_properties['bob_transmittance']
    detector_efficiency = channel_properties['detector_efficiency']
    return channel_transmittance(alpha, l)*bob_transmittance*detector_efficiency
   
rng = np.random.default_rng()

N = 100
mu = 1
nu = 0.5
decoy_rate = 0.25
vac_weak_rate = 0.5 # Vaccum pulses / # Weak pulses

#Compute channel properties
eta = eta(channel_properties)
y_0 = channel_properties['y0']
e_0 = channel_properties['e0']
e_d = channel_properties['detector_error']
# Alice's bits
alice_bits = rng.integers(0,2, size = N)

#Alices's Basis choice
alice_basis = rng.integers(0, 2, size = N)


#Alices pulse state choice
signal_prob = 1 - decoy_rate
weak_prob = decoy_rate*(1 - vac_weak_rate)
vac_prob = decoy_rate*vac_weak_rate
pulse_state_probs = [signal_prob, weak_prob, vac_prob]

alice_state = rng.choice(
    [2, 1, 0], 
    size = N, 
    p = pulse_state_probs
    )

#Now we need to simulate how Alice sends each pulse with a different photon number following a poissonian distribution

alice_pulses = np.zeros(N, dtype=int)
signal_mask = alice_state == 2
decoy_mask = alice_state == 1
# print(signal_mask)
# print(decoy_mask)

alice_pulses[signal_mask] = rng.poisson(mu, size = signal_mask.sum())
alice_pulses[decoy_mask] = rng.poisson(nu, size = decoy_mask.sum())

#The complete information can be condensed into a single array representing Alice

source = [alice_bits, alice_basis, alice_state, alice_pulses]

#Now we need to simulate detection. First, we define the basis choice for Bob and initialize the bit sequence

bob_bits = np.zeros(N, dtype=int)
bob_basis = rng.integers(0, 2, size = N)

#We need to eliminate the indexes for which the basis are different. 

bit_sift_mask = bob_basis != alice_basis #if True, basis are different. If False basis coincide.

bob_bits[bit_sift_mask] = -2 #Bits of the original sequence that have different basis choices are tagged with index -2. 

#We need to separate detection events into scenarios. If the basis coincide, the detection events can be separated as follows:
#   Scenario 1: Alice sent an n-photon state
#       - Sub-scenario 1: Bob detects nothing.
#       - Sub-scenario 2: Bob detects one of the n photons. 
#           - Sub-sub-scenario 1: Bob detects the photon in the incorrect detector so the bit is incorrect.
#           - Sub-sub-scenario 2: Bob detects the photon in the correct detector. 
#   Scenario 2: Alice sent nothing  
#       - Subscenario 1: Bob detects nothing
#       - Subscenario 2: Bob detects a dark count

#First we need to sift the pulses that have 0 photons 

zero_photon_mask = (alice_pulses == 0) & (bob_bits != -2)
non_zero_photon_mask = (alice_pulses != 0) & (bob_bits != -2)

bob_bits[zero_photon_mask] = -1
 


debug = True
if debug:
    print(eta)
    print(f"[DEBUG] Alice's bits: {alice_bits}")
    print_relative_frequencies(alice_bits)
    print(f"[DEBUG] Alice's basis: {alice_basis}")
    print_relative_frequencies(alice_basis)
    print(f"[DEBUG] Alice's state choice (decoy or signal): {alice_state}")
    print_relative_frequencies(alice_state)
    print(f"[DEBUG] Photon number associated to each pulse: {alice_pulses}")
    print_relative_frequencies(alice_pulses)
    print(f"[DEBUG] Bob's basis: {bob_basis}")
    print_relative_frequencies(bob_basis)
    #print(f"[DEBUG] Pulses with 0 photons and coincident basis:{zero_photon_mask}")
    #print(f"[DEBUG] Non equal basis index: {bit_sift_mask}")
    print(f"[DEBUG] Bob's Bits: {bob_bits}")
    print_relative_frequencies(bob_bits)
    #print(f"[DEBUG] Detected's Bits: {detected}")

