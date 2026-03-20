import numpy as np
import random

channel_properties = {
    'alpha': 0.5,
    'l': 10, 
    'detector_efficiency': 0.1,
    'bob_transmittance': 0.3, 
    'detector_error': 0.02,
    'dark_count_rate': 1e-6
}

def channel_transmittance(alpha, l) -> float:
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

eta = eta(channel_properties)
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

#Probability of detection of one photon: eta
#Probability of not-detection of one photon: 1 - eta
photon_detection_prob = 1 - eta

bob_bits[zero_photon_mask] = rng.choice

debug = True
if debug:
    print(f"[DEBUG] Alice's bits: {alice_bits}")
    print(f"[DEBUG] Alice's basis: {alice_basis}")
    print(f"[DEBUG] Alice's state choice (decoy or signal): {alice_state}")
    print(f"[DEBUG] Photon number associated to each pulse: {alice_pulses}")
    print(f"[DEBUG] Bob's basis: {bob_basis}")
    print(f"[DEBUG] Pulses with 0 photons and coincident basis:{zero_photon_mask}")
    print(f"[DEBUG] Non equal basis index: {bit_sift_mask}")
    print(f"[DEBUG] Bob's Bits: {bob_bits}")

