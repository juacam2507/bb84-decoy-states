import numpy as np
import random

channel_properties = {
    'alpha': 0.5,
    'l': 10, 
    'detector_efficiency': 0.1,
    'bob_transmittance': 1.0, 
    'detector_error': 0.02,
    'dark_count_rate': 1e-6
}

def channel_transmittance(alpha, l) -> float:
    return 10 ** (-alpha * l / 10)

def eta(alpha, l, detector_efficiency, bob_transmittance) -> float:
    return channel_transmittance(alpha, l)*bob_transmittance*detector_efficiency


def choose_state(prob_signal = 0.7, prob_decoy = 0.2, prob_vac = 0.1) -> str:
    if(prob_decoy + prob_signal + prob_vac != 1):
        return "Error: Probabilities should add up to 1."
    r = random.random()
    #print(f"Random number for state selection: {r}")
    if r < prob_signal:
        return '2'
    elif r > prob_signal + prob_decoy:
        return '0'
    else:
        return '1'

def sample_photon_number(mu = 1.0) -> int:
    return np.random.poisson(mu)
    
#Yield of an n-photon pulse:

def Y_n(n = 3, eta = 0.6, y_0 = 1e-6) -> float: 
    return 1 - (1-eta)**n + y_0

#print(Y_n())

def detect(n = 3, eta = 0.6, y_0 = 1e-6) -> bool:
    return random.random() < Y_n(n, eta, y_0)

    
N = 100
mu = 1
nu = 0.5
decoy_rate = 0.25
vac_weak_rate = 0.5 # Vaccum pulses / # Weak pulses

rng = np.random.default_rng()

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








    



