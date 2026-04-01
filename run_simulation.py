import numpy as np


simulation_parameters = {
    "N"                 : 50,                       # Number of generated pulses
    "mu"                : 1.0,                      # Signal intensity
    "decoy_intensities" : [0.5, 0.0],               # Decoy intensities
    "decoy_rate"        : 0.25,                     # Decoy probability
    "channel_properties" : {                                   
        "beta"          : 0.3,                      # Loss coefficient (dB/Km)
        "l"             : 0.5,                      # Channel lenght (Km)
        "channel_loss"  : 0.8                       # Attenuation coefficient
        },
    "detector_properties": {
        "receiver_transmit"    : 0.6, 
        "detector_efficiency"  : 0.47,
        "detector_error"       : 0.3,
        "dark_count_rate"      : 1e-6,
        "dark_count_error"     : 0.5
        
    },
    "debug": True
}

rng = np.random.default_rng()