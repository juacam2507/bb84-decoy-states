import numpy as np
from tqdm import tqdm
from datetime import datetime
import json
import os

class DistanceAnalysis:
    def __init__(self, simulation_parameters, distance_analysis_params):
        self.sim_params = simulation_parameters
        self.dist_params = distance_analysis_params

