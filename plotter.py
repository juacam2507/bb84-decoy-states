import numpy as np
import matplotlib.pyplot as plt
from data import Data
import os

class Plotter:
    def __init__(self) -> None:
        
        self.fig_dir = "figures"
        os.makedirs(self.fig_dir, exist_ok=True)
    
    def scatter_plot(self, filepath, )