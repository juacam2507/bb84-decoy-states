import numpy as np
from datetime import datetime
import json
import os

class Data:
    def __init__(self, separator :str = ",", dtype=float, dir : str = ""):
        
        self.data_dir = "data"
        os.makedirs(self.data_dir, exist_ok=True)
        
        if dir:
            self.dirpath = os.path.join(self.data_dir, dir)
            os.makedirs(self.dirpath, exist_ok=True) 
        
        self.separator = separator
        self.dtype = dtype
        return None
    
    def write_data(self, *arrays, header : list = [], filename:str):
        
        
        timestamp = datetime.now()
        footer = f"_data{timestamp.strftime('%Y%m%d_%H%M%S')}"
        filename = filename + footer
        filepath = os.path.join(self.dirpath, filename)
        
        if not arrays:
            raise ValueError("You must provide at least an array.")
        
        arrays = [np.asarray(a) for a in arrays]
        
        lengths = [len(a) for a in arrays]
        
        if len(set(lengths)) != 1: 
            raise ValueError("All arrays must be the same length.")
        
        data = np.column_stack(arrays)
        
        header_str = ""
        if header is not None:
            if len(header) != len(arrays):
                raise ValueError("Header list must have the same length as array list")
            
            header_str = self.separator.join(header)
        
        np.savetxt(
        filepath,
        data,
        delimiter=self.separator,
        header=header_str,
        fmt="%.10g",
         )    
        
        