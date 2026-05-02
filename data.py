import numpy as np
from datetime import datetime
import json
import os

class Data:
    def __init__(self, simulation_parameters: dict, separator : str = ",", dtype=float, dir : str = ""):
        
        self.simulation_parameters = simulation_parameters
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
        footer = f"_data_{timestamp.strftime('%Y%m%d_%H%M%S')}.csv"
        filename = filename + footer
        if self.dirpath:
            filepath = os.path.join(self.dirpath, filename)
        else:
            filepath = filename
        
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
        
        meta = self.simulation_parameters.copy()
        meta["time"] = timestamp.strftime("%Y/%m/%d - %H:%M:%S")
        
        with open(filepath, "w", encoding="utf-8") as f:
            meta_json = json.dumps(meta, indent=2, ensure_ascii=False)
            for line in meta_json.splitlines():
                f.write(f"#{line}\n")
            f.write("#---\n")
        
        np.savetxt(
        filepath,
        data,
        delimiter=self.separator,
        header=header_str,
        fmt="%.10g",
         )    
        
    def read(self, filepath):
            
        with open(filepath,"r", encoding="utf-8") as f:
                first_line = f.readline().strip()
                
        header = first_line.split(sep=self.separator)
        
        data = np.loadtxt(
            filepath,
            delimiter=self.separator,
            skiprows=1,
            dtype = self.dtype
        )
        
        if data.ndim == 1:
            data.reshape(-1,1)
        
        arrays = [data[:, i] for i in range(data.shape[1])]
        
        return header, arrays
        