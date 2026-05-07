import numpy as np
from datetime import datetime
import json
import os
import pandas as pd


class DataHandler:
    def __init__(
        self,
        dir: str = "",
    ):
        self.dir = dir
        self.data_dir = "data"
        os.makedirs(self.data_dir, exist_ok=True)

        if dir:
            self.dirpath = os.path.join(self.data_dir, self.dir)
            os.makedirs(self.dirpath, exist_ok=True)
        else:
            self.dirpath = self.data_dir

    def write_data(
        self,
        *arrays,
        header: list = [],
        filename: str,
        simulation_parameters: dict,
        separator: str = ",",
        dtype=float,
    ):

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
            if len(header) != len(data[0]):
                raise ValueError("Header list must have the same length as array list")

            header_str = separator.join(header)

        meta = simulation_parameters.copy()
        meta["time"] = timestamp.strftime("%Y/%m/%d - %H:%M:%S")

        with open(filepath, "w", encoding="utf-8") as f:
            meta_json = json.dumps(meta, indent=2, ensure_ascii=False)
            for line in meta_json.splitlines():
                f.write(f"#{line}\n")
            f.write("#---\n")
            if header_str:
                f.write(f"{header_str}\n")

            np.savetxt(
                f,
                data,
                delimiter=separator,
                header="",
                fmt="%.10g",
            )

    def read_data(self, filepath):

        meta_lines = []

        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("#---"):
                    break
                if line.startswith("#"):
                    meta_lines.append(line[1:].strip())

        meta = json.loads("\n".join(meta_lines))

        df = pd.read_csv(filepath, comment="#")

        return df, meta
