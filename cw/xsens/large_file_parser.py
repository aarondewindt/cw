from cw.xsens.parser import XDI, XSensParser
from cw.flex_file import flex_dump
import tqdm
from functools import partial
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


class XSensLargeFileParser:
    def __init__(self,
                 file_like,
                 output_path,
                 output_format=None,
                 calibration_data: dict=None,
                 verbose=False,
                 chunk_size=8692195):
        self.file_like = file_like
        self.calibration_data = calibration_data
        self.verbose = verbose
        self.chunk_size = chunk_size
        self.xp = XSensParser(None, calibration_data=self.calibration_data)
        self.output_format = output_format or ".msgp.gz"

        self.output_path = Path(output_path)

        if not self.output_path.is_dir():
            self.output_path.mkdir()

    def start(self):
        var_list = []

        total_size = None
        try:
            self.file_like.seek(0, 2)
            total_size = self.file_like.tell()
            self.file_like.seek(0)
        except:
            pass

        with tqdm.tqdm(total=total_size, desc="parsing", disable=(not self.verbose), unit="bytes") as pbar:
            for i, chunk in enumerate(iter(partial(self.file_like.read, self.chunk_size), b'')):
                pbar.update(len(chunk))
                self.xp.parse(chunk)

                chunk_tables = self.xp.get_tables()
                self.xp.mtdata2_deque.clear()
                flex_dump(chunk_tables, self.output_path / f"chunk_{i}{self.output_format}")
                var_list.append(np.var(chunk_tables['acceleration']['data']))

        (self.output_path / "variance.csv").write_text("\n".join([f"{i}\t{x}" for i, x in enumerate(var_list)]))

        plt.plot(var_list)
