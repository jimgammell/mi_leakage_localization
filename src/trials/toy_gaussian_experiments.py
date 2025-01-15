from typing import *
import os

class Trial:
    def __init__(self,
        logging_dir: Union[str, os.PathLike] = None
    ):
        self.logging_dir = logging_dir
    
    def run_xor_trial(self):
        pass