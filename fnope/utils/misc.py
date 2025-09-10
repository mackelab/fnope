import numpy as np
import torch
import random
from time import time_ns
from pathlib import Path
import pickle
import io

def set_seed(seed:int):
    """This methods just sets the seed."""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def time_ns_string():
    return str(time_ns())


#Some functions to get useful paths in the project
def get_project_root() -> Path:
    return Path(__file__).absolute().parent.parent.parent

def get_output_dir() -> Path:
    return get_project_root() / "out"

def get_data_dir() -> Path:
    return get_project_root() / "data"


def read_pickle(file_path):
    with open(file_path, "rb") as f:
        try:
            return pickle.load(f)
        except Exception:
            # 2) Fallback to raw pickle + CPU mapping
            with open(file_path, 'rb') as f:
                # rewind to start!
                f.seek(0)
                return CPU_Unpickler(f).load()
    
class CPU_Unpickler(pickle.Unpickler):
    #Load inference objects saved on GPU to CPU
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location=torch.device('cpu'))
        else:
            return super().find_class(module, name)