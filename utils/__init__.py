from .pgm import *
from .constants import *
from .pam import *
from .lcs import *
from.constants_cas9 import *
device = "cpu"

def push(file, key, val):
    data = torch.load(file)
    data[key] = val
    torch.save(data, file)

def pull(file, keys):
    data = torch.load(file)
    return {k: v for k, v in data.items() if v in keys}

def to_onehot(a, shape):
    if shape[0] is None:
        shape = len(a), shape[1]
    onehot = np.zeros(shape)
    onehot[np.arange(len(a)), a] = 1
    return onehot
