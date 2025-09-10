import math
import numpy as np

method_names = [
    "FNOPE",
    "FNOPE-fix",
    "NPE (spectral)",
    "FMPE (spectral)",
    "FMPE (raw)",
    "simformer",
    "NPE (raw)",
]  

colors = [
        "#9b2226",
        "#CA6702",
        "#023e8a",
        "#0077b6",
        "#00b4d8",
        "#791E94",
        "#90e0ef",
    ]

def float_to_power_of_ten(val: float):
    exp = math.log10(val)
    exp = int(exp)
    return rf"$10^{exp}$"


def get_size_tuple(fig):
    """returns the size of a figure

    Args:
        fig (_type_): _description_

    Returns:
        _type_: _description_
    """
    size = fig.get_size()
    size = [float(size[0][:-2]), float(size[1][:-2])]
    return size


def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i / inch for i in tupl[0])
    else:
        return tuple(i / inch for i in tupl)
    

    # Define a custom formatter for scientific notation
def scientific_formatter(x, pos):
    if x == 0:
        return "0"
    exponent = int(np.log10(x))
    base = x / 10**exponent
    if base != 1:
        return r"${} \times 10^{{{}}}$".format(int(base), exponent)
    else:
        return r"$10^{{{}}}$".format(exponent)
