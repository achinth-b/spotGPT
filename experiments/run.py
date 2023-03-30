import os, argparse
from pathlib import Path
import pickle, numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy import stats

os.chdir(Path(__file__).parent.resolve())

from code.utils.function_runners import (
    main, 
    handle
)


@handle("baseline_rvae")
def baseline_test_rvae_to_GPT():
    print("Hello World")



if __name__ == "__main__":
    main()