import os
import random
import numpy as np

# Reproducibility Constants
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Data Directories
RAW_PATH = os.path.join("data", "raw", "recipe_list.csv")
PROCESSED_DIR = os.path.join("data", "processed")