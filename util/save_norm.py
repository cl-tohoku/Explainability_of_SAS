import json
import numpy as np


class Norm:
    def __init__(self):
        self.norm_dict = {"train": [], "dev": [], "test": []}

    def append(self, norm, mode):
        norm = np.linalg.norm(norm, axis=1).tolist()
        self.norm_dict[mode].append(norm)

    def save_norm(self, save_path):
        with open(save_path, 'w') as f:
            print(json.dumps(self.norm_dict), file=f)
