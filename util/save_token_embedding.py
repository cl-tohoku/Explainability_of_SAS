import json
import numpy as np
from collections import defaultdict


class TokenDict:
    def __init__(self):
        self.token_dict = defaultdict(lambda: defaultdict(float))

    # def append(self, norm, mode):
    #     norm = np.linalg.norm(norm, axis=1).tolist()
    #     self.token_dict[mode].append(norm)

    def save_norm(self, save_path):
        with open(save_path, 'w') as f:
            print(json.dumps(self.token_dict), file=f)
