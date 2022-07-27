from typing import List

import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import normalize
import umap
import pandas as pd


def dimention_reduction_umap(tensor_data: List, metric="euclidean": str):
    mapper = umap.UMAP(random_state=10, metric=metric)
    tensor_data = mapper.fit_transform(tensor_data)
    return tensor_data
