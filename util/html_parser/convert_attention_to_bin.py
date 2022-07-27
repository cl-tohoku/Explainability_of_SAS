import argparse
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument()
    args = parser.parse_args()

    return args


def to_bin(weight,threshold):
    assert len(weight.shape) == 1, "Invalid shape, must input 1 dimension vector"
    max_w = weight.max()
    normalized_weight = weight / max_w
    bin_weight = np.where(normalized_weight >= threshold,1,0)

    return bin_weight


def main():
    args = parse_args()



if __name__ == '__main__':
    main()
