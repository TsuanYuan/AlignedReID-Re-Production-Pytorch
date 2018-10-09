"""
visualize excitation of se model
Quan Yuan
2018-10-08
"""

import argparse
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from struct_format import utils
from evaluate import feature_compute, load_model, misc

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="visualize SE model excitations")

    parser.add_argument('folder', type=str, help="index of training folders, each folder contains multiple pid folders")
    parser.add_argument('model_file', type=str, help="the model file")
    parser.add_argument('--gpu_ids', type=int, default= 0, help="gpu id to use")

    args = parser.parse_args()