import argparse
import pickle
from argparse import RawTextHelpFormatter
from ast import literal_eval

from tqdm.auto import tqdm

def parse_option():
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)

    parser.add_argument('--seed', type=int, default=0,
                        help='seed')

    parser.add_argument('--epochs', type=int, default=10,
                        help='number of training epochs for SGD / max iterations for ALS')

    parser.add_argument('--batchsize', type=int, default=16,
                        help="batch size for training and validation sets")

    parser.add_argument('--dropout', type=float, default=0.2,
                        help="dropout probability")

    parser.add_argument('--lr', type=float, default=0.001,
                        help="learning rate for the optimizer")

    parser.add_argument('--use_wandb', type=bool, default=False)


    opt = parser.parse_args()

    return opt
