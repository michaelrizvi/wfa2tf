import argparse
import pickle
from argparse import RawTextHelpFormatter
from ast import literal_eval

from tqdm.auto import tqdm

def parse_option():
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)

    parser.add_argument('--seed', type=int, default=0,
                        help='seed')

    parser.add_argument('--epochs', type=int, default=100,
                        help='number of training epochs for SGD')

    parser.add_argument('--batchsize', type=int, default=32,
                        help="batch size for training and validation sets")

    parser.add_argument('--dropout', type=float, default=0.2,
                        help="dropout probability")

    parser.add_argument('--lr', type=float, default=0.001,
                        help="learning rate for the optimizer")

    parser.add_argument('--debug', type=bool, default=False)

    parser.add_argument('--nlayers', type=int, default=4,
                        help="number of layers for the transformer")

    parser.add_argument('--nbEx', type=int, default=10000,
                        help="number of examples in the training set")

    parser.add_argument('--seqlen', type=int, default=16,
                        help="number of tokens in a sequence")

    parser.add_argument('--emsize', type=int, default=16,
                        help="embedding dimension of the transformer")

    parser.add_argument('--d_hid', type=int, default=16,
                        help="hidden layer size or the transformer")

    parser.add_argument('--nhead', type=int, default=2,
                        help="number of heads for multihead attention")

    parser.add_argument('--ntry', type=int, default=1,
                        help="number of times we train the model (for repeatability)")

    parser.add_argument('--nb_aut', type=int, default=4,
                        help="number corresponding to automata (for Pautomac experiments)")

    parser.add_argument('--k', type=int, default=1,
                        help="number of characters to count for k counting automata")

    parser.add_argument('--nbL', type=int, default=2,
                        help="number of letters in the alphabet")

    opt = parser.parse_args()

    return opt
