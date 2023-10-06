import os
import numpy as np
import sys
import time
import pickle
import splearn
from splearn.datasets.base import load_data_sample
from tqdm import tqdm
from pathlib import Path
from options import parse_option

def get_states(input_seq, wfa):
    out = wfa.initial 
    state_list = [out]
    for char in input_seq:
        if int(char) >= 0:
            out = out@wfa.transitions[int(char)]
            state_list.append(out)
        else:
            state_list.append(np.zeros_like(out))

    return np.stack(state_list, axis=0)

def kcounting_automata(k, nbL, mask=None):
    # Mask is used to choose which symbols we are counting
    # If no mask basic mask is to count k first letters of Sigma
    if mask is None:
        mask = np.zeros(nbL)
        mask[:k] = 1
        mask = mask != 0
    # Make initial and final vectors
    initial = np.zeros(k+1)
    initial[-1] = 1

    # Make final vector
    final = np.zeros(k+1)
    final[0] = 1

    # Make transition maps
    transitions=[]
    for i in range(nbL):
        I = np.eye(k+1)
        if mask[i]:
            I[-1,i] += 1
            A = I
        else:
            A = np.eye(k+1)
        transitions.append(A)
    wfa = splearn.automaton.Automaton(nbL=nbL, nbS=k+1, initial=initial,
            final=final, transitions=transitions) 
    return wfa

if __name__ == "__main__":
    # Get cmd line args
    opt = parse_option()

    np.random.seed(opt.seed)
    home = str(Path.home())
    OUTPUT_PATH = home + '/data/wfa2tf-data/'
    
    # nbL = 10, k = 2,4,6,8
    wfa = kcounting_automata(opt.k, opt.nbL)
    automata_name = f'counting_automata_k{opt.k}_nbL{opt.nbL}'
    nbEx = opt.nbEx 
    T = opt.seqlen 
    # Create data tensor
    inputs = np.random.randint(wfa.nbL, size=(nbEx, T)) # Maybe there is a better way to generate the data?
    # Does not sample from the "mode" of the data distribution!!!
    np.save(OUTPUT_PATH + f'{automata_name}_data_len{T}_size{nbEx}.npy', inputs)

    # Compute states for the data
    outputs = np.zeros((nbEx, T+1, wfa.nbS))
    for i in range(nbEx):
        outputs[i] = get_states(inputs[i], wfa)
    np.save(OUTPUT_PATH + f'{automata_name}_states_len{T}_size{nbEx}.npy', outputs)
