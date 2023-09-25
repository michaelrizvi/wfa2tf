import os
import numpy as np
import sys
import time
import pickle
import splearn
from splearn.datasets.base import load_data_sample
from tqdm import tqdm
from pathlib import Path

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


if __name__ == "__main__":
    np.random.seed(420)
    home = str(Path.home())
    OUTPUT_PATH = home + '/data/wfa2tf-data/'
    
    # Create an automaton that counts the nb of 0s in alphabet 0,1
    transitions = [np.array([[1.0,0.0],[1.0,1.0]]), np.eye(2)]
    initial = np.array([0.0,1.0])
    final = np.array([1.0,0.0])
    wfa = splearn.automaton.Automaton(nbL=2, nbS=2, initial=initial,
            final=final, transitions=transitions)

    nbEx = 10
    T = 16 
    # Create data tensor
    test_examples_tensor = np.random.randint(wfa.nbL, size=(nbEx, T)) # Maybe there is a better way to generate the data?
    # Does not sample from the "mode" of the data distribution!!!
    np.save(OUTPUT_PATH + f'counting_wfa_data_len{T}_size{nbEx}.npy', test_examples_tensor)

    # Compute states for the data
    test_data_tensor = np.zeros((nbEx, T+1, wfa.nbS))
    for i in range(nbEx):
        test_data_tensor[i] = get_states(test_examples_tensor[i], wfa)

    np.save(OUTPUT_PATH + f'counting_wfa_states_len{T}_size{nbEx}.npy', test_data_tensor)
