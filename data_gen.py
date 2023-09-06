import os
import numpy as np
import sys
import time
import pickle
import splearn
from splearn.datasets.base import load_data_sample
from tqdm import tqdm
from pathlib import Path

np.set_printoptions(threshold=sys.maxsize)

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
    INPUT_PATH = home + '/data/PAutomaC-competition_sets/'

    nb_models = 2
    for n in (range(42, nb_models+42)):
        model_name = f'{n}.pautomac_model.txt'

        # Extract weights from pautomac dataset
        wfa = splearn.Automaton.load_Pautomac_Automaton(INPUT_PATH + model_name)

        nbEx = 10000
        T = 16 

        # Create data tensor
        test_examples_tensor = np.random.randint(wfa.nbL, size=(nbEx, T)) # Maybe there is a better way to generate the data?
        # Does not sample from the "mode" of the data distribution!!!
        np.save(OUTPUT_PATH + f'{n}.pautomac_synth_data_len{T}_size{nbEx}.npy', test_examples_tensor)

        # Compute states for the data
        test_data_tensor = np.zeros((nbEx, T+1, wfa.nbS))
        for i in range(nbEx):
            test_data_tensor[i] = get_states(test_examples_tensor[i], wfa)

        np.save(OUTPUT_PATH + f'{n}.pautomac_synth_states_len{T}_size{nbEx}.npy', test_data_tensor)
