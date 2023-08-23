import os
import numpy as np
import time
import pickle
import splearn
from splearn.datasets.base import load_data_sample
from tqdm import tqdm

def get_states(input, wfa):
    out = wfa.initial 
    state_list = [out]
    for char in input:
        if int(char) >= 0:
            out = out@wfa.transitions[int(char)]
            state_list.append(out)
            #print(out)
        else:
            state_list.append(np.zeros_like(out))

    return np.stack(state_list, axis=0)


if __name__ == "__main__":
    INPUT_PATH = '/Users/michaelrizvi/data/PAutomaC-competition_sets/'
    OUTPUT_PATH = '/Users/michaelrizvi/data/wfa2tf-data/'

    nb_models = 2
    for n in tqdm(range(1, nb_models+1)):
        model_name = f'{n}.pautomac_model.txt'

        # Extract weights from pautomac dataset
        wfa = splearn.Automaton.load_Pautomac_Automaton(INPUT_PATH + model_name)

        nbEx = 1000
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
