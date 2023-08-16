import os
import numpy as np
import time
import pickle
import splearn
from splearn.datasets.base import load_data_sample


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
    # get data path
    INPUT_PATH = '/Users/michaelrizvi/data/PAutomaC-competition_sets/'
    OUTPUT_PATH = '/Users/michaelrizvi/data/wfa2tf-data/'

    n = 1
    model_name = f'{n}.pautomac_model.txt'
    train_file = f'{n}.pautomac.train'
    test_file = f'{n}.pautomac.test'
    sols_file = f'{n}.pautomac_solution.txt'

    # Get data from pautomac dataset
    train_data = load_data_sample(INPUT_PATH + train_file, filetype='Pautomac')
    test_data = load_data_sample(INPUT_PATH + test_file, filetype='Pautomac')
    print(test_data.data.shape)
    print(test_data.data[1])
    print(train_data.data.shape)

    # Extract weights from pautomac dataset
    wfa = splearn.Automaton.load_Pautomac_Automaton(INPUT_PATH + model_name)


    # Compute the WFA states on the same data
    word = test_data.data[1]
    states = get_states(word, wfa)

    print(states.shape)
    print(states[0])
    print(states[-1])
    






    #print(wfa.initial.T@wfa.final)






