import numpy as np
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
    # get data path
    INPUT_PATH = '/Users/michaelrizvi/data/PAutomaC-competition_sets/'
    OUTPUT_PATH = '/Users/michaelrizvi/data/wfa2tf-data/'

    nb_models = 2
    for n in tqdm(range(1, nb_models+1)):
        model_name = f'{n}.pautomac_model.txt'
        train_file = f'{n}.pautomac.train'
        test_file = f'{n}.pautomac.test'
        sols_file = f'{n}.pautomac_solution.txt'

        # Get data from pautomac dataset
        train_data = load_data_sample(INPUT_PATH + train_file, filetype='Pautomac')
        test_data = load_data_sample(INPUT_PATH + test_file, filetype='Pautomac')

        # Get targets for test
        targets = open(INPUT_PATH + sols_file, "r")
        targets.readline()
        target_probas = [float(line[:-1]) for line in targets]

        # Extract weights from pautomac dataset
        wfa = splearn.Automaton.load_Pautomac_Automaton(INPUT_PATH + model_name)

        # Create data tensor
        test_data_tensor = np.zeros((test_data.nbEx, test_data.data.shape[1] + 1, wfa.nbS)) # The +1 is a fix because of the first alpha state...

        # Compute the WFA states on the same data
        for i in range(test_data.nbEx):
            test_data_tensor[i] = get_states(test_data.data[i], wfa)

        print(test_data_tensor.shape)

        np.save(OUTPUT_PATH + f'{n}.pautomac_states_test.npy', test_data_tensor)

        # Create data tensor
        train_data_tensor = np.zeros((train_data.nbEx, train_data.data.shape[1] + 1, wfa.nbS)) # The +1 is a fix because of the first alpha state...
        #print(train_data_tensor.shape)

        # Compute the WFA states on the same data
        for i in range(train_data.nbEx):
            train_data_tensor[i] = get_states(train_data.data[i], wfa)

        np.save(OUTPUT_PATH + f'{n}.pautomac_states_train.npy', train_data_tensor)

# Questions 
# What is up with the mismatch between solution and computed values??

# How to manage the dataset where outputs are Txd??
# How to pad the output if using max T everywhere?
# How to manage calculating the loss with this kind of data
# Is it necessary to keep the first alpha state?

# How to manage inputs? give the numbers straight? or OHE vectors?
# How to manage the -1 padding tokens

# No targets/solutions for the training set??



