import numpy as np
import splearn
from splearn.datasets.base import load_data_sample
from tqdm import tqdm
from pathlib import Path
from sklearn.preprocessing import normalize
from options import parse_option
import sys

np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(suppress=True)

def get_probs(wfa, word=None):
    # 1 get the initial distribuion
    nstates = wfa.nbS

    # 2 calculate term for previous word
    A_w = np.eye(nstates)
    if word:
        for w in word:
            A_w = A_w@wfa.transitions[w]

    # 3 calculate probabilities
    # Create the term for p(Sigma^*)
    A_star = np.eye(nstates)
    for w in range(wfa.nbL):
        A_star = A_star - wfa.transitions[w]

    probs = np.zeros(wfa.nbL)
    # Calculate the probabilities for every letter
    for w in range(wfa.nbL):
        probs[w] = wfa.initial@A_w@wfa.transitions[w]@np.linalg.inv(A_star)@wfa.final

    probs = probs / probs.sum()
    return probs    

def generate_word(wfa, T=16):
    # Get initial distribution and sample a character from it
    word = []
    probs = get_probs(wfa)
    probs = np.where(probs > 0, 1/np.count_nonzero(probs), 0) # Sample uniformly from the support over Sigma
    word.append(np.random.choice(np.arange(wfa.nbL), p=probs))

    # Loop and generate rest of word
    for t in range(T-1):
        probs = get_probs(wfa, word=word)
        probs = np.where(probs > 0, 1/np.count_nonzero(probs), 0) # Sample uniformly from the support over Sigma
        word.append(np.random.choice(np.arange(wfa.nbL), p=probs))

    return word

def get_states(input_seq, wfa):
    out = wfa.initial 
    state_list = [out]
    for char in input_seq:
        if int(char) >= 0:
            #normed_transitions = normalize(wfa.transitions[int(char)], axis=0)
            #out = out@normed_transitions
            out = out@wfa.transitions[int(char)]
            out = out / out.sum() if out.sum() > 0 else out
            state_list.append(out)
        else:
            state_list.append(-1*np.ones_like(out))

    return np.stack(state_list, axis=0)

def main():
    opt = parse_option()

    np.random.seed(opt.seed)
    home = str(Path.home())
    OUTPUT_PATH = home + '/data/wfa2tf-data/'
    INPUT_PATH = home + '/data/PAutomaC-competition_sets/'
    nb_aut = opt.nb_aut 
    model_name = f'{nb_aut}.pautomac_model.txt'

    wfa = splearn.Automaton.load_Pautomac_Automaton(INPUT_PATH + model_name)
    
    T = opt.seqlen 
    nbEx = opt.nbEx 
    inputs= np.zeros((nbEx, T))
    for i in range(nbEx):
        word = generate_word(wfa, T=T)
        inputs[i] = word
    print(inputs.shape)
    np.save(OUTPUT_PATH + f'{nb_aut}.pautomac_synth_data_len{T}_size{nbEx}.npy', inputs)

    outputs = np.zeros((nbEx, T+1, wfa.nbS))
    for i in range(nbEx):
        outputs[i] = get_states(inputs[i], wfa)
    print(outputs.shape)
    np.save(OUTPUT_PATH + f'{nb_aut}.pautomac_synth_states_len{T}_size{nbEx}.npy', outputs)


if __name__ == "__main__":
    main()