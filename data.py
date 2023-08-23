import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, Sampler
import torch.nn.functional as F
from splearn.datasets.data_sample import DataSample

class OneHotDataset(Dataset):
  """ 
    Tranforms a DataSample (splearn) with format
    [3. 2. 0. ...-1. -1. -1.] into a Dataset (torch)
    of one hot with begining with SOS
    input size = nbL + SOS + EOS
    output size = nbL + EOS
  """
  def __init__(self, loaded_data, max_seq_len=None):
    self.data_tensor = torch.LongTensor(loaded_data.data) + 1 #[N, seq_length]
    self.nbL = loaded_data.nbL
    self.sos_symbol = torch.tensor(self.nbL +1) # if not splear loaded_data.max()+1
    self.data_one_hot = F.one_hot(self.data_tensor, self.nbL +2) #[N, seq_length, input_size]
    self.target_one_hot = F.one_hot(self.data_tensor, self.nbL +1) #[N, seq_length, output_size]
    self.max_seq_len = max_seq_len

#   def __getitem__(self, i):
#     word = self.data_one_hot[i]
#     word = torch.cat((F.one_hot(self.sos_symbol).unsqueeze(0), word), dim=0) #concate SOS
#     word = torch.cat((word, F.one_hot(torch.tensor([0]),self.sos_symbol +1 )), dim=0) #add EOS to longest word
#     return word[:-1], word[1:]

  def __getitem__(self, i):
    word = self.data_one_hot[i]
    word = torch.cat((F.one_hot(self.sos_symbol).unsqueeze(0), word), dim=0) #concate SOS
    target = self.target_one_hot[i]
    target = torch.cat((target, F.one_hot(torch.tensor([0]),self.nbL +1)), dim=0) #add EOS to longest word
    # if self.max_seq_len != None:
    #     return word[:self.max_seq_len], target[:self.max_seq_len]
    # else:
    #     return word, target
    return word, target

  def __len__(self):
    return self.data_tensor.shape[0]
  
  def seq_len(self):
    return self.data_tensor.shape[1]+1


def one_hot(word_in_tensor, alphabet_size):
    word_one_hot_tensor = nn.functional.one_hot(word_in_tensor.long(), num_classes=alphabet_size)
    word_in_one_hot = list(word_one_hot_tensor.unbind())
    return word_in_one_hot


def readset(f):
    sett = [] 
    line = f.readline()
    line_list = line.split(" ")
    num_strings = int(line_list[0])
    alphabet_size = int(line_list[1])
    for n in range(num_strings):
        line = f.readline()
        line_list = line.split(" ")
        sett = sett + [[int(i) for i in line_list[1:len(line_list)]]]
    return alphabet_size, sett


def get_list_of_words_PAUTOMAC(file):
    alphabet_size, train = readset(open(file,"r"))
    train_clean = [word for word in train if word != []]
    list_of_words_PAUTOMAC = index_to_one_hot(train_clean, alphabet_size)
    return list_of_words_PAUTOMAC

def get_list_of_words_PAUTOMAC_with_EOS(file):
    alphabet_size, train = readset(open(file,"r"))
    # print("alpha size", alphabet_size)
    # print("train", train[0:3])
    train_clean = [word for word in train]
    print("train_clean", train_clean[0:3])
    train_clean_with_EOS = add_EOS(train_clean)
    list_of_words_PAUTOMAC = index_to_one_hot(train_clean_with_EOS, alphabet_size+1)
    return list_of_words_PAUTOMAC

def get_word_prob_PAUTOMAC(file):
    """
    word_prob = 1D tensor
    """
    solution_file = open(file, "r")
    lines = solution_file.read().splitlines()
    solution_file.close()
    prob_list = []
    for i in lines:
        prob_list.append(float(i))
    word_prob = torch.FloatTensor(prob_list[1:])
    return word_prob


def index_to_one_hot(list_of_words_in_index, alphabet_size):
    list_of_words_in_one_hot = []
    for word_in_index in list_of_words_in_index:
        word_in_one_hot = one_hot(torch.Tensor(word_in_index), alphabet_size)
        list_of_words_in_one_hot.append(word_in_one_hot)
    return list_of_words_in_one_hot


def sort_by_lenght(list_of_words):
    list_of_word_len = []
    max_word_length = 0
    for word in list_of_words:
        list_of_word_len.append(len(word))
        if len(word) > max_word_length:
            max_word_length = len(word)

    dataset_size = len(list_of_words)
    index_sorted_len_list = sorted(range(len(list_of_word_len)), key=lambda x: list_of_word_len[x])
    sorted_list_of_words = [list_of_words[i] for i in index_sorted_len_list]
    # sorted_len_of_words = [list_of_word_len[i] for i in index_sorted_len_list]
    return sorted_list_of_words


def make_dataset_and_mask(list_of_words, required_max_len=None):
    dataset_size = len(list_of_words)
    in_size = len(list_of_words[0][0])
    if required_max_len == None:
        max_len = max([len(word) for word in list_of_words])
    else : 
        max_len = required_max_len
    dataset = torch.zeros(dataset_size, max_len, in_size)
    mask = torch.zeros(dataset_size, max_len, in_size)
    for i, word in enumerate(list_of_words):
        for j, letter in enumerate(word):
            dataset[i, j, :] = letter
            mask[i, j, :] = 1
    dataset = dataset[:, 1:, :]
    mask = mask[:, 1:, :]
    return dataset, mask


class WordsDataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # IMPORTANT: a list of tensor doesn't work as a data element
        # hence the need to make it a tensor of rank 2
        return torch.stack(self.data[index])


class BatchSamplerByLenght(Sampler):
    '''
    Sampler that assemble batches of words with similar lenghts.

    Parameters:
    `list_of_words` (list) : same list used in the dataset
    `batch_size` (int) : maximum size of batches
    `same_lenght` (bool) : makes all words of a batch the same lenght (default=`True`)
    `drop_last` (bool) : (default=`False`)

    this sampler is all about saving an `index_sorted_len_list` which is a list
    of indices `idx`. For example, here, `idx` encodes the order to retreive
    words by lenght
    ```
    i word idx
    0 ___  2  
    1 _    0
    2 __   1
    3 ____ 4
    4 ___  3
    ```
    For `idx` in `index_sorted_len_list`, a batch can be assembled with words of 
    similar lenght. If `same_lenght` flag is `True`, batches will be of
    arbitrary sizes but will only contain words of the same lenght.
    '''
    def __init__(self, list_of_words, batch_size, same_lenght=True, drop_last=False):
        super().__init__(list_of_words)
        self.list_of_words = list_of_words
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.same_lenght = same_lenght

        list_of_word_len = [] 
        for word in list_of_words:
            list_of_word_len.append(len(word))

        self.list_of_word_len = list_of_word_len
        self.index_sorted_len_list = sorted(range(len(list_of_word_len)), key=lambda x: list_of_word_len[x])

    def __iter__(self):
        batch = []
        first_idx = self.index_sorted_len_list[0]
        first_lenght = self.list_of_word_len[first_idx]

        for i, idx in enumerate(self.index_sorted_len_list):
            batch.append(idx)

            lenght = self.list_of_word_len[idx]
            try:
                next_idx = self.index_sorted_len_list[i+1]
                next_lenght = self.list_of_word_len[next_idx]
            except IndexError:
                pass

            is_batch_full = len(batch) == self.batch_size
            is_lenght_changed = lenght != next_lenght
            if (self.same_lenght and is_lenght_changed) or is_batch_full:
                yield batch
                batch = []

        if len(batch) > 0 and (not self.drop_last or self.same_lenght):
            yield batch

    def __len__(self):
        raise NotImplementedError('Our custom sampler does not provide a lenght')


### DATA WITHOUT DATALOADER


def split_dataset(dataset, fraction_train):
    cut = round(fraction_train*dataset.shape[0])
    dataset_train = dataset[:cut, :, :]
    dataset_valid = dataset[cut:, :, :]
    return dataset_train, dataset_valid


def get_labels(dataset):
    """
    Generate labels from and for dataset of words
    Labels are same words shifted by one letter
    """
    seq_length = dataset.shape[1]
    dataset_size = dataset.shape[0]
    input_size = dataset.shape[2]
    dataset_short = torch.narrow(dataset, 1, 1, seq_length-1)
    dataset_labels = torch.zeros((dataset_size, seq_length, input_size))
    dataset_labels[:, :seq_length-1, :] = dataset_short

    return dataset_labels


def get_batch(x, i, batch_size):
    """
    x.shape= (dataset size, seq_len, input_size)
    batch_x.shape = (seq_len, batch_size, input_size)
    """  
    if x.ndim == 3:
        batch_x = x[(i*batch_size):(i*batch_size)+batch_size, :, :]
        batch_x = batch_x.transpose(1, 0)
    else: 
        batch_x = x[(i*batch_size):(i*batch_size)+batch_size]
    return batch_x

def add_EOS(list_of_words):
    list_with_EOS = []
    for word in list_of_words:
        if word != []:
            new_word = [letter+1 for letter in word]
        else: 
            new_word = []    
        new_word.append(0)
        list_with_EOS.append(new_word)
    return list_with_EOS

def get_training_subset_for_spectral_learning(onehotdata_subset):
    '''
    onehotdata_subset[i] = (word, target) : get item i for the set which is 
    word.shape = [seq len, alphabet+2]
    target.shape = [seq len, alphabet+1]
    '''
    number_of_samples = len(onehotdata_subset)
    seq_len = onehotdata_subset[1][0].shape[0]-1 #without SOS
    alphabet = onehotdata_subset[1][0].shape[1]-2 #without SOS and EOS
    spectral_training_set = np.empty(shape=(number_of_samples, seq_len))
    for i, item in enumerate(onehotdata_subset): 
        onehot_word, onehot_target = item
        word = torch.argmax(onehot_word[1:], dim=-1)
        spectral_training_set[i] =  word.numpy()-1
    subset_for_spectral_learning = DataSample(data=(alphabet, number_of_samples, spectral_training_set))
    return subset_for_spectral_learning

def compute_size_of_subsets(set_to_divide, fractions=(0.8, 0.1, 0.1)):
    size_training = int(len(set_to_divide)*fractions[0])
    size_validation = int(len(set_to_divide)*fractions[1])
    size_test = int(len(set_to_divide)*fractions[2])
    diff = len(set_to_divide) - (size_training + size_validation + size_test)
    assert diff >= 0, 'Chosen fractions exceed 1'
    if diff == 1:
        lengths = [size_training + 1, size_validation, size_test]
    elif diff == 2:
        lengths = [size_training, size_validation + 1, size_test + 1]
    else :
        lengths = [size_training, size_validation, size_test]
    return lengths

def load_targets_spice(file):
    targets = []
    f = open(file, "r")
    line = f.readline()
    while line:
        l = line.split()
        targets.append(int(l[1])+1)
        line = f.readline()
    return targets
