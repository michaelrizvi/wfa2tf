# Import stuff
from tqdm import tqdm
import sys
import numpy as np
from torch import Tensor
import torch.nn.functional as f
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from model import TransformerModel
from splearn.datasets.base import load_data_sample
import torch.nn.functional as F

# TODO: add wandb support here

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_data(datapath, labelpath):
    data = load_data_sample(datapath, filetype='Pautomac')
    labels = torch.Tensor(np.load(labelpath))
    nbL = data.nbL
    data_tensor = torch.LongTensor(data.data) + 1 # +1 is to remove negative values
    data_list = []
    for i in range(len(data_tensor)):
        data_list.append([data_tensor[i], labels[i,1:,:]])
    loader = torch.utils.data.DataLoader(data_list, shuffle=True, batch_size=4)
    return nbL, loader


# Load data from files
OUTPUT_PATH = '/Users/michaelrizvi/data/wfa2tf-data/'
y_train_file = '1.pautomac_states_train.npy'
y_test_file = '1.pautomac_states_test.npy'

INPUT_PATH = '/Users/michaelrizvi/data/PAutomaC-competition_sets/'
train_file = f'1.pautomac.train'
test_file = f'1.pautomac.test'


nbL, training_loader = load_data(INPUT_PATH+train_file, OUTPUT_PATH+y_train_file)
x, y = next(iter(training_loader))
#print('x',x)
_, T, nbQ = y.shape
_, validation_loader = load_data(INPUT_PATH+test_file, OUTPUT_PATH+y_test_file)

ntokens = nbL + 1 # size of vocabulary
emsize = 2*nbQ**2 + 2  # embedding dimension
d_hid = 2*nbQ**2 + 2  # dimension of the feedforward network model in ``nn.TransformerEncoder``
nlayers = int(np.floor(np.log(T)))  # number of ``nn.TransformerEncoderLayer`` in ``nn.TransformerEncoder``
nhead = 2  # number of heads in ``nn.MultiheadAttention``
dropout = 0.2  # dropout probability

model = TransformerModel(ntokens, emsize, nhead, d_hid, nlayers, dropout).to(device)

loss_fn = nn.MSELoss()
# Define what to do for one epoch 
def train_one_epoch(model, training_loader, optimizer, epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(tqdm(training_loader)):
        # Every data instance is an input + label pair
        inputs, labels = data
        inputs.to(device)
        labels.to(device)

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs).to(device)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss


### TRAIN THE MODEL ### 
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('runs/wfa_trainer_{}'.format(timestamp))
epoch_number = 0

EPOCHS = 1

best_vloss = 1_000_000.

for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch_number + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss = train_one_epoch(model, training_loader, optimizer, epoch, writer)

    # We don't need gradients on to do reporting
    model.train(False)

    running_vloss = 0.0
    for i, vdata in enumerate(validation_loader):
        vinputs, vlabels = vdata
        voutputs = model(vinputs)
        vloss = loss_fn(voutputs, vlabels)
        running_vloss += vloss

    avg_vloss = running_vloss / (i + 1)
    print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

    # Log the running loss averaged per batch
    # for both training and validation
    writer.add_scalars('Training vs. Validation Loss',
                    { 'Training' : avg_loss, 'Validation' : avg_vloss },
                    epoch_number + 1)
    writer.flush()

    # Track best performance, and save the model's state
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = 'model_{}_{}'.format(timestamp, epoch_number)
        torch.save(model.state_dict(), model_path)

    epoch_number += 1
