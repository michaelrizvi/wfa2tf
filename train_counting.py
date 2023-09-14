# Import stuff
from tqdm import tqdm
import sys
import os
import numpy as np
from torch import Tensor
import torch.nn.functional as f
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from datetime import datetime
from model import TransformerModel
from splearn.datasets.base import load_data_sample
import torch.nn.functional as F
import wandb
from pathlib import Path
from options import parse_option

USE_WANDB = True


# Define Dataset object from labels and sequences
class SyntheticPautomacDataset(Dataset):
    def __init__(self, label_path, data_path):
        self.labels = torch.Tensor(np.load(label_path)[:, 1:, :])
        self.data_tensor = torch.LongTensor(np.load(data_path)) + 1  # [N, seq_length]
        self.nbL = int(torch.max(self.data_tensor))
        self.nbQ = self.labels.shape[2]
        self.T = self.data_tensor.shape[1]

    def __len__(self):
        return self.data_tensor.shape[0]

    def __getitem__(self, idx):
        return self.data_tensor[idx], self.labels[idx]


def main():
    def reinit_wandb():
        run = wandb.init(reinit=True, tags=[opt.tag])
        wandb.config.update(opt, allow_val_change=True)
        return run

    # Get cmd line args
    opt = parse_option()

    # Set the seed
    torch.manual_seed(opt.seed)

    # Make sure to use CUDA if possible
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load data from files
    nbEx = 10000
    seq_len = 16
    home = str(Path.home())

    OUTPUT_PATH = home + "/data/wfa2tf-data/"
    y_train_file = f"counting_wfa_states_len{seq_len}_size{nbEx}.npy"

    INPUT_PATH = OUTPUT_PATH
    train_file = f"counting_wfa_data_len{seq_len}_size{nbEx}.npy"

    full_set = SyntheticPautomacDataset(
        OUTPUT_PATH + y_train_file, INPUT_PATH + train_file
    )
    train_set, validation_set = torch.utils.data.random_split(full_set, [0.8, 0.2])
    training_loader = DataLoader(train_set, batch_size=opt.batchsize, shuffle=True)
    validation_loader = DataLoader(
        validation_set, batch_size=opt.batchsize, shuffle=True
    )

    ntokens = full_set.nbL + 1  # size of vocabulary
    emsize = 2 * full_set.nbQ**2 + 2  # embedding dimension
    d_hid = emsize  # dimension of the feedforward network model in ``nn.TransformerEncoder``

    nlayers = int(
        np.floor(np.log2(full_set.T))
    )  # number of ``nn.TransformerEncoderLayer`` in ``nn.TransformerEncoder``
    print(full_set.T)
    print(nlayers)
    sys.exit(0)
    nhead = 2  # number of heads in ``nn.MultiheadAttention``
    dropout = opt.dropout  # dropout probability

    model = TransformerModel(
        ntokens, emsize, nhead, d_hid, nlayers, full_set.nbQ, dropout
    ).to(device)

    loss_fn = nn.MSELoss(reduction="mean")

    # Define what to do for one epoch
    def train_one_epoch(model, training_loader, optimizer, epoch_index):
        running_loss = 0.0
        last_loss = 0.0

        # Here, we use enumerate(training_loader) instead of
        # iter(training_loader) so that we can track the batch
        # index and do some intra-epoch reporting
        for i, data in enumerate(training_loader):
            # Every data instance is an input + label pair
            inputs, labels = data
            inputs.to(device)
            labels.to(device)

            # Zero your gradients for every batch!
            optimizer.zero_grad()

            # Make predictions for this batch
            outputs = model(inputs.cuda()).to(device)

            # Compute the loss and its gradients
            loss = loss_fn(outputs.cuda(), labels.cuda()).to(device)
            loss.backward()

            # Adjust learning weights
            optimizer.step()

            # Gather data
            running_loss += loss.item()

        # Report loss at end of loop
        last_loss = running_loss / len(training_loader)  # loss per batch
        print("training loss: {}".format(last_loss))
        running_loss = 0.0

        return last_loss

    ### TRAIN THE MODEL ###
    optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    epoch_number = 0

    EPOCHS = opt.epochs

    best_vloss = 1_000_000

    # Setup wandb
    automata_name = "counting wfa"
    USE_WANDB = opt.use_wandb
    run = wandb.init(project="wfa2tf")
    wandb.run.name = f"{automata_name} T={full_set.T},nQ={full_set.nbQ},epochs={EPOCHS},dropout={dropout},batchsize={opt.batchsize}, lr={opt.lr}"

    for epoch in range(EPOCHS):
        print("EPOCH {}:".format(epoch_number + 1))
        # Make sure gradient tracking is on, and do a pass over the data
        model.train()
        avg_loss = train_one_epoch(model, training_loader, optimizer, epoch)

        # We don't need gradients on to do reporting
        model.eval()

        running_vloss = 0.0
        with torch.no_grad():
            for i, vdata in enumerate(validation_loader):
                vinputs, vlabels = vdata
                vinputs.to(device)
                vlabels.to(device)
                voutputs = model(vinputs.cuda()).to(device)
                vloss = loss_fn(voutputs.cuda(), vlabels.cuda()).to(device)
                running_vloss += vloss.item()

        avg_vloss = running_vloss / len(validation_loader)
        print("LOSS train {} valid {}".format(avg_loss, avg_vloss))

        # Log the running loss averaged per batch
        # for both training and validation
        wandb.log({"training loss": avg_loss}, step=epoch)
        wandb.log({"validation loss": avg_vloss}, step=epoch)

        # Track best performance, and save the model's state
        # if avg_vloss < best_vloss:
        #    best_vloss = avg_vloss
        #    model_path = 'model_{}_{}'.format(timestamp, epoch_number)
        #    torch.save(model.state_dict(), model_path)

        epoch_number += 1


if __name__ == "__main__":
    main()