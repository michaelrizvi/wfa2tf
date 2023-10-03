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
from utils import SyntheticDataset

def main():
    # Get cmd line args
    opt = parse_option()

    # Set the seed
    torch.manual_seed(opt.seed)

    # Make sure to use CUDA if possible
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(device)

    # Load data from files
    nbEx = opt.nbEx
    seq_len = opt.seqlen 
    home = str(Path.home())

    home = str(Path.home())
    PATH = home + "/data/wfa2tf-data/"
    output_file = f'{opt.nb_aut}.pautomac_synth_states_len{opt.seqlen}_size{opt.nbEx}.npy' 
    input_file = f'{opt.nb_aut}.pautomac_synth_data_len{opt.seqlen}_size{opt.nbEx}.npy' 
    full_set = SyntheticDataset(
        PATH + output_file, PATH + input_file
    )
    train_set, validation_set, test_set = torch.utils.data.random_split(full_set, [0.8, 0.1, 0.1])
    training_loader = DataLoader(train_set, batch_size=opt.batchsize, shuffle=True)
    validation_loader = DataLoader(validation_set, batch_size=opt.batchsize, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=opt.batchsize, shuffle=True)

    ntokens = full_set.nbL + 1  # size of vocabulary
    
    model = TransformerModel(
        ntokens, opt.emsize, opt.nhead, opt.d_hid, opt.nlayers, full_set.nbQ, opt.dropout
    ).to(device)

    loss_fn = nn.MSELoss()

    # Define what to do for one epoch
    def train_one_epoch(model, training_loader, optimizer, epoch_index):
        running_loss = 0.0
        last_loss = 0.0

        for i, data in enumerate(training_loader):
            # Every data instance is an input + label pair
            inputs, labels = data
            inputs.to(device)
            labels.to(device)

            # Zero your gradients for every batch!
            optimizer.zero_grad()

            # Make predictions for this batch
            outputs = model(inputs.to(device), use_softmax=True).to(device)
#            print(outputs)

            # Compute the loss and its gradients
            loss = loss_fn(outputs.to(device), labels.to(device)).to(device)
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
    def validate_one_epoch(model, loader, loss_fn, is_eval =False):
        running_loss = 0.0
        with torch.no_grad():
            for i, data in enumerate(loader):
                inputs, labels = data
                inputs.to(device)
                labels.to(device)
                outputs = model(inputs.to(device), use_softmax=True).to(device)
                if (i == 0 or i == len(loader) - 1) and is_eval==True:
                    print("inputs:", inputs)
                    print("outputs: ",outputs)
                    print("labels", labels)
                    print("diff: ",loss_fn(outputs.to(device), labels.to(device)))
                loss = loss_fn(outputs.to(device), labels.to(device)).to(device)
                running_loss += loss.item()

        avg_loss = running_loss / len(loader)
        return avg_loss

    ### TRAIN THE MODEL ###
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr)

    # Setup wandb
    automata_name = f"{opt.nb_aut} pautomac"
    if not opt.debug:
        run = wandb.init(project="wfa2tf")
        wandb.run.name = f"{automata_name} T={full_set.T}, nlayers={opt.nlayers}"

    test_losses = []
    for n in range(opt.ntry):
        epoch_number = 0
        EPOCHS = opt.epochs
        best_vloss = 1_000_000
        model.init_weights()
        for epoch in range(EPOCHS):
            print("EPOCH {}:".format(epoch_number + 1))
            # Make sure gradient tracking is on, and do a pass over the data
            model.train()
            avg_loss = train_one_epoch(model, training_loader, optimizer, epoch)

            # We don't need gradients on to do reporting
            model.eval()
            avg_vloss = validate_one_epoch(model, validation_loader, loss_fn)
            print("LOSS train {} valid {}".format(avg_loss, avg_vloss))

            # Track best performance, and save the model's state
            if avg_vloss < best_vloss:
                best_vloss = avg_vloss
                model_path = f'models/best_model_nlayer{opt.nlayers}_pautomac{opt.nb_aut}'
                torch.save(model.state_dict(), model_path)

            epoch_number += 1

        # Evaluation on the test set
        model.load_state_dict(torch.load(model_path))
        model.eval()
        test_loss = validate_one_epoch(model, test_loader, loss_fn, is_eval=False)
        test_losses.append(test_loss)
        print("LOSS train {} valid {}, test {}".format(avg_loss, avg_vloss, test_loss))
        if not opt.debug:
            wandb.log({"training loss": avg_loss})
            wandb.log({"validation loss": avg_vloss})

    avg_test_loss = np.mean(test_losses)
    min_test_loss = np.min(test_losses)
    if not opt.debug:
        wandb.log({"avg test loss": avg_test_loss})
        wandb.log({"min test loss": min_test_loss})
    print("average test loss:", avg_test_loss)
    print("min test loss:", min_test_loss)

     


if __name__ == "__main__":
    main()
