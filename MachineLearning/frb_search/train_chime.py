import numpy as np
from dataclasses import dataclass
from argparse import ArgumentParser
import logging
import math

from torch import nn
from torch.optim import SGD
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torchvision.transforms as transforms

from ignite.engine import (
    Events, create_supervised_trainer, create_supervised_evaluator)
from ignite.metrics import Accuracy, Loss
from ignite.handlers import ModelCheckpoint

import torch 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

@dataclass(eq=False)
class singlePulseSet(Dataset):
    """ Dataset class sublcassed from torch's dataset utility
    Args:
            input_size (int): input size of a datapoint or in this case, the binary size
            start (int): whole number from where the dataset starts counting, exclusive
            end (int): whole number till where the dataset keeps counting, inclusive
    """
    start: int 
    def __getitem__(self, idx):
        value=self.start+idx
        filename = str('/data/repeaters/training_data/train_data_')+str(value)+str('.dat')
        x = np.fromfile(filename, dtype='float32')
        y = int(x[6])
        x = x[7:]
        std = np.std(x)
        if math.isnan(std):
            print(str('smothing is wrong nan ')+str(filename))
        elif std == 0:
            print(str('smothing is wrong 0 ')+str(filename)) 
        x = x-np.mean(x)
        x = x/std
        x = np.reshape(x,(256,256))
        x = torch.from_numpy(x)
        x = x.unsqueeze(0)
        return x, y

    def __len__(self):
        """ setting the length to a limit. Theoretically fizbuz dataset can have
        infinitily long dataset but dataloaders fetches len(dataset) to loop decide
        what's the length. Returning any number from this function sets that as the
        length of the dataset
        """
        if(self.start==0):
            l=400000
        elif (self.start==400000):
            l=50000
        else:
            l=50
        return l
        
trainset = singlePulseSet(0)
testset = singlePulseSet(400000)


def get_data_loaders(train_batch_size, val_batch_size):
    
    train_loader = DataLoader(trainset, batch_size=train_batch_size,shuffle=True, num_workers=2)
    val_loader = DataLoader(testset, batch_size=val_batch_size,shuffle=True, num_workers=2)
    return train_loader, val_loader
    

import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=1, padding=3),
            nn.Dropout2d(p=0.3),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2))
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.Dropout2d(p=0.3),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2))
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.Dropout2d(p=0.3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.Dropout2d(p=0.3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.layer5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.Dropout2d(p=0.3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.drop_out1 = nn.Dropout(0.3)
        self.fc1 = nn.Linear(8 * 8 * 512, 4096)
        self.drop_out2 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(4096, 128)
        self.fc3 = nn.Linear(128, 2)

    def forward(self, x):
       out = self.layer1(x)
       out = self.layer2(out)
       out = self.layer3(out)
       out = self.layer4(out)
       out = self.layer5(out)
       out = out.reshape(out.size(0), -1)
       out = self.drop_out1(out)
       out = self.fc1(out)
       out = self.drop_out2(out) 
       out = self.fc2(out)
       out = self.fc3(out)
       return out             

def run(train_batch_size, val_batch_size,
        epochs, lr, momentum,
        log_interval, restore_from, crash_iteration=150000):

    train_loader, val_loader = get_data_loaders(train_batch_size, val_batch_size)
    model = Net()
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    optimizer = SGD(model.parameters(), lr=lr, momentum=momentum)
    trainer = create_supervised_trainer(model, optimizer, nn.CrossEntropyLoss(), device=device)
    evaluator = create_supervised_evaluator(model, metrics={'accuracy': Accuracy(),
                                                     'nll': Loss(nn.CrossEntropyLoss())},
                                            device=device)
    # Setup debug level of engine logger:
    trainer._logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s|%(name)s|%(levelname)s| %(message)s")
    ch.setFormatter(formatter)
    trainer._logger.addHandler(ch)

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        iter = (engine.state.iteration - 1) % len(train_loader) + 1
        if iter % log_interval == 0:
            print("Epoch[{}] Iteration[{}/{}] Loss: {:.5f}"
                  "".format(
                      engine.state.epoch, iter,
                      len(train_loader), engine.state.output))

        if engine.state.iteration == crash_iteration:
            raise Exception("STOP at {}".format(engine.state.iteration))

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        evaluator.run(train_loader)
        metrics = evaluator.state.metrics
        avg_accuracy = metrics['accuracy']
        avg_nll = metrics['nll']
        print(
            "Training Results - Epoch: {}\
            Avg accuracy: {:.5f} Avg loss: {:.5f}".format(
                engine.state.epoch, avg_accuracy, avg_nll))

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        avg_accuracy = metrics['accuracy']
        avg_nll = metrics['nll']
        print(
            "Validation Results - Epoch: {}\
            Avg accuracy: {:.5f} Avg loss: {:.5f}".format(
                engine.state.epoch, avg_accuracy, avg_nll))

    objects_to_checkpoint = {"model": model, "optimizer": optimizer}
    engine_checkpoint = ModelCheckpoint(
        dirname="/data/repeaters/models/",
        filename_prefix='fifth',
        require_empty=False,
        save_interval=10000)
    trainer.add_event_handler(
        Events.ITERATION_COMPLETED, engine_checkpoint, objects_to_checkpoint)

    if restore_from == "":
        trainer.run(train_loader, max_epochs=epochs)
    else:
        raise NotImplementedError('Not implemented yet')


run(20, 20,5, 0.01, 0.9,1000, "", 1500000)

