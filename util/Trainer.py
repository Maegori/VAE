from datetime import datetime

import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn

LOGPATH = "./logs/VAE_musicgen_log"
MOD_PATH = "./models/VAE_musicgen_model"

CHECKPOINT_INTERVAL = 10
LOAD_FROM_CHECKPOINT = True
SEED = 42
PATIENCE_STOP = False
OUTPUT = False

class Trainer(object):
    """
        General trainer object that can be used to train specific models
        by defining the calc_loss() function for your data and network
    """

    def __init__(self, model, optimizer, criterion, scheduler, trainLoader, testLoader, device='cpu', sample_func=None):
        """
        model:          nn.Module to train with
        optimizer:      optimizer for the model
        criterion:      criterion for the model
        scheduler:      scheduler for the model
        trainLoader:    dataloader for the train set
        testLoader:     dataloader for the test set
        LogPath:        relative path to tensorboard logs
        device:         device on which to train
        sample_func:    function by which to sample from the model (for output during the training)
        """

        self.device = device
        self.writer = SummaryWriter(LOGPATH)

        self.model = model.to(self.device)
        self.trainLoader = trainLoader
        self.testLoader = testLoader
        self.sample_func = sample_func

        self.optim = optimizer
        self.crit = criterion
        self.sched = scheduler

        self._lenTrain = len(trainLoader)
        self._lenTest = len(testLoader)

    def calc_loss(self, *args):
        """Calculates and returns the loss for a batch of data.
        Should be overriden by all subclasses.
        """
        raise NotImplementedError

    def _train(self):
        """Calculate test loss and backpropogates the error
        """
        self.model.train()
        runningLoss = 0

        for batch in self.trainLoader:
            self.optim.zero_grad()
            loss = self.calc_loss(batch.to(self.device))
            runningLoss += loss.detach().item()
            loss.backward()
            self.optim.step()
            self.sched.step()

        return runningLoss / self._lenTrain

    def _validate(self):
        """ Calculate validation loss
        """
        self.model.eval()
        runningLoss = 0

        with torch.no_grad():
            for batch in self.testLoader:
                loss = self.calc_loss(batch.to(self.device))
                runningLoss += loss.detach().item()

        return runningLoss / self._lenTest

    def _update(self, epoch, trainLoss, testLoss):
        """ Printed at every epoch to show progress
        """
        print(
            '{0} --- '.format(datetime.now().time().replace(microsecond=0)),
            'Epoch: {0}\t'.format(epoch),
            'Train loss: {0}\t'.format(round(trainLoss, 2)),
            'Valid loss: {0}\t'.format(round(testLoss, 2))
        )

    def _load_checkpoint(self, path):
        """Function to load a saved model from path
        """
        path += '_checkpoint.pth'

        print("[LOADING CHECKPOINT] {0}".format(path))
        state = torch.load(path)
        self.model.load_state_dict(state['model_state_dict'])
        self.optim.load_state_dict(state['optimizer_state_dict'])
        self.writer.log_dir = state['logger']

        return state['epoch']

    def _save_checkpoint(self, epoch, path):
        """Function to save a model to path
        """
        state = {
            'epoch' : epoch,
            'model_state_dict' : self.model.state_dict(),
            'optimizer_state_dict' : self.optim.state_dict(),
            'logger' : self.writer.log_dir
        }

        torch.save(state, path + "_checkpoint.pth")
        print("[CHECKPOINT CREATED] epoch={0}".format(epoch))

    def sample(self, *args):
        """ Puts the model in evaluation mode to sample from the model
        """
        self.model.eval()
        self.model.to('cpu', non_blocking=True)
        self.sample_func(*args)
        self.model.to(self.device, non_blocking=True)
        self.model.train()

    def run(self, epochs, batch_size):
        """
        epochs:             number of epochs to run
        dictPath:           relative path to saved model  
        batchSize:          size of batches
        """

        if SEED:
            torch.manual_seed(SEED)
        
        bestLoss = float('inf')
        patience = 0
        offset = 1
        
        if LOAD_FROM_CHECKPOINT:
            offset += self._load_checkpoint(MOD_PATH)

        print(
            "[TRAINING] start={0}, num_epochs={1}, batch_size={2}{3}".format(offset, epochs, batch_size, ', seed='+ str(SEED) if SEED else '')
        )

        for epoch in range(offset, epochs + offset):
            trainLoss = self._train()
            testLoss = self._validate()

            self.writer.add_scalars(
                "Loss",
                {
                    "train": trainLoss,
                    "test": testLoss
                },
                epoch
            )

            self._update(epoch, trainLoss, testLoss)
            
            if not epoch % CHECKPOINT_INTERVAL:
                self._save_checkpoint(epoch, MOD_PATH)
                if OUTPUT:
                    self.sample(epoch)

            if PATIENCE_STOP:
                if bestLoss > testLoss:
                    bestLoss = testLoss
                    patience = 1
                else:
                    patience += 1

                if patience > 3:
                    break

        self.writer.close()
        torch.save(self.model.state_dict(), MOD_PATH + '.pth')
        self._save_checkpoint(epoch, MOD_PATH)
        print("Training run completed\n Parameters saved to '{0}'".format(MOD_PATH))

        return self.model