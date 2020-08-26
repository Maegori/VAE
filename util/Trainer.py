from datetime import datetime

import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn


class Trainer(object):
    """
        General trainer object that can be used to train specific models
        by defining the calc_loss() function for your data and network
    """

    def __init__(self, model, optimizer, criterion, trainLoader, testLoader, logPath):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.writer = SummaryWriter(logPath)

        self.model = model.to(self.device)
        self.trainLoader = trainLoader
        self.testLoader = testLoader

        self.optim = optimizer
        self.crit = criterion

        self._lenTrain = len(trainLoader.dataset)
        self._lenTest = len(testLoader.dataset)

    def calc_loss(self, *args):
        """Calculates and returns the loss for a batch of data.
        Should be overriden by all subclasses.
        """
        raise NotImplementedError

    def _train(self):
        self.model.train()
        runningLoss = 0

        for batch in self.trainLoader:
            self.optim.zero_grad()
            
            loss = self.calc_loss(batch.to(self.device))
            runningLoss += loss.item()
            loss.backward()
            self.optim.step()

        return runningLoss / self._lenTrain

    def _validate(self):
        self.model.eval()
        runningLoss = 0

        with torch.no_grad():
            for batch in self.testLoader:
                loss = self.calc_loss(batch.to(self.device))
                runningLoss += loss.item()

        return runningLoss / self._lenTest

    def _update(self, epoch, trainLoss, testLoss):
        print(
            f'{datetime.now().time().replace(microsecond=0)} --- '
            f'Epoch: {epoch}\t'
            f'Train loss: {trainLoss:.2f}\t'
            f'Valid loss: {testLoss:.2f}\t'
        )

    def _load_checkpoint(self, path):
        path += '_checkpoint.pth'

        print(f"[LOADING CHECKPOINT] {path}")
        state = torch.load(path)
        self.model.load_state_dict(state['model_state_dict'])
        self.optim.load_state_dict(state['optimizer_state_dict'])
        self.writer.log_dir = state['logger']

        return state['epoch']

    def _save_checkpoint(self, epoch, path):
        state = {
            'epoch' : epoch,
            'model_state_dict' : self.model.state_dict(),
            'optimizer_state_dict' : self.optim.state_dict(),
            'logger' : self.writer.log_dir
        }

        torch.save(state, path + "_checkpoint.pth")
        print(f"[CHECKPOINT CREATED] epoch={epoch}")


    def run(self, epochs, dictPath, batchSize=64, checkpointInterval=20, checkpoint=False, seed=None, output=False):
        if seed:
            torch.manual_seed(seed)
        
        bestLoss = float('inf')
        patience = 0
        offset = 1
        
        if checkpoint:
            offset += self._load_checkpoint(dictPath)

        print(
            f"[TRAINING] start={offset}, num_epochs={epochs}, batch_size={batchSize}{', seed='+ str(seed) if seed else ''}"
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
            
            if not epoch % checkpointInterval:
                self._save_checkpoint(epoch, dictPath)
                if output:
                    self.model.producer(epoch=epoch)

            if bestLoss > testLoss:
                bestLoss = testLoss
                patience = 1
            else:
                patience += 1

            if patience > 3:
                break

        self.writer.close()
        torch.save(self.model.state_dict(), dictPath + '.pth')
        self._save_checkpoint(epoch, dictPath)
        print(f"Training run completed\n Parameters saved to '{dictPath}'")

        return self.model