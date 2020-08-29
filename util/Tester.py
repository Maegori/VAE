import torch

class Tester(object):

    def __init__(self, model, mod_path, sample_func, device='cpu'):
        self.model = model
        self.device = device
        self.sample_func = sample_func

        self.model.to(self.device)
        self.model.eval()
        self.epoch = self._load_checkpoint(mod_path)

    def _load_checkpoint(self, path):
        path += '_checkpoint.pth'

        print("[LOADING CHECKPOINT] {0}".format(path))
        state = torch.load(path)
        self.model.load_state_dict(state['model_state_dict'])

        return state['epoch']

    def sample(self, *args):
        self.sample_func(*args)

