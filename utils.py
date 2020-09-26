import random

import torch
import numpy as np

def print_parameters(model):
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total parameters: {:.4f}M'.format(total / 1e6))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False