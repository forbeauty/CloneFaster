import os
import math
import yaml
import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from transformers import get_linear_schedule_with_warmup
from .dataset import PretrainDataset
from .model import InnoModel
from evaluate import evaluate


# For reproducibility.
def seed_everything(my_seed=0):
    np.random.seed(my_seed)
    os.environ['PYTHONHASHSEED'] = str(my_seed)
    torch.manual_seed(my_seed)
    torch.cuda.manual_seed_all(my_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def train(config, args, device, logger):
    # Dataset
    pretrain_dataset = PretrainDataset(config, overfit=args.overfit)

    logger.info(
        'Training set number of samples: {}'.format(len(train_dataset))
    )
    logger.info(
        'Validation set number of samples: {}'.format(len(val_dataset))
    )

    assert(
        config['solver']['batch_size']
        % config['solver']['accumulation_steps'] == 0
    )
    actual_batch_size = (
        config['solver']['batch_size']
        // config['solver']['accumulation_steps']
    )
    logger.info('Acture batch size: {}'.format(actual_batch_size))
    logger.info(
        'Gradient accumulation steps: {}'
        .format(config['solver']['accumulation_steps'])
    )
    logger.info(
        'Effective batch size: {}'.format(config['solver']['batch_size'])
    )

    pretrain_dataloader = DataLoader(
        pretrain_dataset,
        batch_size=actual_batch_size,
        shuffle=True,
        num_workers=args.cpu_workers
    )


    # Model
    model = InnoModel(config).to(device)
