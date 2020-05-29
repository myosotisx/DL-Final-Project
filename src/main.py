import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F


import numpy as np

import data
import src.model as model

import time


def rand_bbox(size, lam):
    W = size[-2]
    H = size[-1]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


if __name__ == "__main__":
    # Hyperparameter for Cutmix
    cutmix_beta = 0.3


    train_loader, test_loader = data.load_data(batch_size=64)
    model = model.model()

    # time_str = time.strftime("%m_%d-%Hh%Mm%Ss", time.localtime())
    # tb_writer = SummaryWriter(log_dir="../log/%s" % time_str)
    # data_iter = iter(train_loader)
    # tb_writer.add_graph(model, next(data_iter)[0])
    # tb_writer.add_scalars(
    #     "Loss",
    #     {'Training': 1, 'Validation': 1},
    #     1
    # )

    # test size match
    for inputs, labels in train_loader:
        # CutMix regularizer
        y_original = F.one_hot(labels, 10)
        lam = np.random.beta(cutmix_beta, cutmix_beta)
        rand_index = torch.randperm(inputs.size()[0])
        x_cutmix = inputs.clone().detach()
        x_a = inputs[rand_index, :, :, :]
        y_a = y_original[rand_index, :]
        bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), lam)
        M = torch.zeros((inputs.size()[-2], inputs.size()[-1]))
        M[bbx1:bbx2, bby1:bby2] = 1
        x_cutmix[:, :, bbx1:bbx2, bby1:bby2] = x_a[:, :, bbx1:bbx2, bby1:bby2]
        lam = ((bbx2 - bbx1) * (bby2 - bby1) / (inputs.size()[-1] * inputs.size()[-2]))
        y_cutmix = lam * y_a + (1 - lam) * y_original

        # outputs = model(inputs)
        # print(outputs)
        # print(outputs.size())
        break

