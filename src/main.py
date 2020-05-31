import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn

import data
import src.model as model
import time
import csv
import os


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


def train():
    tot_loss = 0.0
    tot_correct = 0
    tot_lsl = 0.0
    tot_lss_1 = 0.0
    tot_lss_2 = 0.0
    tot_lsd = 0.0
    # itr = 0
    for inputs, labels in train_loader:
        # print("iteration: ", itr)
        # itr += 1
        inputs = inputs.to(device)
        labels = labels.to(device)

        # CutMix regularizer
        label_original = F.one_hot(labels, 10)
        lam = np.random.beta(cutmix_beta, cutmix_beta)
        rand_index = torch.randperm(inputs.size()[0])
        x_cutmix = inputs.clone().detach()
        x_a = inputs[rand_index, :, :, :]
        label_a = label_original[rand_index, :]
        bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), lam)
        M = torch.zeros((inputs.size()[-2], inputs.size()[-1]))

        M = M.to(device)

        M[bbx1:bbx2, bby1:bby2] = 1
        x_cutmix[:, :, bbx1:bbx2, bby1:bby2] = x_a[:, :, bbx1:bbx2, bby1:bby2]
        lam = ((bbx2 - bbx1) * (bby2 - bby1) / (inputs.size()[-1] * inputs.size()[-2]))
        label_cutmix = lam * label_a + (1 - lam) * label_original

        # x_a
        model.eval()
        with torch.no_grad():
            _dummy1, _dummy2, _dummpy3, Y_a = model(x_a)
        # CutMix
        model.train(True)
        optimizer.zero_grad()
        outputs, pool_outputs, M_hat, Y_cutmix = model(x_cutmix)

        # Resize M to H0 * W0
        M = M.unsqueeze(dim=0).unsqueeze(dim=1)
        M = M.repeat(inputs.size()[0], 1, 1, 1)
        M_resizer = torch.nn.MaxPool2d(int(M.size()[-1] / M_hat.size()[-1]))
        M = M_resizer(M)

        lsl = criterion_ce(outputs, labels)
        lss_1 = criterion_lss1(M_hat, M)
        lss_2 = criterion_lss2(M[0, 0, :, :] * Y_cutmix, M[0, 0, :, :] * Y_a)
        lsd = criterion_lss2(outputs, pool_outputs) + 0.5 * criterion_ce(pool_outputs, labels)

        # loss = lsl + lss_1 + lss_2 + lsd
        loss = lsl + lss_1 + lss_2
        # print("lsl", lsl.item())
        # print("lss_1", lss_1.item())
        # print("lss_2", lss_2.item())
        # print("lsd", lsd.item())
        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)
        _, labels = torch.max(label_cutmix, 1)

        tot_loss += loss.item() * inputs.size(0)
        tot_correct += torch.sum(preds == labels.data).item()
        tot_lsl += lsl.item() * inputs.size(0)
        tot_lss_1 += lss_1.item() * inputs.size(0)
        tot_lss_2 += lss_2.item() * inputs.size(0)
        tot_lsd += lsd.item() * inputs.size(0)

    len_ = len(train_loader.dataset)
    epoch_loss = tot_loss / len_
    epoch_acc = tot_correct / len_
    epoch_lsl = tot_lsl / len_
    epoch_lss_1 = tot_lss_1 / len_
    epoch_lss_2 = tot_lss_2 / len_
    epoch_lsd = tot_lsd / len_

    return epoch_loss, epoch_acc, epoch_lsl, epoch_lss_1, epoch_lss_2, epoch_lsd


def valid():
    tot_loss = 0.0
    tot_correct = 0
    tot_lsl = 0.0
    tot_lss_1 = 0.0
    tot_lss_2 = 0.0
    tot_lsd = 0.0
    for inputs, labels in valid_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # CutMix regularizer
        label_original = F.one_hot(labels, 10)
        lam = np.random.beta(cutmix_beta, cutmix_beta)
        rand_index = torch.randperm(inputs.size()[0])
        x_cutmix = inputs.clone().detach()
        x_a = inputs[rand_index, :, :, :]
        label_a = label_original[rand_index, :]
        bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), lam)
        M = torch.zeros((inputs.size()[-2], inputs.size()[-1]))

        M = M.to(device)

        M[bbx1:bbx2, bby1:bby2] = 1
        x_cutmix[:, :, bbx1:bbx2, bby1:bby2] = x_a[:, :, bbx1:bbx2, bby1:bby2]
        lam = ((bbx2 - bbx1) * (bby2 - bby1) / (inputs.size()[-1] * inputs.size()[-2]))
        label_cutmix = lam * label_a + (1 - lam) * label_original

        # x_a
        model.eval()
        with torch.no_grad():
            _dummy1, _dummy2, _dummpy3, Y_a = model(x_a)
        # CutMix
        # model.train(True)
        optimizer.zero_grad()
        outputs, pool_outputs, M_hat, Y_cutmix = model(x_cutmix)

        # Resize M to H0 * W0
        M = M.unsqueeze(dim=0).unsqueeze(dim=1)
        M = M.repeat(inputs.size()[0], 1, 1, 1)
        M_resizer = torch.nn.MaxPool2d(int(M.size()[-1] / M_hat.size()[-1]))
        M = M_resizer(M)

        lsl = criterion_ce(outputs, labels)
        lss_1 = criterion_lss1(M_hat, M)
        lss_2 = criterion_lss2(M[0, 0, :, :] * Y_cutmix, M[0, 0, :, :] * Y_a)
        lsd = criterion_lss2(outputs, pool_outputs) + 0.5 * criterion_ce(pool_outputs, labels)

        # loss = lsl + lss_1 + lss_2 + lsd
        loss = lsl + lss_1 + lss_2

        _, preds = torch.max(outputs, 1)
        _, labels = torch.max(label_cutmix, 1)

        tot_loss += loss.item() * inputs.size(0)
        tot_correct += torch.sum(preds == labels.data).item()
        tot_lsl += lsl.item() * inputs.size(0)
        tot_lss_1 += lss_1.item() * inputs.size(0)
        tot_lss_2 += lss_2.item() * inputs.size(0)
        tot_lsd += lsd.item() * inputs.size(0)

    len_ = len(train_loader.dataset)
    epoch_loss = tot_loss / len_
    epoch_acc = tot_correct / len_
    epoch_lsl = tot_lsl / len_
    epoch_lss_1 = tot_lss_1 / len_
    epoch_lss_2 = tot_lss_2 / len_
    epoch_lsd = tot_lsd / len_

    return epoch_loss, epoch_acc, epoch_lsl, epoch_lss_1, epoch_lss_2, epoch_lsd


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("Device: %s\n" % device_str)
    device = torch.device(device_str)

    # Hyperparameter for Cutmix
    cutmix_beta = 0.3

    # Hyperparameter
    epochs = 100
    lr = 0.001

    train_loader, valid_loader = data.load_data(batch_size=64)
    print("Train samples: %d" % len(train_loader.dataset))
    print("Valid samples: %d" % len(valid_loader.dataset))
    model = model.model()
    model = model.to(device)

    criterion_lss1 = nn.BCELoss()
    criterion_lss2 = nn.KLDivLoss(reduction='batchmean')
    criterion_ce = nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.parameters(), lr=lr)

    time_str = time.strftime("%m_%d-%Hh%Mm%Ss", time.localtime())
    file = open("../log/%s.csv" % time_str, 'w')
    writer = csv.writer(file)
    headers = ["train_loss", "train_acc", "train_lsl", "train_lss_1", "train_lss_2", "train_lsd",
               "valid_loss", "valid_acc", "valid_lsl", "valid_lss_1", "valid_lss_2", "valid_lsd"]

    for epoch in range(epochs):
        print("-" * 5 + "Epoch:  %3d/%3d" % (epoch, epochs) + "-" * 5)
        train_result = train()
        valid_result = valid()
        writer.writerow(train_result+valid_result)
        print("Train Loss: %f Train Accuracy: %f" % (train_result[0], train_result[1]))
        print("Valid Loss: %f Valid Accuracy: %f" % (valid_result[0], valid_result[1]))
        print("-"*25)

    file.close()
