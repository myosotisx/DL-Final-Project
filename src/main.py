import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

import data
import src.model as model


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

    criterion_lss1 = nn.BCELoss()
    criterion_lss2 = nn.KLDivLoss(reduction='batchmean')
    criterion_ce = nn.CrossEntropyLoss()

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
        label_original = F.one_hot(labels, 10)
        lam = np.random.beta(cutmix_beta, cutmix_beta)
        rand_index = torch.randperm(inputs.size()[0])
        x_cutmix = inputs.clone().detach()
        x_a = inputs[rand_index, :, :, :]
        label_a = label_original[rand_index, :]
        bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), lam)
        M = torch.zeros((inputs.size()[-2], inputs.size()[-1]))
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
        outputs, pool_outputs, M_hat, Y_cutmix = model(inputs)

        # Resize M to H0 * W0
        M = M.unsqueeze(dim=0).unsqueeze(dim=1)
        M = M.repeat(inputs.size()[0], 1, 1, 1)
        M_resizer = torch.nn.MaxPool2d(int(M.size()[-1] / M_hat.size()[-1]))
        M = M_resizer(M)

        lsl = criterion_ce(outputs, labels)
        lss_1 = criterion_lss1(M_hat, M)
        lss_2 = criterion_lss2(M[0, 0, :, :] * Y_cutmix, M[0, 0, :, :] * Y_a)
        lsd = criterion_lss2(outputs, pool_outputs) + 0.5 * criterion_ce(pool_outputs, labels)
        print(lsl)
        print(lss_1)
        print(lss_2)
        print(lsd)
        break
