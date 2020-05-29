import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter

import data
import src.model as model

import time


if __name__ == "__main__":
    train_loader, test_loader = data.load_data(batch_size=64)
    model = model.model()

    time_str = time.strftime("%m_%d-%Hh%Mm%Ss", time.localtime())
    tb_writer = SummaryWriter(log_dir="../log/%s" % time_str)
    data_iter = iter(train_loader)
    tb_writer.add_graph(model, next(data_iter)[0])
    tb_writer.add_scalars(
        "Loss",
        {'Training': 1, 'Validation': 1},
        1
    )

    # test size match
    for inputs, labels in train_loader:
        outputs = model(inputs)
        print(outputs)
        print(outputs.size())
        break
