from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def load_data(data_dir='../data', input_size=224, batch_size=36):
    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
    train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

    valid_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize])
    train_dataset = datasets.CIFAR10(root=data_dir, train=True, transform=train_transform, download=True)
    valid_dataset = datasets.CIFAR10(root=data_dir, train=False, transform=valid_transform, download=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    return train_loader, valid_loader





