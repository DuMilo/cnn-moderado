# nesse arquivo, criamos os DataLoaders para o dataset FashionMNIST.

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloaders(data_root, batch_size, num_workers):

    # cria e retorna os DataLoaders de treino e teste para o FashionMNIST.

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)) # (mean,), (std,) para 1 canal
    ])

    trainset = datasets.FashionMNIST(
        root=data_root, train=True, download=True, transform=transform
    )
    testset = datasets.FashionMNIST(
        root=data_root, train=False, download=True, transform=transform
    )

    trainloader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    testloader = DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    
    print(f'Train size: {len(trainset)}, Test size: {len(testset)}')
    return trainloader, testloader