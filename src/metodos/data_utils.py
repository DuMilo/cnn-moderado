# nesse arquivo, criamos os DataLoaders para o dataset FashionMNIST.

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def modifica_imagem():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))])
    return transform


def get_dataloaders(data_root, batch_size, shuffle, num_workers):

    # cria e retorna os DataLoaders de treino e teste para o FashionMNIST.
    trainset = datasets.FashionMNIST(
        root=data_root, train=True, download=True, transform=modifica_imagem
    )
    testset = datasets.FashionMNIST(
        root=data_root, train=False, download=True, transform=modifica_imagem
    )

    trainloader = DataLoader(
        trainset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True
    )
    testloader = DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    
    print(f'Train size: {len(trainset)}, Test size: {len(testset)}')

    return trainset, testset, trainloader, testloader