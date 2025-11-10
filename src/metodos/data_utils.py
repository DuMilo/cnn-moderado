# nesse arquivo, criamos os DataLoaders para o dataset FashionMNIST.
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def modifica_imagem():
    """
    Cria uma transformação composta para as imagens do dataset FashionMNIST.

    A transformação converte a imagem para tensor e normaliza os valores dos pixels 
    com média 0.5 e desvio padrão 0.5, preparando os dados para entrada em modelos 
    baseados em PyTorch.

    Returns:
        torchvision.transforms.Compose: Transformação composta aplicada nas imagens.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))])
    return transform


def get_dataloaders(data_root, batch_size, shuffle, num_workers):
    """
    Cria e retorna os DataLoaders e datasets de treino e teste.

    Args:
        data_root (str): Diretório raiz onde os dados serão armazenados ou carregados.
        batch_size (int): Número de amostras por batch para os DataLoaders.
        shuffle (bool): Se as amostras de treino devem ser embaralhadas.
        num_workers (int): Número de subprocessos para carregar os dados.

    Returns:
       retorna uma tupla.
    """
    trainset = datasets.FashionMNIST(
        root=data_root, train=True, download=True, transform=modifica_imagem()
    )
    testset = datasets.FashionMNIST(
        root=data_root, train=False, download=True, transform=modifica_imagem()
    )

    trainloader = DataLoader(
        trainset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True
    )
    testloader = DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    
    print(f'Train size: {len(trainset)}, Test size: {len(testset)}')

    return trainset, testset, trainloader, testloader