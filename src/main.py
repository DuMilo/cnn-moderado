#importa as bibliotecas necessarias
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms, datasets
from torch.utils.tensorboard import SummaryWriter
import yaml

#importando as classes 
from modelo.model import SimpleCNN
from metodos.data_utils import get_dataloaders
from metodos.train import treinamento

def main():
    # Carregar Configuração
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    cfg_model = config['model']
    cfg_train = config['training']
    cfg_data = config['data']
   
   # seleciona o dispositivo
    device = torch.device(cfg_train['device'])

    #instancia os datasets
    trainset, testset, trainloader, testloader = get_dataloaders(cfg_data['root'],cfg_data['batch_size'],
                                                                  True, cfg_data['num_workers'])
    #expoe o tamanho dos datasets
    print('Train size:', len(trainset), 'Test size:', len(testset))

    #instancia o modelo
    model = SimpleCNN(**cfg_model).to(device)
    print(model)

    criterion = nn.CrossEntropyLoss()

    if cfg_train['optimizer'] == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=cfg_train['learning_rate'])
    else:
        # Você pode adicionar 'SGD' aqui, como sugerido nos experimentos
        optimizer = optim.SGD(model.parameters(), lr=cfg_train['learning_rate'], momentum=0.9)

    treinamento(trainloader, testloader, model, device, optimizer, criterion)


if __name__ == "__main__":
    main()