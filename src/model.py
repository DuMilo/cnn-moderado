# nesse arquivo, definimos o modelo CNN para classificação de imagens FashionMNIST.

import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, in_channels, h1_channels, h2_channels, linear_dim, dropout_rate, num_classes):

        # __init__ agora aceita parâmetros, tornando o modelo configurável.

        super().__init__()
        
        # O valor 7*7 é calculado a partir de uma imagem 28x28
        # (FashionMNIST) após 2 MaxPoolings 2x2. (28 -> 14 -> 7)
        self.flat_dim = h2_channels * 7 * 7 

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, h1_channels, 3, padding=1), 
            nn.ReLU(), 
            nn.MaxPool2d(2),
            
            nn.Conv2d(h1_channels, h2_channels, 3, padding=1), 
            nn.ReLU(), 
            nn.MaxPool2d(2),
            
            nn.Flatten(),
            nn.Linear(self.flat_dim, linear_dim), 
            nn.ReLU(), 
            nn.Dropout(dropout_rate),
            
            nn.Linear(linear_dim, num_classes)
        )
        
    def forward(self, x): 
        return self.net(x)