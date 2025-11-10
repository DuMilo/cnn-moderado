import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms, datasets
from torch.utils.tensorboard import SummaryWriter

def evaluate(loader, model, device, criterion):
    """Função de avaliação (usada para validação)."""
    model.eval() # Coloca o modelo em modo de avaliação (desliga dropout, etc.)
    total_loss = 0.0
    total, correct = 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out = model(imgs)
            loss = criterion(out, labels)
            total_loss += loss.item() * labels.size(0)
            
            _, preds = torch.max(out, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    
    avg_loss = total_loss / total
    avg_acc = correct / total
    return avg_loss, avg_acc


