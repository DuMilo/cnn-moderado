import torch

def evaluate(loader, model, device, criterion):
    """
    Coloca o modelo em modo de avaliação, calcula a perda média e a acurácia
    sobre todos os lotes do DataLoader fornecido. Utiliza o critério de perda especificado.

    Args:
        loader (DataLoader): DataLoader com os dados para avaliação.
        model (torch.nn.Module): Modelo a ser avaliado.
        device (torch.device): Dispositivo onde o modelo e dados serão alocados (CPU ou GPU).
        criterion (torch.nn.modules.loss._Loss): Função de perda para cálculo da perda.

    Returns:
        retorna tupla contendo a perda média  e a acurácia média no conjunto de avaliação.
    """
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


