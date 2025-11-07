# nesse arquivo, implementamos o loop de treinamento para o modelo CNN.
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import os
import time
from torch.utils.tensorboard import SummaryWriter

# importando de nossos próprios arquivos .py
from src.model import SimpleCNN
from src.data_utils import get_dataloaders

# --- funções auxiliares ---

def save_checkpoint(model, optimizer, epoch, val_acc, config, is_best):
    # salva o checkpoint, com 'best_model.pth' separado.
    run_name = config['run_name']
    checkpoint_dir = os.path.join(config['outputs']['checkpoint_dir'], run_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint_path = os.path.join(checkpoint_dir, "last_model.pth")
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc
    }
    torch.save(state, checkpoint_path)
    print(f"Checkpoint salvo em: {checkpoint_path}")

    if is_best:
        best_path = os.path.join(checkpoint_dir, "best_model.pth")
        torch.save(state, best_path)
        print(f"*** NOVO MELHOR MODELO salvo em: {best_path} ***")

def evaluate(model, loader, device, criterion):
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

# --- Função Principal de Treinamento ---

def main_train():
    # 1. Carregar Configuração
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # 2. Setup (Dispositivo, Seed, etc.)
    cfg_train = config['training']
    device = torch.device(cfg_train['device'] if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg_train['seed'])
    print(f'Rodando no dispositivo: {device}')

    # 3. Carregar Dados
    trainloader, testloader = get_dataloaders(
        data_root=config['data']['root'],
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers']
    )

    # 4. Inicializar Modelo (lendo do config!)
    model = SimpleCNN(**config['model']).to(device)

    # 5. Otimizador e Loss
    criterion = nn.CrossEntropyLoss()
    if cfg_train['optimizer'] == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=cfg_train['learning_rate'])
    else:
        # Você pode adicionar 'SGD' aqui, como sugerido nos experimentos
        optimizer = optim.SGD(model.parameters(), lr=cfg_train['learning_rate'], momentum=0.9)

    # 6. Setup de LOGS (TensorBoard)
    log_dir = os.path.join(config['outputs']['log_dir'], config['run_name'])
    writer = SummaryWriter(log_dir)
    print(f"Logs do TensorBoard em: {log_dir}")

    # --- 7. Loop de Treinamento ---
    print(f"Iniciando treino para {cfg_train['num_epochs']} épocas...")
    best_val_acc = 0.0
    
    for epoch in range(1, cfg_train['num_epochs'] + 1):
        start_time = time.time()
        model.train() # Coloca o modelo em modo de treino
        running_loss = 0.0
        
        for i, (imgs, labels) in enumerate(trainloader):
            imgs, labels = imgs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        # --- Fim da Época ---
        
        # Calcular métricas de treino
        avg_train_loss = running_loss / len(trainloader)
        
        # Calcular métricas de validação
        avg_val_loss, val_acc = evaluate(model, testloader, device, criterion)
        
        # Log no TensorBoard
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Loss/val', avg_val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        
        # Print no console
        elapsed = time.time() - start_time
        print(f'Epoch {epoch:02d}/{cfg_train["num_epochs"]}: '
              f'Loss: {avg_train_loss:.4f} | '
              f'Val.Loss: {avg_val_loss:.4f} | '
              f'Val.Acc: {val_acc:.4f} | '
              f'Tempo: {elapsed:.2f}s')

        # 8. Salvar Checkpoint
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            
        save_checkpoint(model, optimizer, epoch, val_acc, config, is_best)

    writer.close()
    print(f"Treinamento concluído. Melhor Val. Acc: {best_val_acc:.4f}")


if __name__ == "__main__":
    main_train()