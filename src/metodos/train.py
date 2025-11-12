# nesse arquivo, implementamos o loop de treinamento para o modelo CNN.
import torch
import yaml
import os
import time
from torch.utils.tensorboard import SummaryWriter
from metodos.evaluation import evaluate

# --- funções auxiliares ---
def save_checkpoint(model, optimizer, epoch, val_acc, config, is_best):
    """
    Salva o estado atual do modelo e do otimizador em um arquivo de checkpoint.
    Args:
        model (torch.nn.Module): Modelo cujo estado será salvo.
        optimizer (torch.optim.Optimizer): Otimizador cujo estado será salvo.
        epoch (int): Época atual do treinamento.
        val_acc (float): Acurácia de validação obtida na época atual.
        config (dict): Configurações do experimento, incluindo caminhos para salvar checkpoints.
        is_best (bool): Indica se o modelo atual é o melhor até o momento.

    Returns:
        None
    """
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


# --- Função Principal de Treinamento ---
def treinamento(trainloader, testloader, model, device, optimizer, criterion):
    """
    Executa o loop de treinamento para um modelo CNN

    Essa função realiza o treinamento do modelo por um número configurado de épocas, 
    calcula métricas de perda e acurácia no conjunto de validação, registra logs no TensorBoard
    e salva checkpoints dos modelos (incluindo o melhor modelo).

    Args:
        trainloader (DataLoader): DataLoader para os dados de treino.
        testloader (DataLoader): DataLoader para os dados de validação/teste.
        model (torch.nn.Module): Modelo CNN a ser treinado.
        device (torch.device): Dispositivo para computação (CPU ou GPU).
        optimizer (torch.optim.Optimizer): Otimizador para ajustar os pesos do modelo.
        criterion (torch.nn.modules.loss._Loss): Função de perda usada no treinamento.

    Returns:
        None
    """
    # Carregar Configuração
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Setup (Dispositivo, Seed, etc.)
    cfg_train = config['training']
    torch.manual_seed(cfg_train['seed'])
    print(f'Rodando no dispositivo: {device}')

    # Setup de LOGS (TensorBoard)
    log_dir = os.path.join(config['outputs']['log_dir'], config['run_name'])
    writer = SummaryWriter(log_dir)
    print(f"Logs do TensorBoard em: {log_dir}")

    # Loop de Treinamento
    print(f"Iniciando treino para {cfg_train['num_epochs']} epocas...")
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
        # Calcular métricas de treino
        avg_train_loss = running_loss / len(trainloader)
        # Calcular métricas de validação
        avg_val_loss, val_acc = evaluate(testloader, model, device, criterion)
        
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

