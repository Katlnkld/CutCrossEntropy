import torch
import torch.nn.functional as F
from tqdm import tqdm
from cut_cross_entropy import linear_cross_entropy
import time
import logging
logger = logging.getLogger(__name__)

def train_epoch(model, train_loader, val_loader, optimizer, loss_type, device, epoch=1, log_every=100):
    """
    Обучает одну эпоху модели SASRec

    :param model: SASRec модель
    :param train_loader: train DataLoader
    :param val_loader: validation DataLoader
    :param optimizer: оптимайзер torch.optim
    :param loss_type: тип функции потерь: 'CE' или 'CCE'
    :param device: 'cuda' или 'cpu'
    :param epoch: номер эпохи
    :param log_every: как часто печатать логи
    """
    results = {}
    # Обучение
    model.train()
    total_loss = 0
    total_batches = 0 

    # Профилирование
    if device == 'cuda':
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
    else:
        start_time = time.time()

    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}")
   
    for i, (_, seqs, targets) in pbar:
        # Forward
        seqs = seqs.to(device)       # (B, T)
        targets = targets.to(device) # (B,)

        seq_output = model(seqs)           # (B, T, D)
        final_output = seq_output[:, -1, :]

        if loss_type == 'CE':
            logits = torch.matmul(final_output, model.item_embedding.weight.T)  # (B, |V|)
            loss = F.cross_entropy(logits, targets)

        elif loss_type == 'CCE':
            final_output = final_output.half()
            classifier_weights = model.item_embedding.weight.half()

            loss = linear_cross_entropy(final_output, classifier_weights, targets, shift=True)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_batches += 1

        if (i + 1) % log_every == 0:
            avg_loss = total_loss / total_batches
            pbar.set_postfix(loss=f"{avg_loss:.4f}")

     # Завершаем замер обучения
    if device == 'cuda':
        end_event.record()
        torch.cuda.synchronize()
        train_time = start_event.elapsed_time(end_event) / 1000 # сек
        train_memory = torch.cuda.max_memory_allocated(device) / 1024**3 # GB
    else:
        train_time = time.time() - start_time
        train_memory = 0

    results['train_loss'] = round(total_loss / total_batches, 4)
    results['train_time_s'] = round(train_time, 2)
   

    # Валидация
    if val_loader is not None:
        model.eval()
        val_total_loss = 0
        val_batches = 0

        if device == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize()
            val_start = torch.cuda.Event(enable_timing=True)
            val_end = torch.cuda.Event(enable_timing=True)
            val_start.record()
        else:
            val_time_start = time.time()

        with torch.no_grad():
            for _, seqs, targets in val_loader:
                seqs = seqs.to(device)
                targets = targets.to(device)

                seq_output = model(seqs)
                final_output = seq_output[:, -1, :]

                if loss_type == 'CE':
                    logits = torch.matmul(final_output, model.item_embedding.weight.T)
                    loss = F.cross_entropy(logits, targets)
                elif loss_type == 'CCE':
                    final_output = final_output.half()
                    classifier_weights = model.item_embedding.weight.half()
                    loss = linear_cross_entropy(final_output, classifier_weights, targets, shift=True)

                val_total_loss += loss.item()
                val_batches += 1

        if device == 'cuda':
            val_end.record()
            torch.cuda.synchronize()
            val_time = val_start.elapsed_time(val_end) / 1000 # сек
            val_memory = torch.cuda.max_memory_allocated(device) / 1024**3 # GB
        else:
            val_time = time.time() - val_time_start
            val_memory = 0

        results['val_loss'] = round(val_total_loss / val_batches, 4)
        results['val_time_s'] = round(val_time, 2)

    else:
        results['val_loss'] = None
        results['val_time_s'] = None
        results['val_memory_GB'] = None

    return results

import pandas as pd
import torch
from torch.optim import Adam

def run_training(
    model_class,
    num_items,
    train_loader,
    val_loader,
    loss_type,
    device,
    n_epochs=5,
    hidden_dim=64,
    max_len=50,
    lr=0.001
):
    """
    Запускает обучение на несколько эпох, собирает метрики в датафрейм.

    :param model_class: модель SASRec
    :param num_items: число уникальных item'ов
    :param train_loader: train DataLoader
    :param val_loader: validation DataLoader
    :param loss_type: тип функции потерь ('CE' или 'CCE')
    :param device: 'cuda' или 'cpu'
    :param n_epochs: кол-во эпох
    :param hidden_dim: размер скрытого слоя
    :param max_len: максимальная длина последовательности
    :param lr: learning rate

    :return: model, pd.DataFrame с метриками по эпохам, максимальная затраченная память на обучении
    """
    max_mem = 0
    print(device)
    
    torch.cuda.reset_peak_memory_stats(device)

    model = model_class(num_items=num_items, hidden_dim=hidden_dim, max_len=max_len).to(device)
    optimizer = Adam(model.parameters(), lr=lr)

    logger.info(f"Loss = {loss_type}")
    all_metrics = []

    for epoch in range(1, n_epochs + 1):
        metrics = train_epoch(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            loss_type=loss_type,
            device=device,
            epoch=epoch
        )
        metrics['epoch'] = epoch
        all_metrics.append(metrics)
        print(metrics)
    
    # Считаем пиковую память в конце
    
    max_mem = torch.cuda.max_memory_allocated(device) / 1024**3
    logger.info(f"Peak GPU memory = {round(max_mem, 4)} GB")

    df_metrics = pd.DataFrame(all_metrics)
    return model, df_metrics, max_mem


    
