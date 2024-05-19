import os 

import pickle
import torch
from tqdm import tqdm

from utils.transforms import reverse_transform

def train_epoch(
        model,
        train_dataloader,
        optimizer,
        loss_fn, 
        metric_fn,
        device,
        mean,
        std
):
    model.train()
    epoch_loss = 0.
    epoch_metric = 0.

    for input_ids, targets, seq_len in tqdm(train_dataloader):
        input_ids = input_ids.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        preds = model(input_ids, seq_len).squeeze(dim=1)
        
        loss = loss_fn(preds, targets)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() / len(train_dataloader)
        epoch_metric += metric_fn(reverse_transform(preds.to('cpu').detach().numpy(), mean, std), reverse_transform(targets.to('cpu').detach().numpy(),mean, std)).item() / len(train_dataloader)
        break
    return epoch_loss, epoch_metric

def eval_epoch(
        model,
        val_dataloader,
        loss_fn, 
        metric_fn,
        device,
        mean,
        std
):
    model.eval()
    epoch_loss = 0.
    epoch_metric = 0.
    with torch.no_grad():
        for input_ids, targets, seq_len in tqdm(val_dataloader):
            input_ids = input_ids.to(device)
            targets = targets.to(device)

            preds = model(input_ids, seq_len).squeeze(dim=1)
            
            loss = loss_fn(preds, targets)

            epoch_loss += loss.item() / len(val_dataloader)
            epoch_metric += metric_fn(reverse_transform(preds.to('cpu').detach().numpy(), mean, std), reverse_transform(targets.to('cpu'), mean, std).detach().numpy()).item() / len(val_dataloader)
            break
        return epoch_loss, epoch_metric
    

def train(
        epochs, 
        model,
        train_dataloader,
        val_dataloader,
        optimizer,
        loss_fn, 
        metric_fn,
        device,
        save_dir,
        mean,
        std
):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(f"{save_dir}/backlogs"):    
        os.mkdir(f"{save_dir}/backlogs")
    train_epoch_losses, train_epoch_metrics = [], []
    val_epoch_losses, val_epoch_metrics = [], []
    best_metric = 100000
    model.to(device)
    for epoch_i in range(epochs):
        train_loss, train_metric = train_epoch(
            model,
            train_dataloader,
            optimizer,
            loss_fn,
            metric_fn,
            device,
            mean,
            std
        )
        val_loss, val_metric = eval_epoch(
            model,
            val_dataloader,
            loss_fn,
            metric_fn,
            device,
            mean,
            std
        )
        train_epoch_losses.append(train_loss)
        train_epoch_metrics.append(train_metric)
        val_epoch_losses.append(val_loss)
        val_epoch_metrics.append(val_metric)
        if val_epoch_metrics[-1] < best_metric:
            torch.save(model, f"{save_dir}/model_{epoch_i}.pt")
        with open(f"{save_dir}/backlogs/train_epoch_losses_{epoch_i}.pickle", "wb") as f:
            pickle.dump(
                (train_epoch_losses, train_epoch_metrics, val_epoch_losses, val_epoch_metrics), f
            )
        print(
            f" Epoch {epoch_i} \n Train Loss: {train_epoch_losses[-1]} Train Metric: {train_epoch_metrics[-1]} \n Val Loss: {val_epoch_losses[-1]} Val Metric: {val_epoch_metrics[-1]}"
        )
    return (train_epoch_losses, train_epoch_metrics, val_epoch_losses, val_epoch_metrics)
