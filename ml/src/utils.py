import os
import torch
import logging
import random
import numpy as np
import json
import matplotlib.pyplot as plt


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def worker_init_fn(worker_id):
    seed = torch.initial_seed() % (2**32 - 1)
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)

def save_checkpoint(epoch, model, optimizer, scheduler, best_val_dice, args, tag=""):
    ckpt_path = os.path.join(args.output_dir, f"{epoch}_epoch_{tag}")

    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "best_val_dice": best_val_dice,
        "args": vars(args),
    }, ckpt_path)

    logging.info(f"[checkpoint] Saved: {ckpt_path}")

def load_checkpoint(ckpt_path, model, optimizer=None, scheduler=None, device="cpu"):
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    model.load_state_dict(checkpoint["model_state"])
    
    if optimizer is not None and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    
    if scheduler is not None and "scheduler_state" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state"])
    
    epoch = checkpoint.get("epoch", -1) + 1
    best_val_dice = checkpoint.get("best_val_dice", 0.0)
    
    return epoch, best_val_dice, checkpoint.get("args", {})

def save_metrics_plots(all_epoch_stats, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    metrics = ['dice', 'iou', 'precision', 'recall', 'train_loss']
    mean_history = {m: [] for m in metrics}
    epochs = []

    def calc_mean(values):
        if len(values) > 0:
            return np.mean(values)
        return np.nan

    for epoch_stat in all_epoch_stats:
        epochs.append(epoch_stat['epoch'])
        mean_history['train_loss'].append(calc_mean(epoch_stat['losses']))
        mean_history['dice'].append(calc_mean(epoch_stat.get('dices', [])))
        mean_history['iou'].append(calc_mean(epoch_stat.get('ious', [])))
        mean_history['precision'].append(calc_mean(epoch_stat.get('precisions', [])))
        mean_history['recall'].append(calc_mean(epoch_stat.get('recalls', [])))

    # line plots
    plt.figure(figsize=(10,6))
    plt.plot(epochs, mean_history['train_loss'], marker='o', label='Train loss')
    if not all(np.isnan(mean_history['dice'])):
        plt.plot(epochs, mean_history['dice'], marker='x', label='Val mean Dice')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.title('Train loss & Val mean Dice per epoch')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'metrics_epoch_line.png'), dpi=150)
    plt.close()

    # boxplots per epoch for key metrics
    for metric_key, label in [('dices','Dice'), ('ious','IoU')]:
        fig, ax = plt.subplots(figsize=(12,6))
        data = [epoch_stat.get(metric_key, []) for epoch_stat in all_epoch_stats]
        data = [np.array(d, dtype=float) if len(d)>0 else np.array([np.nan]) for d in data]
        ax.boxplot(data, tick_labels=[f"E{e}" for e in epochs], showfliers=False)
        ax.set_title(f'{label} distribution per epoch (boxplot)')
        ax.set_xlabel('Epoch'); ax.set_ylabel(label)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'{metric_key}_boxplot.png'), dpi=150)
        plt.close()

    # json
    summary = []
    for epoch_stat in all_epoch_stats:
        row = {
            'epoch': epoch_stat['epoch'],
            'train_loss_mean': calc_mean(epoch_stat['losses']),
            'val_dice_mean': calc_mean(epoch_stat['dices']),
            'val_iou_mean': calc_mean(epoch_stat['ious']),
            'val_precision_mean': calc_mean(epoch_stat['precisions']),
            'val_recall_mean': calc_mean(epoch_stat['recalls']),
            'val_cases': int(epoch_stat.get('val_cases', 0)),
            'val_skipped_empty_gt': int(epoch_stat.get('skipped_empty_gt', 0)),
            'val_fp_on_empty_gt': int(epoch_stat.get('fp_on_empty_gt', 0)),
        }
        summary.append(row)

    with open(os.path.join(out_dir, 'metrics_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    logging.info(f"[metrics] Saved plots and summaries to {out_dir}")