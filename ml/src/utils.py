import os
import torch
import time
import logging
import random
import numpy as np
import json
import csv
import matplotlib.pyplot as plt
from .model import UNet3D


logger = logging.getLogger(__name__)

def save_checkpoint(epoch, model, optimizer, scheduler, best_val_dice, args, tag="epoch"):
    ckpt_path = os.path.join(args.output_dir, f"{tag}_{epoch}.pth")

    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "best_val_dice": best_val_dice,
        "args": vars(args),
    }, ckpt_path)

    logging.info(f"[checkpoint] Saved: {ckpt_path}")

def load_checkpoint(model, ckpt_path, device):
    start = time.time()
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    logging.info(f"Checkpoint loaded in {time.time() - start:.2f}s")
    
    if isinstance(ck, dict):
        state_dict = ck.get("model_state", ck.get("state_dict", ck.get("model", ck)))
    else:
        state_dict = ck
    
    new_state_dict = {}
    for k, v in state_dict.items():
        # OP : handle DataParallel model prefixes for compatibility
        if k.startswith('module.'):  
            k = k[7:]
        new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict, strict=False)
    return model

def load_model(ckpt_path, device, base_filters, use_fp16_on_gpu):
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    device_t = torch.device(device)

    model = UNet3D(in_ch=1, out_ch=1, base_filters=base_filters)
    model = load_checkpoint(model, ckpt_path, device_t)
    model.to(device_t)

    if device_t.type == "cuda" and use_fp16_on_gpu:
        try:
            model.half()
        except Exception:
            pass

    model.eval()

    info = {
        "device": str(device_t),
        "fp16": (device_t.type == "cuda" and use_fp16_on_gpu),
        "ckpt": os.path.basename(ckpt_path),
        "base_filters": base_filters,
    }
    return model, info

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

def save_metrics_plots(all_epoch_stats, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    metrics = ['dice', 'iou', 'precision', 'recall', 'f1', 'hd95', 'assd', 'train_loss']
    mean_history = {m: [] for m in metrics}
    epochs = []
    for s in all_epoch_stats:
        epochs.append(s['epoch'])
        mean_history['train_loss'].append(np.mean(s['losses']) if len(s['losses'])>0 else np.nan)
        for m, key in [('dice','dices'), ('iou','ious'), ('precision','precisions'),
                       ('recall','recalls'), ('f1','f1s'), ('hd95','hd95s'), ('assd','assds')]:
            arr = np.array(s.get(key, []), dtype=float)
            if arr.size == 0:
                mean_history[m].append(np.nan)
            else:
                mean_history[m].append(np.nanmean(arr))

    # line plots
    plt.figure(figsize=(10,6))
    plt.plot(epochs, mean_history['train_loss'], marker='o', label='Train loss')
    if not all(np.isnan(mean_history['dice'])):
        plt.plot(epochs, mean_history['dice'], marker='x', label='Val mean Dice')
    plt.xlabel('Epoch'); plt.ylabel('Value'); plt.grid(True, linestyle=':', alpha=0.6)
    plt.title('Train loss & Val mean Dice per epoch')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'metrics_epoch_line.png'), dpi=150)
    plt.close()

    # boxplots per epoch for key metrics
    for metric_key, label in [('dices','Dice'), ('ious','IoU'), ('f1s','F1')]:
        fig, ax = plt.subplots(figsize=(12,6))
        data = [s.get(metric_key, []) for s in all_epoch_stats]
        data = [np.array(d, dtype=float) if len(d)>0 else np.array([np.nan]) for d in data]
        ax.boxplot(data, tick_labels=[f"E{e}" for e in epochs], showfliers=False)
        ax.set_title(f'{label} distribution per epoch (boxplot)')
        ax.set_xlabel('Epoch'); ax.set_ylabel(label)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'{metric_key}_boxplot.png'), dpi=150)
        plt.close()

    # json / csv summary
    summary = []
    for s in all_epoch_stats:
        row = {
            'epoch': s['epoch'],
            'train_loss_mean': float(np.mean(s['losses'])) if len(s['losses'])>0 else None,
            'val_dice_mean': float(np.nanmean(s.get('dices',[]))) if len(s.get('dices',[]))>0 else None,
            'val_iou_mean': float(np.nanmean(s.get('ious',[]))) if len(s.get('ious',[]))>0 else None,
            'val_precision_mean': float(np.nanmean(s.get('precisions',[]))) if len(s.get('precisions',[]))>0 else None,
            'val_recall_mean': float(np.nanmean(s.get('recalls',[]))) if len(s.get('recalls',[]))>0 else None,
            'val_f1_mean': float(np.nanmean(s.get('f1s',[]))) if len(s.get('f1s',[]))>0 else None,
            'val_hd95_mean': float(np.nanmean([v for v in s.get('hd95s',[]) if not np.isnan(v)])) if len(s.get('hd95s',[]))>0 else None,
            'val_assd_mean': float(np.nanmean([v for v in s.get('assds',[]) if not np.isnan(v)])) if len(s.get('assds',[]))>0 else None,
            'val_cases': int(s.get('val_cases', 0)),
            'val_skipped_empty_gt': int(s.get('skipped_empty_gt', 0)),
            'val_fp_on_empty_gt': int(s.get('fp_on_empty_gt', 0)),
        }
        summary.append(row)

    with open(os.path.join(out_dir, 'metrics_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    # CSV
    csv_path = os.path.join(out_dir, 'metrics_summary.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()) if summary else [])
        if summary:
            writer.writeheader()
            for row in summary:
                writer.writerow(row)

    print(f"[metrics] Saved plots and summaries to {out_dir}")