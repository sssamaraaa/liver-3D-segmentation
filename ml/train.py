import os
import sys
import random
import json
import csv
import argparse
from glob import glob

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from torch.amp import autocast, GradScaler
from tqdm import tqdm
from torch.utils.data import DataLoader

from data_loader import LiverPatchDataset, augment_ct3d
from model import UNet3D, DiceBCELoss
from inference import sliding_window_inference

from scipy.ndimage import binary_erosion
from scipy.spatial import cKDTree
from scipy import ndimage as ndi

# utils

EPS = 1e-6  

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

def dice_from_counts(inter, p_sum, g_sum):
    denom = p_sum + g_sum + EPS
    return 2.0 * inter / denom

def iou_from_counts(inter, p_sum, g_sum):
    denom = p_sum + g_sum - inter + EPS
    return inter / denom

def precision_from_counts(inter, p_sum):
    return inter / (p_sum + EPS)

def recall_from_counts(inter, g_sum):
    return inter / (g_sum + EPS)

def f1_from_pr(prec, rec):
    return 2 * prec * rec / (prec + rec + EPS)

def mask_to_surface_coords(mask):
    if mask.sum() == 0:
        return np.zeros((0, 3), dtype=np.float32)
    struct = np.ones((3,3,3), dtype=bool)
    eroded = binary_erosion(mask, structure=struct, iterations=1)
    surface = mask.astype(bool) & (~eroded)
    coords = np.array(np.nonzero(surface)).T  
    return coords.astype(np.float32)

def compute_surface_distances(pred_mask, gt_mask):
    s_pred = mask_to_surface_coords(pred_mask)
    s_gt = mask_to_surface_coords(gt_mask)
    if s_pred.shape[0] == 0 or s_gt.shape[0] == 0:
        return np.nan, np.nan

    tree_pred = cKDTree(s_pred)
    tree_gt = cKDTree(s_gt)

    d_pred_to_gt, _ = tree_gt.query(s_pred, k=1)
    d_gt_to_pred, _ = tree_pred.query(s_gt, k=1)

    hd95_val = max(np.percentile(d_pred_to_gt, 95), np.percentile(d_gt_to_pred, 95))
    assd_val = (d_pred_to_gt.mean() + d_gt_to_pred.mean()) / 2.0

    return float(hd95_val), float(assd_val)

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
        ax.boxplot(data, labels=[f"E{e}" for e in epochs], showfliers=False)
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
    print(f"[checkpoint] Saved: {ckpt_path}")

def split_dataset(image_paths, mask_paths, val_frac=0.15):
    indices = np.arange(len(image_paths))
    np.random.shuffle(indices)
    val_count = max(1, int(len(indices) * val_frac))
    val_idx = indices[:val_count]
    train_idx = indices[val_count:]
    train_images = [image_paths[i] for i in train_idx]
    train_masks = [mask_paths[i] for i in train_idx]
    val_images = [image_paths[i] for i in val_idx]
    val_masks = [mask_paths[i] for i in val_idx]
    return train_images, train_masks, val_images, val_masks

def train(args):
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    image_paths = sorted(glob(os.path.join(args.data_dir, "imagesTr_npy", "*.npy")))
    mask_paths = sorted(glob(os.path.join(args.data_dir, "labelsTr_npy", "*.npy")))
    assert len(image_paths) > 0, f"No preprocessed .npy files in {args.data_dir}/imagesTr_npy"
    assert len(image_paths) == len(mask_paths), "Mismatch between image and mask counts"

    train_images, train_masks, val_images, val_masks = split_dataset(image_paths, mask_paths, args.val_frac)
    print(f"Train vols: {len(train_images)}, Val vols: {len(val_images)}")

    # datasets / dataloaders
    train_ds = LiverPatchDataset(
        train_images,
        train_masks,
        patch_size=tuple(args.patch_size),
        samples_per_volume=args.samples_per_volume,
        pos_ratio=args.pos_ratio,
        transform=(augment_ct3d if args.augment else None)
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=True if args.num_workers>0 else False,
        worker_init_fn=worker_init_fn
    )

    # model
    model = UNet3D(in_ch=1, out_ch=1, base_filters=args.base_filters).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs) if args.scheduler == 'cosine' else None
    scaler = GradScaler()
    criterion = DiceBCELoss(weight_bce=args.bce_weight)

    best_val_dice = 0.0
    global_step = 0
    accumulation_steps = args.accumulation_steps
    start_epoch = 1
    all_epoch_stats = []

    if args.resume and os.path.isfile(args.resume):
        print(f"[resume] Loading checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        if "optimizer_state" in checkpoint and checkpoint["optimizer_state"] is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        if "scheduler_state" in checkpoint and checkpoint["scheduler_state"] is not None and scheduler is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state"])
        best_val_dice = checkpoint.get("best_val_dice", 0.0)
        start_epoch = checkpoint.get("epoch", 0) + 1
        print(f"[resume] Resumed from epoch {start_epoch} (best dice={best_val_dice:.4f})")

    try:
        for epoch in range(start_epoch, args.epochs + 1):
            epoch_losses = []
            model.train()
            running_loss = 0.0
            pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}/{args.epochs}")
            optimizer.zero_grad()

            for step, (img, msk) in pbar:
                img = img.clone().detach().float().to(device, non_blocking=True)
                msk = msk.clone().detach().float().to(device, non_blocking=True)

                with autocast(device_type='cuda' if device.type=='cuda' else None):
                    logits = model(img)
                    loss = criterion(logits, msk)
                    loss = loss / accumulation_steps

                scaler.scale(loss).backward()

                if (step + 1) % accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    global_step += 1

                true_loss = loss.item() * accumulation_steps
                running_loss += true_loss
                epoch_losses.append(true_loss)
                pbar.set_postfix({
                    "loss": f"{running_loss / (step + 1):.4f}",
                    "lr": optimizer.param_groups[0]['lr']
                })

            # scheduler step (end of epoch)
            if scheduler is not None:
                scheduler.step()

            # validation
            val_stats = {
                'epoch': epoch,
                'losses': epoch_losses,
                'dices': [], 'ious': [], 'precisions': [], 'recalls': [], 'f1s': [], 'hd95s': [], 'assds': [],
                'val_cases': 0, 'skipped_empty_gt': 0, 'fp_on_empty_gt': 0
            }

            model.eval()
            dices = []
            with torch.no_grad():
                for vi, (img_path, msk_path) in enumerate(zip(val_images, val_masks)):
                    # load whole volume for sliding-window inference
                    img = np.load(img_path).astype(np.float32)
                    msk = np.load(msk_path).astype(np.float32)
                    gt = (msk > 0).astype(np.uint8)

                    # skip empty GT volumes, but count them
                    if gt.sum() == 0:
                        # compute whether model predicts anything (FP on empty GT)
                        prob_map = sliding_window_inference(
                            img, model, device,
                            patch_size=tuple(args.patch_size),
                            stride_factor=args.sw_stride,
                            batch_size=args.sw_batch
                        )
                        pred = (prob_map >= args.threshold).astype(np.uint8)
                        if pred.sum() > 0:
                            val_stats['fp_on_empty_gt'] += 1
                        val_stats['skipped_empty_gt'] += 1
                        continue

                    # normal case
                    prob_map = sliding_window_inference(
                        img, model, device,
                        patch_size=tuple(args.patch_size),
                        stride_factor=args.sw_stride,
                        batch_size=args.sw_batch
                    )
                    pred = (prob_map >= args.threshold).astype(np.uint8)
                    gt_bin = gt.astype(np.uint8)

                    # basic counts
                    inter = int(np.logical_and(pred, gt_bin).sum())
                    p_sum = int(pred.sum())
                    g_sum = int(gt_bin.sum())

                    dice_val = dice_from_counts(inter, p_sum, g_sum)
                    iou_val = iou_from_counts(inter, p_sum, g_sum)
                    prec = precision_from_counts(inter, p_sum)
                    rec = recall_from_counts(inter, g_sum)
                    f1 = f1_from_pr(prec, rec)

                    # surface metrics
                    hd95_val, assd_val = compute_surface_distances(pred, gt_bin)
                    val_stats['dices'].append(float(dice_val))
                    val_stats['ious'].append(float(iou_val))
                    val_stats['precisions'].append(float(prec))
                    val_stats['recalls'].append(float(rec))
                    val_stats['f1s'].append(float(f1))
                    val_stats['hd95s'].append(float(hd95_val) if not np.isnan(hd95_val) else np.nan)
                    val_stats['assds'].append(float(assd_val) if not np.isnan(assd_val) else np.nan)
                    val_stats['val_cases'] += 1

            # compute mean_dice
            if len(val_stats['dices']) == 0:
                mean_dice = 0.0
                print("Warning: no valid dice scores (all GT empty or skipped). Setting mean_dice = 0.")
            else:
                mean_dice = float(np.mean(val_stats['dices']))

            print(f"Epoch {epoch} validation mean Dice: {mean_dice:.4f} (cases: {val_stats['val_cases']}, skipped empty: {val_stats['skipped_empty_gt']}, fp_on_empty_gt: {val_stats['fp_on_empty_gt']})")
            val_stats['mean_dice'] = mean_dice

            # save best model
            if mean_dice > best_val_dice:
                best_val_dice = mean_dice
                torch.save({
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
                    "epoch": epoch,
                    "best_val_dice": best_val_dice
                }, os.path.join(args.output_dir, "best_model.pth"))
                print(f"Saved best model with Dice={best_val_dice:.4f}")

            if epoch % args.checkpoint_every == 0:
                save_checkpoint(epoch, model, optimizer, scheduler, best_val_dice, args, tag="epoch")

            all_epoch_stats.append(val_stats)
            save_metrics_plots(all_epoch_stats, args.output_dir_metrics)

    except KeyboardInterrupt:
        print("\n[interrupted] Training interrupted. Saving current checkpoint and metrics...")
        save_checkpoint(epoch, model, optimizer, scheduler, best_val_dice, args, tag="interrupted")
        save_metrics_plots(all_epoch_stats, args.output_dir_metrics)
        print("Checkpoint saved. Exiting safely.")
        print("Training finished. Best val dice:", best_val_dice)
        sys.exit(0)

    # final save
    save_metrics_plots(all_epoch_stats, args.output_dir_metrics)
    print("Training finished. Best val dice:", best_val_dice)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="./results")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--patch_size", nargs=3, type=int, default=[80, 160, 160])
    p.add_argument("--base_filters", type=int, default=16)
    p.add_argument("--samples_per_volume", type=int, default=48)
    p.add_argument("--pos_ratio", type=float, default=0.65)
    p.add_argument("--augment", action="store_true")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--val_frac", type=float, default=0.15)
    p.add_argument("--val_every", type=int, default=1)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--bce_weight", type=float, default=0.5)
    p.add_argument("--accumulation_steps", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--sw_stride", type=float, default=0.5)
    p.add_argument("--sw_batch", type=int, default=4)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--output_dir_metrics", type=str, default="ml/metrics")
    p.add_argument("--checkpoint_every", type=int, default=10)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--scheduler", type=str, choices=['cosine','none'], default='cosine')
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.output_dir_metrics, exist_ok=True)
    train(args)
