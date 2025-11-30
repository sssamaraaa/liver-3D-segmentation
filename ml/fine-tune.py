import os
import json
import argparse
import random
from glob import glob

import numpy as np
from sklearn.model_selection import KFold
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import csv

import torch
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

# user modules (must be importable)
from data_loader import LiverPatchDataset, augment_ct3d
from model import UNet3D, DiceBCELoss
from inference import sliding_window_inference

class EarlyStopping:
    def __init__(self, patience=5, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best = None
        self.should_stop = False

    def step(self, value):
        if self.best is None or value > self.best + self.min_delta:
            self.best = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

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
    return 2.0 * inter / (p_sum + g_sum + EPS)

def iou_from_counts(inter, p_sum, g_sum):
    return inter / (p_sum + p_sum - inter + EPS) if False else inter / (p_sum + g_sum - inter + EPS)

def precision_from_counts(inter, p_sum):
    return inter / (p_sum + EPS)

def recall_from_counts(inter, g_sum):
    return inter / (g_sum + EPS)

def f1_from_pr(prec, rec):
    return 2 * prec * rec / (prec + rec + EPS)

# surface distances
from scipy.ndimage import binary_erosion
from scipy.spatial import cKDTree

def mask_to_surface_coords(mask):
    if mask.sum() == 0:
        return np.zeros((0,3), dtype=np.float32)
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

# metrics plotting & saving
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

    # line plot
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
    if summary:
        csv_path = os.path.join(out_dir, 'metrics_summary.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
            writer.writeheader()
            for row in summary:
                writer.writerow(row)

def save_checkpoint(epoch, model, optimizer, scheduler, best_val_dice, args, tag="epoch", out_dir="."):
    ckpt_path = os.path.join(out_dir, f"{tag}_{epoch}.pth")
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "best_val_dice": best_val_dice,
        "args": vars(args),
    }, ckpt_path)
    print(f"[checkpoint] Saved: {ckpt_path}")

# dataset helpers 
def prepare_volume_list(data_dir):
    images = sorted(glob(os.path.join(data_dir, "imagesTr_npy", "*.npy")))
    masks = sorted(glob(os.path.join(data_dir, "labelsTr_npy", "*.npy")))
    assert len(images) > 0, f"No .npy found in {os.path.join(data_dir, 'imagesTr_npy')}"
    assert len(images) == len(masks), "Images / masks count mismatch"
    return images, masks

# weights loading
def load_weights_only(model, path, device):
    if path is None:
        return
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Pretrained file not found: {path}")
    ckpt = torch.load(path, map_location=device)
    # support dictionary checkpoint saved by the training script
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        state = ckpt["model_state"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt
    model.load_state_dict(state)
    print(f"[init] Loaded pretrained weights from {path}")

def load_checkpoint_resume(model, optimizer, scheduler, path, device):
    # load full checkpoint including optimizer/scheduler and epoch info
    if path is None:
        return None
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Resume checkpoint not found: {path}")
    ckpt = torch.load(path, map_location=device)
    # try to load model state
    if "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
    elif "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"])
    else:
        # assume raw state_dict
        model.load_state_dict(ckpt)
    # load optimizer/scheduler if present
    if optimizer is not None and "optimizer_state" in ckpt and ckpt["optimizer_state"] is not None:
        try:
            optimizer.load_state_dict(ckpt["optimizer_state"])
            print("[resume] Optimizer state loaded from checkpoint")
        except Exception as e:
            print("[resume] Warning: failed to load optimizer state:", e)
    if scheduler is not None and "scheduler_state" in ckpt and ckpt["scheduler_state"] is not None:
        try:
            scheduler.load_state_dict(ckpt["scheduler_state"])
            print("[resume] Scheduler state loaded from checkpoint")
        except Exception as e:
            print("[resume] Warning: failed to load scheduler state:", e)
    start_epoch = ckpt.get("epoch", 0) + 1
    best_val = ckpt.get("best_val_dice", -1.0)
    return {"start_epoch": start_epoch, "best_val": best_val}

# freezing helper 
def freeze_encoder(model):
    # freeze inc and down* blocks
    for name, module in [("inc", getattr(model, "inc", None)),
                         ("down1", getattr(model, "down1", None)),
                         ("down2", getattr(model, "down2", None)),
                         ("down3", getattr(model, "down3", None)),
                         ("down4", getattr(model, "down4", None))]:
        if module is None:
            continue
        for p in module.parameters():
            p.requires_grad = False

def unfreeze_encoder(model):
    for name, module in [("inc", getattr(model, "inc", None)),
                         ("down1", getattr(model, "down1", None)),
                         ("down2", getattr(model, "down2", None)),
                         ("down3", getattr(model, "down3", None)),
                         ("down4", getattr(model, "down4", None))]:
        if module is None:
            continue
        for p in module.parameters():
            p.requires_grad = True

# core fine-tune per-fold 
def fine_tune_one_fold(fold_id, train_idx, val_idx, image_paths, mask_paths, args):
    out_fold = os.path.join(args.output_dir, f"fold_{fold_id}")
    metrics_out = os.path.join(args.output_dir_metrics, f"fold_{fold_id}")
    os.makedirs(out_fold, exist_ok=True)
    os.makedirs(metrics_out, exist_ok=True)

    train_images = [image_paths[i] for i in train_idx]
    train_masks = [mask_paths[i] for i in train_idx]
    val_images = [image_paths[i] for i in val_idx]
    val_masks = [mask_paths[i] for i in val_idx]

    bs = args.batch_size
    accum_steps = args.accumulation_steps

    train_ds = LiverPatchDataset(
        train_images, train_masks,
        patch_size=tuple(args.patch_size),
        samples_per_volume=args.samples_per_volume,
        pos_ratio=args.pos_ratio,
        transform=(augment_ct3d if args.augment else None)
    )
    train_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=True if args.num_workers>0 else False,
        worker_init_fn=worker_init_fn
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet3D(in_ch=1, out_ch=1, base_filters=args.base_filters).to(device)

    # Option A: resume full checkpoint (weights+opt) if requested
    start_epoch = 1
    best_val_dice = -1.0
    scheduler = None
    optimizer = None

    # If user requested resume, create optimizer & scheduler first so we can load states
    if args.resume is not None:
        # create optimizer with default ft_lr for resume case (will be overwritten by loaded state if present)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.ft_lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.ft_epochs) if args.scheduler == 'cosine' else None
        resume_info = load_checkpoint_resume(model, optimizer, scheduler, args.resume, device)
        if resume_info is not None:
            start_epoch = resume_info.get("start_epoch", 1)
            best_val_dice = resume_info.get("best_val", -1.0)
            print(f"[fold {fold_id}] Resuming from epoch {start_epoch} (best dice {best_val_dice})")
    else:
        # Load weights only (recommended fine-tuning behavior)
        if args.pretrained is not None:
            load_weights_only(model, args.pretrained, device)
        # freeze encoder if requested BEFORE building optimizer
        if args.freeze_encoder:
            freeze_encoder(model)
            print(f"[fold {fold_id}] Encoder frozen (trainable params will be decoder/out only).")

        # build optimizer after freezing to include only trainable params
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = torch.optim.AdamW(trainable_params, lr=args.ft_lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.ft_epochs) if args.scheduler == 'cosine' else None

    scaler = GradScaler()
    criterion = DiceBCELoss(weight_bce=args.bce_weight)
    early = EarlyStopping(patience=5, min_delta=1e-4)
    # training loop
    all_epoch_stats = []
    try:
        for epoch in range(start_epoch, args.ft_epochs + 1):
            # optionally unfreeze at given epoch
            if args.unfreeze_epoch is not None and args.unfreeze_epoch > 0 and epoch == args.unfreeze_epoch:
                unfreeze_encoder(model)
                # rebuild optimizer to include encoder params
                optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.ft_lr, weight_decay=args.weight_decay)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(args.ft_epochs - epoch + 1)) if args.scheduler == 'cosine' else None
                print(f"[fold {fold_id}] Unfroze encoder at epoch {epoch}. Recreated optimizer.")

            seed_everything(args.seed + epoch)
            model.train()
            epoch_losses = []
            running_loss = 0.0
            pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Fold {fold_id} Epoch {epoch}/{args.ft_epochs}")
            optimizer.zero_grad()
            for step, (img, msk) in pbar:
                img = img.clone().detach().float().to(device, non_blocking=True)
                msk = msk.clone().detach().float().to(device, non_blocking=True)

                with autocast(device_type='cuda' if device.type=='cuda' else None):
                    logits = model(img)
                    loss = criterion(logits, msk)
                    loss = loss / accum_steps

                scaler.scale(loss).backward()

                if (step + 1) % accum_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), max_norm=args.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                true_loss = float(loss.item() * accum_steps)
                running_loss += true_loss
                epoch_losses.append(true_loss)
                pbar.set_postfix({"loss": f"{running_loss / (step + 1):.4f}", "lr": optimizer.param_groups[0]['lr']})

            if scheduler is not None:
                scheduler.step()

            # VALIDATION: full-volume sliding-window inference
            val_stats = {
                'epoch': epoch,
                'losses': epoch_losses,
                'dices': [], 'ious': [], 'precisions': [], 'recalls': [], 'f1s': [], 'hd95s': [], 'assds': [],
                'val_cases': 0, 'skipped_empty_gt': 0, 'fp_on_empty_gt': 0
            }

            model.eval()
            with torch.no_grad():
                for vi, (img_path, msk_path) in enumerate(zip(val_images, val_masks)):
                    img = np.load(img_path).astype(np.float32)
                    msk = np.load(msk_path).astype(np.float32)
                    gt = (msk > 0).astype(np.uint8)

                    prob_map = sliding_window_inference(
                        img, model, device,
                        patch_size=tuple(args.patch_size),
                        stride_factor=args.sw_stride,
                        batch_size=args.sw_batch
                    )
                    pred = (prob_map >= args.threshold).astype(np.uint8)

                    # if gt empty, record info and skip metric
                    if gt.sum() == 0:
                        if pred.sum() > 0:
                            val_stats['fp_on_empty_gt'] += 1
                        val_stats['skipped_empty_gt'] += 1
                        continue

                    inter = int(np.logical_and(pred, gt).sum())
                    p_sum = int(pred.sum())
                    g_sum = int(gt.sum())

                    dice_val = dice_from_counts(inter, p_sum, g_sum)
                    iou_val = iou_from_counts(inter, p_sum, g_sum)
                    prec = precision_from_counts(inter, p_sum)
                    rec = recall_from_counts(inter, g_sum)
                    f1 = f1_from_pr(prec, rec)
                    hd95_val, assd_val = compute_surface_distances(pred, gt)

                    val_stats['dices'].append(float(dice_val))
                    val_stats['ious'].append(float(iou_val))
                    val_stats['precisions'].append(float(prec))
                    val_stats['recalls'].append(float(rec))
                    val_stats['f1s'].append(float(f1))
                    val_stats['hd95s'].append(float(hd95_val) if not np.isnan(hd95_val) else np.nan)
                    val_stats['assds'].append(float(assd_val) if not np.isnan(assd_val) else np.nan)
                    val_stats['val_cases'] += 1

            if len(val_stats['dices']) == 0:
                mean_dice = 0.0
                print(f"[fold {fold_id}] Warning: no valid dice scores (all GT empty?). Setting mean_dice=0")
            else:
                mean_dice = float(np.mean(val_stats['dices']))
            val_stats['mean_dice'] = mean_dice
            all_epoch_stats.append(val_stats)

            print(f"[fold {fold_id}] Epoch {epoch} validation mean Dice: {mean_dice:.4f} (cases: {val_stats['val_cases']}, skipped empty: {val_stats['skipped_empty_gt']}, fp_on_empty_gt: {val_stats['fp_on_empty_gt']})")

            # save best
            if mean_dice > best_val_dice:
                best_val_dice = mean_dice
                best_path = os.path.join(out_fold, f"best_fold{fold_id}.pth")
                torch.save({
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
                    "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
                    "epoch": epoch,
                    "best_val_dice": best_val_dice
                }, best_path)
                print(f"[fold {fold_id}] Saved best model: {best_path} (dice {best_val_dice:.4f})")
                early.step(mean_dice)

            early.step(mean_dice)
            if early.should_stop:
                print(f"[fold {fold_id}] Early stopping triggered at epoch {epoch}")
                break

            if epoch % args.checkpoint_every == 0:
                save_checkpoint(epoch, model, optimizer, scheduler, best_val_dice, args, tag=f"fold{fold_id}_epoch", out_dir=out_fold)

            save_metrics_plots(all_epoch_stats, metrics_out)

    except KeyboardInterrupt:
        print(f"\n[fold {fold_id}] Interrupted. Saving checkpoint and metrics.")
        save_checkpoint(epoch, model, optimizer, scheduler, best_val_dice, args, tag=f"fold{fold_id}_interrupted", out_dir=out_fold)
        save_metrics_plots(all_epoch_stats, metrics_out)
        raise

    save_metrics_plots(all_epoch_stats, metrics_out)
    return {"fold": fold_id, "best_val_dice": best_val_dice, "model_path": os.path.join(out_fold, f"best_fold{fold_id}.pth"), "metrics_dir": metrics_out}

# main CV loop 
def fine_tune_cv_main(args):
    seed_everything(args.seed)
    image_paths, mask_paths = prepare_volume_list(args.data_dir)
    n = len(image_paths)
    k = args.k_folds
    assert n >= k, f"Need n >= k_folds (n={n}, k={k})"

    indices = np.arange(n)
    kf = KFold(n_splits=k, shuffle=True, random_state=args.fold_seed)

    results = []
    for fold_id, (train_idx, val_idx) in enumerate(kf.split(indices), start=1):
        print(f"\n=== Fold {fold_id}/{k}: train {len(train_idx)} vols, val {len(val_idx)} vols ===")
        r = fine_tune_one_fold(fold_id, train_idx, val_idx, image_paths, mask_paths, args)
        results.append(r)

    summary_path = os.path.join(args.output_dir, "ft_cv_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    dices = [r['best_val_dice'] for r in results]
    print(f"\nPer-fold best Dice: {dices}")
    print(f"Mean best Dice: {float(np.mean(dices)):.4f}, Std: {float(np.std(dices)):.4f}")
    print(f"Summary saved to {summary_path}")

def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True, help="Root dataset dir with imagesTr_npy and labelsTr_npy")
    p.add_argument("--pretrained", type=str, default=None, help="Path to pretrained weights (weights-only). If provided, loads weights but creates new optimizer.")
    p.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume (loads model+optimizer+scheduler and continues).")
    p.add_argument("--output_dir", type=str, default="ml/ft_results", help="Where to save fold outputs")
    p.add_argument("--output_dir_metrics", type=str, default="ml/ft_metrics", help="Where to save metrics/plots")
    p.add_argument("--k_folds", type=int, default=4)
    p.add_argument("--fold_seed", type=int, default=42)
    p.add_argument("--ft_epochs", type=int, default=30)
    p.add_argument("--ft_lr", type=float, default=1e-4, help="LR for fine-tuning (used to create new optimizer unless --resume)")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--accumulation_steps", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--patch_size", nargs=3, type=int, default=[80,160,160])
    p.add_argument("--samples_per_volume", type=int, default=48)
    p.add_argument("--pos_ratio", type=float, default=0.65)
    p.add_argument("--augment", action="store_true")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--bce_weight", type=float, default=0.5)
    p.add_argument("--checkpoint_every", type=int, default=5)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--sw_stride", type=float, default=0.5)
    p.add_argument("--sw_batch", type=int, default=4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--base_filters", type=int, default=16)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--scheduler", type=str, choices=['cosine','none'], default='cosine')
    p.add_argument("--freeze_encoder", action="store_true", help="Freeze encoder (inc + down blocks) at start of fine-tuning")
    p.add_argument("--unfreeze_epoch", type=int, default=0, help="If >0, unfreeze encoder at this epoch number (global epoch counting within ft_epochs)")
    return p

if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    args.ft_lr = args.ft_lr
    args.ft_epochs = args.ft_epochs

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.output_dir_metrics, exist_ok=True)
    fine_tune_cv_main(args)
