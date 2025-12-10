import os
import json
import argparse
import numpy as np
import matplotlib
import csv
import nibabel as nib
import torch
from sklearn.model_selection import KFold
from tqdm import tqdm
from glob import glob
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from .data_loader import LiverPatchDataset, augment_ct3d
from .inference import sliding_window_inference
from .model import UNet3D, DiceBCELoss
from .train import (
    seed_everything,
    worker_init_fn,
    dice_from_counts, 
    iou_from_counts,
    precision_from_counts,
    recall_from_counts,
    f1_from_pr,
    compute_surface_distances,
    save_metrics_plots
)


class EarlyStopping:
    def __init__(self, patience=5, min_delta=3e-3):
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
matplotlib.use("Agg")
EPS = 1e-6

def prepare_volume_list(data_dir):
    npy_images = sorted(glob(os.path.join(data_dir, "imagesTr_npy", "*.npy")))
    npy_masks = sorted(glob(os.path.join(data_dir, "labelsTr_npy", "*.npy")))
    if len(npy_images) > 0 and len(npy_images) == len(npy_masks):
        return npy_images, npy_masks

    img_patterns = [os.path.join(data_dir, "imagesTr", "*.nii"), os.path.join(data_dir, "imagesTr", "*.nii.gz")]
    msk_patterns = [os.path.join(data_dir, "labelsTr", "*.nii"), os.path.join(data_dir, "labelsTr", "*.nii.gz")]
    images = []
    masks = []
    for p in img_patterns:
        images.extend(sorted(glob(p)))
    for p in msk_patterns:
        masks.extend(sorted(glob(p)))
    if len(images) == 0:
        raise AssertionError(f"No images found in {os.path.join(data_dir, 'imagesTr_npy')} or {os.path.join(data_dir,'imagesTr')}")
    if len(images) != len(masks):
        raise AssertionError("Images / masks count mismatch")
    return images, masks

def load_volume(path):
    # returns numpy float32 array (dense), and optionally the nib header+affine if .nii for saving later
    if path.endswith(".npy"):
        arr = np.load(path).astype(np.float32)
        return arr, None
    else:
        nii = nib.load(path)
        arr = nii.get_fdata(dtype=np.float32)
        return arr, (nii.affine, nii.header)

def save_nifti(array, affine_header_tuple, out_path):
    affine, header = affine_header_tuple
    nii = nib.Nifti1Image(array.astype(np.float32), affine, header)
    nib.save(nii, out_path)

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

def prepare_volume_list_deprecated(data_dir):
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

def freeze_encoder(model):
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

def evaluate_case_list(model, device, image_paths, mask_paths, args, out_dir_metrics, prefix="val", save_pred_nii=False):
    stats = {
        'dices': [], 'ious': [], 'precisions': [], 'recalls': [], 'f1s': [], 'hd95s': [], 'assds': [],
        'cases': 0, 'skipped_empty_gt': 0, 'fp_on_empty_gt': 0,
        'per_case': []
    }
    model.eval()
    with torch.no_grad():
        for img_path, msk_path in zip(image_paths, mask_paths):
            arr, _ = load_volume(img_path)
            msk_arr, hdr = load_volume(msk_path)
            gt = (msk_arr > 0).astype(np.uint8)

            prob_map = sliding_window_inference(
                arr, model, device,
                patch_size=tuple(args.patch_size),
                stride_factor=args.sw_stride,
                batch_size=args.sw_batch
            )
            pred = (prob_map >= args.threshold).astype(np.uint8)

            if gt.sum() == 0:
                if pred.sum() > 0:
                    stats['fp_on_empty_gt'] += 1
                stats['skipped_empty_gt'] += 1
                case_row = {'case': os.path.basename(img_path), 'dice': None, 'iou': None, 'precision': None, 'recall': None, 'f1': None, 'hd95': None, 'assd': None, 'notes': 'empty_gt'}
                stats['per_case'].append(case_row)
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

            stats['dices'].append(float(dice_val))
            stats['ious'].append(float(iou_val))
            stats['precisions'].append(float(prec))
            stats['recalls'].append(float(rec))
            stats['f1s'].append(float(f1))
            stats['hd95s'].append(float(hd95_val) if not np.isnan(hd95_val) else np.nan)
            stats['assds'].append(float(assd_val) if not np.isnan(assd_val) else np.nan)
            stats['cases'] += 1

            case_row = {'case': os.path.basename(img_path), 'dice': float(dice_val), 'iou': float(iou_val),
                        'precision': float(prec), 'recall': float(rec), 'f1': float(f1),
                        'hd95': float(hd95_val) if not np.isnan(hd95_val) else None,
                        'assd': float(assd_val) if not np.isnan(assd_val) else None, 'notes': ''}
            stats['per_case'].append(case_row)

            if save_pred_nii and hdr is not None:
                # save predicted mask as nifti alongside metrics
                pred_save_dir = os.path.join(out_dir_metrics, "preds")
                os.makedirs(pred_save_dir, exist_ok=True)
                save_nifti(pred.astype(np.uint8), hdr, os.path.join(pred_save_dir, f"{os.path.basename(img_path).split('.')[0]}_pred.nii.gz"))

    # aggregate
    if len(stats['dices']) == 0:
        stats['mean_dice'] = 0.0
    else:
        stats['mean_dice'] = float(np.mean(stats['dices']))
    return stats

# core fine-tune per-fold (updated to produce test split and run test eval)
def fine_tune_one_fold(fold_id, train_idx, val_idx, image_paths, mask_paths, args):
    out_fold = os.path.join(args.output_dir, f"fold_{fold_id}")
    metrics_out = os.path.join(args.output_dir_metrics, f"fold_{fold_id}")
    os.makedirs(out_fold, exist_ok=True)
    os.makedirs(metrics_out, exist_ok=True)

    all_indices = np.arange(len(image_paths))
    remaining = np.setdiff1d(all_indices, val_idx)
    rng = np.random.RandomState(args.fold_seed + fold_id)
    if len(remaining) < 2:
        raise ValueError("Not enough volumes to pick 2 test cases from remaining after val split.")
    test_idx = rng.choice(remaining, size=2, replace=False)
    # Now final train indices are remaining - test_idx
    train_idx_final = np.setdiff1d(remaining, test_idx)
    assert len(train_idx_final) == len(all_indices) - len(val_idx) - len(test_idx)

    train_images = [image_paths[i] for i in train_idx_final]
    train_masks = [mask_paths[i] for i in train_idx_final]
    val_images = [image_paths[i] for i in val_idx]
    val_masks = [mask_paths[i] for i in val_idx]
    test_images = [image_paths[i] for i in test_idx]
    test_masks = [mask_paths[i] for i in test_idx]

    print(f"[fold {fold_id}] Sizes -> train: {len(train_images)} val: {len(val_images)} test: {len(test_images)}")

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

    start_epoch = 1
    best_val_dice = -1.0
    scheduler = None
    optimizer = None

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
        if args.pretrained is not None:
            load_weights_only(model, args.pretrained, device)
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

            # validation: full-volume sliding-window inference
            val_stats = {
                'epoch': epoch,
                'losses': epoch_losses,
                'dices': [], 'ious': [], 'precisions': [], 'recalls': [], 'f1s': [], 'hd95s': [], 'assds': [],
                'val_cases': 0, 'skipped_empty_gt': 0, 'fp_on_empty_gt': 0
            }

            model.eval()
            with torch.no_grad():
                for vi, (img_path, msk_path) in enumerate(zip(val_images, val_masks)):
                    arr, _ = load_volume(img_path)
                    msk_arr, _ = load_volume(msk_path)
                    gt = (msk_arr > 0).astype(np.uint8)

                    prob_map = sliding_window_inference(
                        arr, model, device,
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

    # test
    best_model_path = os.path.join(out_fold, f"best_fold{fold_id}.pth")
    if os.path.isfile(best_model_path):
        ck = torch.load(best_model_path, map_location=device)
        if "model_state" in ck:
            model.load_state_dict(ck["model_state"])
        else:
            model.load_state_dict(ck)
        print(f"[fold {fold_id}] Loaded best model for test evaluation: {best_model_path}")
    else:
        print(f"[fold {fold_id}] Warning: best model not found at {best_model_path}. Using last model weights for test eval.")

    test_stats = evaluate_case_list(model, device, test_images, test_masks, args, metrics_out, prefix="test", save_pred_nii=True)

    # save test metrics
    with open(os.path.join(metrics_out, "test_metrics.json"), "w") as f:
        json.dump(test_stats, f, indent=2)
    # csv
    csv_path = os.path.join(metrics_out, "test_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        fieldnames = ['case', 'dice', 'iou', 'precision', 'recall', 'f1', 'hd95', 'assd', 'notes']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for c in test_stats['per_case']:
            writer.writerow(c)

    print(f"[fold {fold_id}] Test mean Dice: {test_stats['mean_dice']:.4f} (cases: {test_stats['cases']}, skipped_empty: {test_stats['skipped_empty_gt']})")

    return {"fold": fold_id, "best_val_dice": best_val_dice, "model_path": os.path.join(out_fold, f"best_fold{fold_id}.pth"), "metrics_dir": metrics_out, "test_mean_dice": test_stats['mean_dice']}

# main loop 
def fine_tune_cv_main(args):
    seed_everything(args.seed)
    image_paths, mask_paths = prepare_volume_list(args.data_dir)
    n = len(image_paths)
    k = args.k_folds
    assert n >= k, f"Need n >= k_folds (n={n}, k={k})"

    indices = np.arange(n)
    kf = KFold(n_splits=k, shuffle=True, random_state=args.fold_seed)

    results = []
    for fold_id, (train_idx_unused, val_idx) in enumerate(kf.split(indices), start=1):
        train_idx, val_idx = train_idx_unused, val_idx 
        print(f"\n=== Fold {fold_id}/{k}: train-candidates {len(train_idx)} vols, val {len(val_idx)} vols ===")
        r = fine_tune_one_fold(fold_id, train_idx, val_idx, image_paths, mask_paths, args)
        results.append(r)

    summary_path = os.path.join(args.output_dir, "ft_cv_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    dices = [r['best_val_dice'] for r in results]
    test_dices = [r.get('test_mean_dice', 0.0) for r in results]
    print(f"\nPer-fold best Dice: {dices}")
    print(f"Mean best Dice: {float(np.mean(dices)):.4f}, Std: {float(np.std(dices)):.4f}")
    print(f"Per-fold test Dice: {test_dices}")
    print(f"Mean test Dice: {float(np.mean(test_dices)):.4f}, Std: {float(np.std(test_dices)):.4f}")
    print(f"Summary saved to {summary_path}")

def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True, help="Root dataset dir with imagesTr_npy and labelsTr_npy or imagesTr/labelsTr (nii)")
    p.add_argument("--pretrained", type=str, default=None, help="Path to pretrained weights (weights-only). If provided, loads weights but creates new optimizer.")
    p.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume (loads model+optimizer+scheduler and continues).")
    p.add_argument("--output_dir", type=str, default="ml/results_ft/run3/ft_results", help="Where to save fold outputs")
    p.add_argument("--output_dir_metrics", type=str, default="ml/results_ft/run3/ft_metrics", help="Where to save metrics/plots")
    p.add_argument("--k_folds", type=int, default=4)
    p.add_argument("--fold_seed", type=int, default=404)
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
