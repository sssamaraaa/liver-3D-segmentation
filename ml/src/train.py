import os
import sys
import logging
import argparse
import numpy as np
import torch
import matplotlib
from sklearn.model_selection import KFold
from logging_conf import setup_logging
from glob import glob
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset import LiverPatchDataset, augment_ct3d, split_dataset
from model import UNet3D
from metrics import DiceBCELoss, dice, iou, precision, recall
from inference import sliding_window_inference
from utils import save_checkpoint, seed_everything, worker_init_fn, save_metrics_plots, load_checkpoint, load_model_from_checkpoint

# utils
matplotlib.use("Agg")

def setup_env(args):
    seed_everything(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Device: {device}")
    return device

def initialize_dataset(args):
    image_paths = sorted(glob(os.path.join(args.data_dir, "imagesTr_npy", "*.npy")))
    mask_paths = sorted(glob(os.path.join(args.data_dir, "labelsTr_npy", "*.npy")))
    assert len(image_paths) > 0, f"No preprocessed .npy files in {args.data_dir}/imagesTr_npy"
    assert len(image_paths) == len(mask_paths), "Mismatch between image and mask counts"
    return image_paths, mask_paths

def build_dataloader(args, train_images, train_masks):
    train_ds = LiverPatchDataset(
        train_images,
        train_masks,
        patch_size=tuple(args.patch_size),
        patches_per_volume=args.patches_per_volume,
        pos_ratio=args.pos_ratio,
        transform=(augment_ct3d if args.augment else None)
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers, 
        pin_memory=True,
        persistent_workers=True if args.num_workers>0 else False,
        worker_init_fn=worker_init_fn
    )

    return train_loader

def build_model(args, device):
    model = UNet3D(in_ch=args.in_ch, out_ch=args.out_ch, base_filters=args.base_filters).to(device)
    return model

def build_training_components(args, model):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs) if args.scheduler == 'cosine' else None
    scaler = GradScaler()
    criterion = DiceBCELoss(weight_bce=args.bce_weight)

    return optimizer, scheduler, scaler, criterion

def build_training_pipeline(args):
    device = setup_env(args)
    image_paths, mask_paths = initialize_dataset(args)
    train_images, train_masks, val_images, val_masks = split_dataset(image_paths,mask_paths,args.val_frac)
    logging.info(f"Train vols: {len(train_images)}, Val vols: {len(val_images)}")

    train_loader = build_dataloader(args, train_images, train_masks)
    model = build_model(args, device)
    optimizer, scheduler, scaler, criterion = build_training_components(args, model)

    return device, train_loader, val_images, val_masks, model, optimizer, scheduler, scaler, criterion

def build_finetune_pipeline(args, fold_idx=0):
    device = setup_env(args)
    image_paths, mask_paths = initialize_dataset(args)

    kf = KFold(n_splits=args.kfold, shuffle=True, random_state=args.seed)
    splits = list(kf.split(image_paths))
    train_idx, val_idx = splits[fold_idx]

    train_images = [image_paths[i] for i in train_idx]
    train_masks = [mask_paths[i] for i in train_idx]
    val_images = [image_paths[i] for i in val_idx]
    val_masks = [mask_paths[i] for i in val_idx]

    logging.info(f"Fold {fold_idx+1}. Train vols: {len(train_images)}, Val vols: {len(val_images)}")

    train_loader = build_dataloader(args, train_images, train_masks)

    model = load_model_from_checkpoint(args, device)
    if args.freeze_encoder:
        model.freeze_encoder()

    optimizer, scheduler, scaler, criterion = build_training_components(args, model)    

    for param_group in optimizer.param_groups:
        param_group['lr'] = args.finetune_lr

    return device, train_loader, val_images, val_masks, model, optimizer, scheduler, scaler, criterion

def train_one_epoch(args, train_loader, epoch, device, model, criterion, optimizer, scaler, accumulation_steps, epoch_losses, running_loss, global_step):
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}/{args.epochs}")

    for step, (img, msk) in pbar:
        img = img.float().to(device, non_blocking=True)
        msk = msk.float().to(device, non_blocking=True)

        with autocast(device_type=device.type):
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

    return running_loss, global_step

def validate(model, epoch, epoch_losses, val_images, val_masks, device, args):
    val_stats = {
        'epoch': epoch,
        'losses': epoch_losses,
        'dices': [], 'ious': [], 'precisions': [], 'recalls': [], 
        'val_cases': 0, 'skipped_empty_gt': 0, 'fp_on_empty_gt': 0
    }

    # eval
    model.eval()
    with torch.no_grad():
        for _, (img_path, msk_path) in enumerate(zip(val_images, val_masks)):
            # load whole volume for sliding-window inference
            img = np.load(img_path).astype(np.float32)
            msk = np.load(msk_path).astype(np.float32)
            gt = (msk > 0).astype(np.uint8) # ground truth

            prob_map = sliding_window_inference(
                    img, model, device,
                    patch_size=tuple(args.patch_size),
                    stride_factor=args.sw_stride,
                    batch_size=args.sw_batch
                )

            # skip empty GT volumes, but count them
            if gt.sum() == 0:
                pred = (prob_map >= args.threshold).astype(np.uint8)
                if pred.sum() > 0:
                    val_stats['fp_on_empty_gt'] += 1
                val_stats['skipped_empty_gt'] += 1
                continue

            pred = (prob_map >= args.threshold).astype(np.uint8)

            # basic counts
            inter = int(np.logical_and(pred, gt).sum()) # TP
            p_sum = int(pred.sum()) # TP + FP - predicted sum
            g_sum = int(gt.sum()) # TP + FN - ground truth sum

            dice_val = dice(inter, p_sum, g_sum)
            iou_val = iou(inter, p_sum, g_sum)
            prec = precision(inter, p_sum)
            rec = recall(inter, g_sum)

            # surface metrics
            val_stats['dices'].append(float(dice_val))
            val_stats['ious'].append(float(iou_val))
            val_stats['precisions'].append(float(prec))
            val_stats['recalls'].append(float(rec))
            val_stats['val_cases'] += 1

    return val_stats

def run_training(args, device, model, criterion, optimizer, scheduler, scaler, train_loader, val_images, val_masks, accumulation_steps=8):
    start_epoch = 1
    all_epoch_stats = []
    global_step = 0
    best_val_dice = 0.0

    if args.resume is not None:
        start_epoch, best_val_dice, args = load_checkpoint(args.resume, model, optimizer, scheduler, device)
        logging.info(f"Resume training from {start_epoch} with best Dice: {best_val_dice}")

    #train
    try:
        for epoch in range(start_epoch, args.epochs + 1):
            epoch_losses = []
            model.train()
            running_loss = 0.0
            optimizer.zero_grad()

            # train
            running_loss, global_step = train_one_epoch(args, train_loader, epoch, device, model, criterion, optimizer, scaler, accumulation_steps, epoch_losses, running_loss, global_step)
            
            # scheduler step (end of epoch)
            if scheduler is not None:
                scheduler.step()

            # validation
            val_stats = validate(model, epoch, epoch_losses, val_images, val_masks, device, args)
            # compute mean_dice
            if len(val_stats['dices']) == 0:
                mean_dice = 0.0
                logging.warning("Warning: no valid dice scores (all GT empty or skipped). Setting mean_dice = 0.")
            else:
                mean_dice = float(np.mean(val_stats['dices']))

            logging.info(f"Epoch {epoch} validation mean Dice: {mean_dice:.4f} (cases: {val_stats['val_cases']}, skipped empty: {val_stats['skipped_empty_gt']}, fp_on_empty_gt: {val_stats['fp_on_empty_gt']})")
            val_stats['mean_dice'] = mean_dice

            # save best model
            if mean_dice > best_val_dice:
                best_val_dice = mean_dice
                save_checkpoint(epoch, model, optimizer, scheduler, best_val_dice, args, tag="best_model.pth")
                logging.info(f"Saved best model with Dice={best_val_dice:.4f}")

            if epoch % args.checkpoint_every == 0:
                save_checkpoint(epoch, model, optimizer, scheduler, best_val_dice, args, tag="intermediate.pth")

            all_epoch_stats.append(val_stats)
            save_metrics_plots(all_epoch_stats, args.output_dir_metrics)

    except KeyboardInterrupt:
        logging.info("\n[interrupted] Training interrupted. Saving current checkpoint and metrics...")
        save_checkpoint(epoch, model, optimizer, scheduler, best_val_dice, args, tag="interrupted")
        save_metrics_plots(all_epoch_stats, args.output_dir_metrics)
        logging.info("Checkpoint saved. Exiting safely.")
        logging.info(f"Training finished. Best val dice: {best_val_dice}")
        sys.exit(0)

    # final save
    save_metrics_plots(all_epoch_stats, args.output_dir_metrics)
    logging.info(f"Training finished. Best val dice: {best_val_dice}")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", type=str, choices=['train', 'finetune'])
    p.add_argument("--freeze_encoder", action="store_true")
    p.add_argument("--kfold", type=int, default=4)
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="../results/run1/weights")
    p.add_argument("--output_dir_metrics", type=str, default="../results/run1/metrics")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--finetune_lr", type=float, default=1e-4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--patch_size", nargs=3, type=int, default=[80, 160, 160])
    p.add_argument("--in_ch", type=int, default=1)
    p.add_argument("--out_ch", type=int, default=1)
    p.add_argument("--base_filters", type=int, default=16)
    p.add_argument("--patches_per_volume", type=int, default=48)
    p.add_argument("--pos_ratio", type=float, default=0.65)
    p.add_argument("--augment", action="store_true")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--val_frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--bce_weight", type=float, default=0.5)
    p.add_argument("--accumulation_steps", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--sw_stride", type=float, default=0.5)
    p.add_argument("--sw_batch", type=int, default=4)
    p.add_argument("--checkpoint_every", type=int, default=10)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--scheduler", type=str, choices=['cosine','none'], default='cosine')
    p.add_argument("--resume", type=str, default=None, help="Path to checkpoint")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.output_dir_metrics, exist_ok=True)
    setup_logging()
    device, train_loader, val_images, val_masks, model, optimizer, scheduler, scaler, criterion = build_training_pipeline(args)

    if args.mode == 'train':
        run_training(args, device, model, criterion, optimizer, scheduler, scaler, train_loader, val_images, val_masks, accumulation_steps=args.accumulation_steps)

    if args.mode == 'finetune':
        for fold_idx in range(args.kfold):
            logging.info(f"Fold {fold_idx + 1}/{args.kfold}")
            device, train_loader, val_images, val_masks, model, optimizer, scheduler, scaler, criterion = build_finetune_pipeline(args, fold_idx)
            run_training(args, device, model, criterion, optimizer, scheduler, scaler, train_loader, val_images, val_masks, accumulation_steps=args.accumulation_steps)


