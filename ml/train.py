import os
import sys
import random
import numpy as np
import argparse
from glob import glob
import torch
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from data_loader import LiverPatchDataset, load_nifti, resample_to_spacing, intensity_clip_normalize, augment_ct3d
from torch.utils.data import DataLoader
from model import UNet3D, DiceBCELoss
from inference import sliding_window_inference


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def save_checkpoint(epoch, model, optimizer, scheduler, best_val_dice, args, tag="epoch"):
    ckpt_path = os.path.join(args.output_dir, f"{tag}_{epoch}.pth")
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "best_val_dice": best_val_dice,
        "args": vars(args),
    }, ckpt_path)
    print(f"[checkpoint] Saved: {ckpt_path}")

def train(args):
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    image_paths = sorted(glob(os.path.join(args.data_dir, "imagesTr", "*.nii*")))
    mask_paths = sorted(glob(os.path.join(args.data_dir, "labelsTr", "*.nii*")))
    assert len(image_paths) > 0, "No .nii files in the imagesTr directory"
    assert len(image_paths) == len(mask_paths), "Inconsistency between image names and masks"

    # split train/val
    indices = list(range(len(image_paths)))
    random.shuffle(indices)
    val_count = max(1, int(len(indices)*args.val_frac))
    val_idx = indices[:val_count]
    train_idx = indices[val_count:]
    train_images = [image_paths[i] for i in train_idx]
    train_masks  = [mask_paths[i] for i in train_idx]
    val_images   = [image_paths[i] for i in val_idx]
    val_masks    = [mask_paths[i] for i in val_idx]

    print(f"Train vols: {len(train_images)}, Val vols: {len(val_images)}")

    # dataset & loader
    train_ds = LiverPatchDataset(train_images, train_masks, patch_size=tuple(args.patch_size),
                                 samples_per_volume=args.samples_per_volume, pos_ratio=args.pos_ratio,
                                 transform=(augment_ct3d if args.augment else None),
                                 resample_spacing=tuple(args.resample_spacing) if args.resample_spacing else None)
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    # model
    model = UNet3D(in_ch=1, out_ch=1, base_filters=args.base_filters).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler()
    criterion = DiceBCELoss(weight_bce=args.bce_weight)

    best_val_dice = 0.0
    global_step = 0
    accumulation_steps = args.accumulation_steps
    start_epoch = 1

    if args.resume and os.path.isfile(args.resume):
            print(f"[resume] Loading checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            best_val_dice = checkpoint.get("best_val_dice", 0.0)
            start_epoch = checkpoint["epoch"] + 1
            print(f"[resume] Resumed from epoch {start_epoch} (best dice={best_val_dice:.4f})")

    try:
        # training
        for epoch in range(start_epoch, args.epochs + 1):
            model.train()
            running_loss = 0.0
            pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}/{args.epochs}")
            optimizer.zero_grad()

            for step, (img, msk) in pbar:
                img = img.to(device, non_blocking=True)
                msk = msk.to(device, non_blocking=True)

                with autocast(device_type='cuda'):
                    logits = model(img)
                    loss = criterion(logits, msk)
                    loss = loss / accumulation_steps

                scaler.scale(loss).backward()

                if (step + 1) % accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=12)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    global_step += 1

                running_loss += loss.item() * accumulation_steps
                pbar.set_postfix({"loss": f"{running_loss/(step+1):.4f}", "lr": optimizer.param_groups[0]['lr']})

            scheduler.step()

            # validation
            if epoch % args.val_every == 0 or epoch == 1:
                model.eval()
                dices = []
                for vi, (img_path, msk_path) in enumerate(zip(val_images, val_masks)):
                    img, sp = load_nifti(img_path)
                    msk, sp2 = load_nifti(msk_path)
                    if args.resample_spacing:
                        if tuple(sp) != tuple(args.resample_spacing):
                            img = resample_to_spacing(img, sp, args.resample_spacing)
                            msk = resample_to_spacing(msk, sp, args.resample_spacing)
                    img = intensity_clip_normalize(img, clip_min=args.clip_min, clip_max=args.clip_max)
                    gt = (msk > 0).astype(np.uint8)
                    prob_map = sliding_window_inference(img, model, device, patch_size=tuple(args.patch_size),
                                                    stride_factor=args.sw_stride, batch_size=args.sw_batch)
                    pred = (prob_map >= 0.5).astype(np.uint8)
                    inter = (pred * gt).sum()
                    dice = (2*inter) / (pred.sum() + gt.sum() + 1e-8)
                    dices.append(dice)

                mean_dice = float(np.mean(dices))
                print(f"Epoch {epoch} validation mean Dice: {mean_dice:.4f}")

                if mean_dice > best_val_dice:
                    best_val_dice = mean_dice
                    torch.save({
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "scheduler_state": scheduler.state_dict(),
                        "epoch": epoch,
                        "best_val_dice": best_val_dice
                    }, os.path.join(args.output_dir, "best_model.pth"))
                    print(f"Saved best model with Dice={best_val_dice:.4f}")

            # checkpoint every 4 epochs
            if epoch % 4 == 0:
                save_checkpoint(epoch, model, optimizer, scheduler, best_val_dice, args, tag="epoch")

    except KeyboardInterrupt:
        print("\n[interrupted] Training interrupted. Saving current checkpoint...")
        save_checkpoint(epoch, model, optimizer, scheduler, best_val_dice, args, tag="interrupted")
        print("Checkpoint saved. Exiting safely.")
        sys.exit(0)

    print("Training finished. Best val dice:", best_val_dice)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="./ml/results")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--patch_size", nargs=3, type=int, default=[64,128,128])
    p.add_argument("--base_filters", type=int, default=16)
    p.add_argument("--samples_per_volume", type=int, default=32)
    p.add_argument("--pos_ratio", type=float, default=0.5, help="the probability of taking a patch with a liver")
    p.add_argument("--augment", action="store_true")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--val_frac", type=float, default=0.15)
    p.add_argument("--val_every", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bce_weight", type=float, default=0.5)
    p.add_argument("--accumulation_steps", type=int, default=4)
    p.add_argument("--resample_spacing", nargs=3, type=float, default=None, help="example, 1.5 1.5 1.5") # not implemented
    p.add_argument("--sw_stride", type=float, default=0.5, help="stride factor для sliding window")
    p.add_argument("--sw_batch", type=int, default=4, help="batch size для sliding window inference")
    p.add_argument("--clip_min", type=float, default=-200)
    p.add_argument("--clip_max", type=float, default=250)
    p.add_argument("--resume", type=str, default=None, help="path to checkpoint")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    train(args)
