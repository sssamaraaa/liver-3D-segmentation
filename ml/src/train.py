import os
import sys
import logging
import numpy as np
import torch
import matplotlib
import hydra
from dotenv import load_dotenv
from sklearn.model_selection import KFold
from glob import glob
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from torch.utils.data import DataLoader
from omegaconf import DictConfig
from hydra.utils import get_original_cwd
from configs.logging import setup_logging
from dataset import LiverPatchDataset, augment_ct3d, split_dataset
from model import UNet3D
from metrics import DiceBCELoss, dice, iou, precision, recall
from inference import sliding_window_inference
from utils import save_checkpoint, seed_everything, worker_init_fn, save_metrics_plots, load_checkpoint, load_model_from_checkpoint

load_dotenv()
ML_CONFIGS_DIR = os.getenv("ML_CONFIGS_DIR")
matplotlib.use("Agg")
logger = logging.getLogger(__name__)


def setup_env(cfg):
    seed_everything(cfg.training.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    return device

def initialize_dataset(cfg):
    orig_cwd = get_original_cwd()
    data_dir = os.path.join(orig_cwd, cfg.data.data_dir)

    image_paths = sorted(glob(os.path.join(data_dir, "imagesTr_npy", "*.npy")))
    mask_paths = sorted(glob(os.path.join(data_dir, "labelsTr_npy", "*.npy")))

    assert len(image_paths) > 0, f"No .npy files in {data_dir}"
    assert len(image_paths) == len(mask_paths), "Mismatch image/mask"
    logger.error(f"Mismatch image/mask")

    return image_paths, mask_paths

def build_dataloader(cfg, train_images, train_masks):
    train_ds = LiverPatchDataset(
        train_images,
        train_masks,
        patch_size=tuple(cfg.data.patch_size),
        patches_per_volume=cfg.data.patches_per_volume,
        pos_ratio=cfg.data.pos_ratio,
        transform=(augment_ct3d if cfg.data.augment else None)
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        persistent_workers=True if cfg.data.num_workers > 0 else False,
        worker_init_fn=worker_init_fn
    )

    return train_loader

def build_model(cfg, device):
    model = UNet3D(
        in_ch=cfg.model.in_ch,
        out_ch=cfg.model.out_ch,
        base_filters=cfg.model.base_filters
    ).to(device)
    return model

def build_training_components(cfg, model):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay
    )

    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.training.epochs
        )
        if cfg.training.scheduler == "cosine"
        else None
    )

    scaler = GradScaler()
    criterion = DiceBCELoss(weight_bce=cfg.training.bce_weight)

    return optimizer, scheduler, scaler, criterion

def build_training_pipeline(cfg):
    device = setup_env(cfg)
    image_paths, mask_paths = initialize_dataset(cfg)

    train_images, train_masks, val_images, val_masks = split_dataset(
        image_paths,
        mask_paths,
        cfg.data.val_frac
    )

    logger.info(f"Train vols: {len(train_images)}, Val vols: {len(val_images)}")

    train_loader = build_dataloader(cfg, train_images, train_masks)
    model = build_model(cfg, device)
    optimizer, scheduler, scaler, criterion = build_training_components(cfg, model)

    return device, train_loader, val_images, val_masks, model, optimizer, scheduler, scaler, criterion

def build_finetune_pipeline(cfg, fold_idx=0):
    device = setup_env(cfg)
    image_paths, mask_paths = initialize_dataset(cfg)

    kf = KFold(n_splits=cfg.training.kfold, shuffle=True, random_state=cfg.training.seed)
    splits = list(kf.split(image_paths))
    train_idx, val_idx = splits[fold_idx]

    train_images = [image_paths[i] for i in train_idx]
    train_masks = [mask_paths[i] for i in train_idx]
    val_images = [image_paths[i] for i in val_idx]
    val_masks = [mask_paths[i] for i in val_idx]

    logger.info(f"Fold {fold_idx+1}. Train vols: {len(train_images)}, Val vols: {len(val_images)}")

    train_loader = build_dataloader(cfg, train_images, train_masks)

    model = load_model_from_checkpoint(cfg, device)

    if cfg.training.freeze_encoder:
        model.freeze_encoder()

    optimizer, scheduler, scaler, criterion = build_training_components(cfg, model)

    for param_group in optimizer.param_groups:
        param_group["lr"] = cfg.training.finetune_lr

    return device, train_loader, val_images, val_masks, model, optimizer, scheduler, scaler, criterion

def train_one_epoch(cfg, train_loader, epoch, device, model, criterion, optimizer, scaler, accumulation_steps, epoch_losses, running_loss, global_step):
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}/{cfg.training.epochs}")

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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.training.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            global_step += 1

        true_loss = loss.item() * accumulation_steps
        running_loss += true_loss
        epoch_losses.append(true_loss)

        pbar.set_postfix({
            "loss": f"{running_loss / (step + 1):.4f}",
            "lr": optimizer.param_groups[0]["lr"]
        })

    return running_loss, global_step

def validate(model, epoch, epoch_losses, val_images, val_masks, device, cfg):
    val_stats = {
        "epoch": epoch,
        "losses": epoch_losses,
        "dices": [], "ious": [], "precisions": [], "recalls": [],
        "val_cases": 0, "skipped_empty_gt": 0, "fp_on_empty_gt": 0
    }

    model.eval()
    with torch.no_grad():
        for img_path, msk_path in zip(val_images, val_masks):

            img = np.load(img_path).astype(np.float32)
            msk = np.load(msk_path).astype(np.float32)
            gt = (msk > 0).astype(np.uint8)

            prob_map = sliding_window_inference(
                img, model, device,
                patch_size=tuple(cfg.data.patch_size),
                stride_factor=cfg.inference.sw_stride,
                batch_size=cfg.inference.sw_batch
            )

            if gt.sum() == 0:
                pred = (prob_map >= cfg.inference.threshold).astype(np.uint8)
                if pred.sum() > 0:
                    val_stats["fp_on_empty_gt"] += 1
                val_stats["skipped_empty_gt"] += 1
                continue

            pred = (prob_map >= cfg.inference.threshold).astype(np.uint8)

            inter = int(np.logical_and(pred, gt).sum())
            p_sum = int(pred.sum())
            g_sum = int(gt.sum())

            val_stats["dices"].append(float(dice(inter, p_sum, g_sum)))
            val_stats["ious"].append(float(iou(inter, p_sum, g_sum)))
            val_stats["precisions"].append(float(precision(inter, p_sum)))
            val_stats["recalls"].append(float(recall(inter, g_sum)))
            val_stats["val_cases"] += 1

    return val_stats

def run_training(cfg, device, model, criterion, optimizer, scheduler, scaler, train_loader, val_images, val_masks):
    start_epoch = 1
    all_epoch_stats = []
    global_step = 0
    best_val_dice = 0.0

    if cfg.training.resume is not None:
        start_epoch, best_val_dice, cfg = load_checkpoint(
            cfg.training.resume, model, optimizer, scheduler, device
        )

    try:
        for epoch in range(start_epoch, cfg.training.epochs + 1):
            epoch_losses = []
            model.train()
            running_loss = 0.0
            optimizer.zero_grad()

            running_loss, global_step = train_one_epoch(
                cfg, train_loader, epoch, device, model,
                criterion, optimizer, scaler,
                cfg.training.accumulation_steps,
                epoch_losses, running_loss, global_step
            )

            if scheduler is not None:
                scheduler.step()

            val_stats = validate(model, epoch, epoch_losses, val_images, val_masks, device, cfg)

            mean_dice = float(np.mean(val_stats["dices"])) if val_stats["dices"] else 0.0
            val_stats["mean_dice"] = mean_dice

            logger.info(f"Epoch {epoch} Dice: {mean_dice:.4f}")

            if mean_dice > best_val_dice:
                best_val_dice = mean_dice
                save_checkpoint(epoch, model, optimizer, scheduler, best_val_dice, cfg, tag="best_model.pth")

            if epoch % cfg.training.checkpoint_every == 0:
                save_checkpoint(epoch, model, optimizer, scheduler, best_val_dice, cfg, tag="intermediate.pth")

            all_epoch_stats.append(val_stats)
            save_metrics_plots(all_epoch_stats, cfg.training.output_dir_metrics)

    except KeyboardInterrupt:
        logger.info("Interrupted. Saving checkpoint...")
        save_checkpoint(epoch, model, optimizer, scheduler, best_val_dice, cfg, tag="interrupted")
        sys.exit(0)

@hydra.main(config_path=ML_CONFIGS_DIR, config_name="ml_conf", version_base=None)
def main(cfg: DictConfig):
    hydra_loggers = ["hydra", "omegaconf"]
    for logger_name in hydra_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)
        logging.getLogger(logger_name).handlers.clear()
    
    setup_logging("ml")

    os.makedirs(cfg.training.output_dir, exist_ok=True)
    os.makedirs(cfg.training.output_dir_metrics, exist_ok=True)

    device, train_loader, val_images, val_masks, model, optimizer, scheduler, scaler, criterion = build_training_pipeline(cfg)

    if cfg.training.mode == "train":
        run_training(cfg, device, model, criterion, optimizer, scheduler, scaler, train_loader, val_images, val_masks)
    
    elif cfg.training.mode == "finetune":
        for fold_idx in range(cfg.training.kfold):
            logger.info(f"Fold {fold_idx+1}/{cfg.training.kfold}")
            device, train_loader, val_images, val_masks, model, optimizer, scheduler, scaler, criterion = build_finetune_pipeline(cfg, fold_idx)
            run_training(cfg, device, model, criterion, optimizer, scheduler, scaler, train_loader, val_images, val_masks)


if __name__ == "__main__":
    main()