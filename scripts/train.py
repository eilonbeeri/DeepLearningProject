# scripts/train.py
"""
ViT training script with staged mixup->finetune, LLRD, EMA, calibration,
threshold tuning, rich eval (ROC/PR/CM/reliability), and robustness tests.

Runs great on Intel GPUs (XPU, bf16) and CUDA (fp16). CPU fallback works too.
"""

import os
import math
import csv
import json
import time
import random
import contextlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from tqdm import tqdm

from timm import create_model
from timm.data import Mixup
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.utils.model_ema import ModelEmaV2

# sklearn is optional (AUC, calibration curve bins). We handle absence gracefully.
_HAVE_SK = True
try:
    from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve, confusion_matrix
except Exception:
    _HAVE_SK = False


# ------------------------------ Config ------------------------------

@dataclass
class Config:
    data_root: str = str(Path(__file__).resolve().parents[1] / "data" / "processed")
    results_root: str = str(Path(__file__).resolve().parents[1] / "results")
    ckpt_dir: str = str(Path(__file__).resolve().parents[1] / "checkpoints")

    model_name: str = "vit_base_patch16_224"
    img_size: int = 224
    drop_rate: float = 0.2
    drop_path_rate: float = 0.1

    epochs: int = 35
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 0.05
    warmup_epochs: int = 3
    max_grad_norm: float = 1.0
    early_stopping_patience: int = 8
    num_workers: int = 4
    seed: int = 42

    # Regularization & aug
    label_smoothing: float = 0.1
    mixup_alpha: float = 0.2
    cutmix_alpha: float = 0.2
    mixup_stop_epoch: int = 0            # 0 => auto: ~40% of total epochs
    mixup_prob: float = 1.0
    mixup_switch_prob: float = 0.5
    mixup_mode: str = "batch"
    random_erasing_p: float = 0.25

    # LLRD
    use_llrd: bool = True
    llrd_layer_decay: float = 0.75
    head_lr_mult: float = 20.0           # stronger head

    # EMA
    use_ema: bool = True
    ema_decay: float = 0.9999

    # Training strategy
    freeze_epochs: int = 1
    balance_sampler: bool = True

    # Eval
    tta: bool = True
    run_robustness: bool = True

    # Resume
    resume: bool = True


# --------------------------- Utilities ---------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False, warn_only=True)

def choose_device() -> torch.device:
    if hasattr(torch, "xpu") and getattr(torch.xpu, "is_available", lambda: False)():
        return torch.device("xpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def get_amp_and_dtype(device: torch.device):
    if device.type == "cuda":
        autocast = torch.cuda.amp.autocast
        scaler = torch.cuda.amp.GradScaler(enabled=True)
        amp_dtype = torch.float16
    elif device.type == "xpu" and hasattr(torch, "xpu") and hasattr(torch.xpu, "amp") and hasattr(torch.xpu.amp, "autocast"):
        autocast = torch.xpu.amp.autocast
        scaler = None
        amp_dtype = torch.bfloat16
    else:
        autocast = contextlib.nullcontext
        scaler = None
        amp_dtype = None
    return autocast, scaler, amp_dtype

def warmup_cosine_lambda_factory(num_epochs: int, warmup_epochs: int):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))
        progress = float(epoch - warmup_epochs) / float(max(1, num_epochs - warmup_epochs))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr_lambda

class LabelSmoothingCE(nn.Module):
    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        assert 0.0 <= smoothing < 1.0
        self.smoothing = smoothing
    def forward(self, logits, target):
        num_classes = logits.size(-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(logits)
            true_dist.fill_(self.smoothing / (num_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        log_probs = torch.log_softmax(logits, dim=-1)
        return torch.mean(torch.sum(-true_dist * log_probs, dim=-1))

class SoftTargetCrossEntropy(nn.Module):
    def forward(self, logits, target):
        return torch.mean(torch.sum(-target * torch.log_softmax(logits, dim=-1), dim=-1))

def count_class_samples(image_folder: datasets.ImageFolder) -> Dict[int, int]:
    counts = {c: 0 for c in range(len(image_folder.classes))}
    for _, y in image_folder.samples:
        counts[y] += 1
    return counts

def make_weighted_sampler(dataset: datasets.ImageFolder) -> WeightedRandomSampler:
    counts = count_class_samples(dataset)
    class_weights = {c: 1.0 / max(1, cnt) for c, cnt in counts.items()}
    sample_weights = [class_weights[y] for _, y in dataset.samples]
    sample_weights = torch.DoubleTensor(sample_weights)
    return WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

def build_transforms(cfg: Config):
    train_tfms = transforms.Compose([
        transforms.RandomResizedCrop(cfg.img_size, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(0.2, 0.2, 0.2, 0.1)], p=0.4),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.15),
        transforms.RandomGrayscale(p=0.1),
        transforms.RandomPerspective(distortion_scale=0.1, p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        transforms.RandomErasing(p=cfg.random_erasing_p, scale=(0.02, 0.12), ratio=(0.3, 3.3), value='random')
    ])
    eval_tfms = transforms.Compose([
        transforms.Resize(int(cfg.img_size * 1.15)),
        transforms.CenterCrop(cfg.img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    ])
    return train_tfms, eval_tfms

def build_dataloaders(cfg: Config, device: torch.device):
    train_tfms, eval_tfms = build_transforms(cfg)
    train_dir = os.path.join(cfg.data_root, "train")
    val_dir = os.path.join(cfg.data_root, "val")
    test_dir = os.path.join(cfg.data_root, "test")

    train_set = datasets.ImageFolder(train_dir, transform=train_tfms)
    val_set = datasets.ImageFolder(val_dir, transform=eval_tfms)
    test_set = datasets.ImageFolder(test_dir, transform=eval_tfms) if os.path.isdir(test_dir) else None

    if cfg.balance_sampler:
        sampler = make_weighted_sampler(train_set)
        shuffle = False
    else:
        sampler = None
        shuffle = True

    pin = device.type in ("cuda", "xpu")
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=shuffle,
                              sampler=sampler, num_workers=cfg.num_workers,
                              pin_memory=pin, persistent_workers=cfg.num_workers > 0)
    val_loader = DataLoader(val_set, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=pin,
                            persistent_workers=cfg.num_workers > 0)
    test_loader = None
    if test_set is not None:
        test_loader = DataLoader(test_set, batch_size=cfg.batch_size, shuffle=False,
                                 num_workers=cfg.num_workers, pin_memory=pin,
                                 persistent_workers=cfg.num_workers > 0)
    return train_loader, val_loader, test_loader

def set_vit_freeze_policy(model: nn.Module, epoch: int, cfg: Config):
    requires_grad = epoch > cfg.freeze_epochs
    for name, p in model.named_parameters():
        if requires_grad:
            p.requires_grad = True
        else:
            if ("head" in name) or ("fc" in name) or ("blocks.%d" % (len(model.blocks) - 1) in name):
                p.requires_grad = True
            else:
                p.requires_grad = False

def get_llrd_param_groups(model: nn.Module, base_lr: float, weight_decay: float,
                          layer_decay: float = 0.75, head_lr_mult: float = 20.0):
    param_groups: Dict[str, Dict] = {}
    num_blocks = len(model.blocks) if hasattr(model, "blocks") else 0

    def add_group(gname, params, lr_mult, wd=weight_decay):
        if gname not in param_groups:
            param_groups[gname] = {"params": [], "lr": base_lr * lr_mult, "weight_decay": wd}
        param_groups[gname]["params"].extend(list(params))

    if hasattr(model, "patch_embed"):
        add_group("layer_0", model.patch_embed.parameters(), layer_decay ** (num_blocks + 1))
    if hasattr(model, "pos_embed"):
        add_group("pos_embed", [model.pos_embed], layer_decay ** (num_blocks + 1))

    for i, blk in enumerate(model.blocks):
        lr_mult = layer_decay ** (num_blocks - i)
        add_group(f"block_{i}", blk.parameters(), lr_mult)

    if hasattr(model, "norm"):
        add_group("norm", model.norm.parameters(), 1.0)
    if hasattr(model, "head"):
        add_group("head", model.head.parameters(), head_lr_mult)

    return list(param_groups.values())

def build_model_and_optimizer(cfg: Config, device: torch.device):
    model = create_model(
        cfg.model_name,
        pretrained=True,
        num_classes=2,
        drop_rate=cfg.drop_rate,
        drop_path_rate=cfg.drop_path_rate
    ).to(device)

    if cfg.use_llrd:
        param_groups = get_llrd_param_groups(model, cfg.lr, cfg.weight_decay,
                                             layer_decay=cfg.llrd_layer_decay,
                                             head_lr_mult=cfg.head_lr_mult)
        optimizer = optim.AdamW(param_groups, eps=1e-8, betas=(0.9, 0.999))
    else:
        optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, eps=1e-8, betas=(0.9, 0.999))
    return model, optimizer

def make_criterion_and_mixup(cfg: Config, use_mixup: bool):
    mixup_fn = None
    if use_mixup and (cfg.mixup_alpha > 0.0 or cfg.cutmix_alpha > 0.0):
        mixup_fn = Mixup(
            mixup_alpha=cfg.mixup_alpha,
            cutmix_alpha=cfg.cutmix_alpha,
            prob=cfg.mixup_prob,
            switch_prob=cfg.mixup_switch_prob,
            mode=cfg.mixup_mode,
            label_smoothing=cfg.label_smoothing,
            num_classes=2
        )
        criterion = SoftTargetCrossEntropy()
    else:
        criterion = LabelSmoothingCE(smoothing=cfg.label_smoothing)
    return criterion, mixup_fn

def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    return (preds == targets).float().mean().item()

def compute_binary_metrics(all_logits: torch.Tensor, all_targets: torch.Tensor, threshold: float = 0.5) -> Dict[str, float]:
    probs = torch.softmax(all_logits, dim=1)[:, 1]
    preds = (probs >= threshold).long()
    t = all_targets.long()

    tp = ((preds == 1) & (t == 1)).sum().item()
    tn = ((preds == 0) & (t == 0)).sum().item()
    fp = ((preds == 1) & (t == 0)).sum().item()
    fn = ((preds == 0) & (t == 1)).sum().item()

    acc = (tp + tn) / max(1, tp + tn + fp + fn)
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-12, (prec + rec))

    metrics = {"acc": acc, "precision": prec, "recall": rec, "f1": f1}
    if _HAVE_SK:
        try:
            auc = float(roc_auc_score(t.cpu().numpy(), probs.cpu().numpy()))
            metrics["auc"] = auc
        except Exception:
            pass
    return metrics

def save_training_plot_and_csv(run_dir: Path, history: Dict[str, List[float]]):
    run_dir.mkdir(parents=True, exist_ok=True)
    # CSV
    csv_path = run_dir / "training_log.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["epoch", "train_loss", "val_loss", "train_acc", "val_acc"]
        writer.writerow(header)
        for i in range(len(history["train_loss"])):
            writer.writerow([
                i + 1,
                f"{history['train_loss'][i]:.6f}",
                f"{history['val_loss'][i]:.6f}",
                f"{history['train_acc'][i]:.6f}",
                f"{history['val_acc'][i]:.6f}",
            ])

    # Plot
    plt.figure(figsize=(12, 5))
    xs = list(range(1, len(history["train_loss"]) + 1))
    plt.subplot(1, 2, 1)
    plt.plot(xs, history["train_loss"], label="Train Loss")
    plt.plot(xs, history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(xs, history["train_acc"], label="Train Acc")
    plt.plot(xs, history["val_acc"], label="Val Acc")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend()

    plt.tight_layout()
    plt.savefig(run_dir / "training_plot.png")
    plt.close()

def save_checkpoint(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(path))

def load_checkpoint(path: Path, device: torch.device):
    return torch.load(str(path), map_location=device)

# ---------- Calibration (temperature scaling) ----------

class _NLLLoss(nn.Module):
    def forward(self, logits, targets):
        return nn.functional.cross_entropy(logits, targets)

def fit_temperature(logits: torch.Tensor, targets: torch.Tensor, init_temp: float = 1.0, max_iter: int = 50) -> float:
    """
    Learn a single temperature T to minimize NLL on validation logits.
    """
    T = torch.tensor([init_temp], requires_grad=True, dtype=logits.dtype, device=logits.device)
    optimizer = torch.optim.LBFGS([T], lr=0.01, max_iter=max_iter, tolerance_grad=1e-7, tolerance_change=1e-9)
    nll = _NLLLoss()

    def closure():
        optimizer.zero_grad()
        loss = nll(logits / T.clamp_min(1e-3), targets)
        loss.backward()
        return loss
    optimizer.step(closure)
    return float(T.detach().clamp_min(1e-3).item())

def apply_temperature(logits: torch.Tensor, T: float) -> torch.Tensor:
    if T is None or T == 1.0:
        return logits
    return logits / max(1e-3, T)

# ---------- Plots ----------

def plot_confusion_matrix(cm: np.ndarray, classes: List[str], save_path: Path):
    plt.figure(figsize=(4.2, 4))
    plt.imshow(cm, interpolation='nearest')
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha='right')
    plt.yticks(tick_marks, classes)
    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], fmt),
                 ha="center", va="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label'); plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_roc_pr_reliability(probs: np.ndarray, t: np.ndarray, save_dir: Path):
    if not _HAVE_SK:
        return
    # ROC
    fpr, tpr, _ = roc_curve(t, probs)
    plt.figure(); plt.plot(fpr, tpr); plt.plot([0,1],[0,1],'--')
    plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('ROC')
    plt.tight_layout(); plt.savefig(save_dir / "roc.png"); plt.close()

    # PR
    prec, rec, _ = precision_recall_curve(t, probs)
    plt.figure(); plt.plot(rec, prec)
    plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title('Precision-Recall')
    plt.tight_layout(); plt.savefig(save_dir / "pr.png"); plt.close()

    # Reliability (calibration) diagram
    bins = np.linspace(0.0, 1.0, 11)
    binids = np.digitize(probs, bins) - 1
    accs, confs = [], []
    for i in range(len(bins)-1):
        idx = binids == i
        if np.any(idx):
            accs.append(np.mean((probs[idx] >= 0.5) == (t[idx] == 1)))
            confs.append(np.mean(probs[idx]))
    plt.figure()
    plt.plot([0,1],[0,1],'--')
    if len(confs) > 0:
        plt.plot(confs, accs, marker='o')
    plt.xlabel('Confidence'); plt.ylabel('Accuracy'); plt.title('Reliability')
    plt.tight_layout(); plt.savefig(save_dir / "reliability.png"); plt.close()

# ------------------------------ Train / Eval ------------------------------

def train_one_epoch(model, optimizer, loader, device, criterion, mixup_fn, scaler, autocast_ctx, cfg: Config, epoch: int, ema: Optional[ModelEmaV2]):
    set_vit_freeze_policy(model, epoch, cfg)

    model.train()
    running_loss = 0.0
    running_acc = 0.0
    samples = 0

    pbar = tqdm(loader, desc=f"Train {epoch}/{cfg.epochs}", unit="batch")
    for imgs, targets in pbar:
        imgs = imgs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            imgs, targets = mixup_fn(imgs, targets)

        optimizer.zero_grad(set_to_none=True)

        with autocast_ctx():
            logits = model(imgs)
            loss = criterion(logits, targets)

        if not math.isfinite(loss.item()):
            for g in optimizer.param_groups:
                g["lr"] *= 0.5
            continue

        if scaler is not None:
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()

        if ema is not None:
            ema.update(model)

        bsz = imgs.size(0)
        running_loss += loss.item() * bsz
        if mixup_fn is None:
            running_acc += accuracy_from_logits(logits, targets) * bsz
        samples += bsz

        pbar.set_postfix(loss=running_loss / max(1, samples),
                         acc=(running_acc / max(1, samples)) if mixup_fn is None else "mixup")

    train_loss = running_loss / max(1, samples)
    train_acc = (running_acc / max(1, samples)) if mixup_fn is None else float("nan")
    return train_loss, train_acc

@torch.no_grad()
def collect_logits_targets(model, loader, device, autocast_ctx, use_ema=False, ema: Optional[ModelEmaV2]=None):
    model_to_eval = ema.module if (use_ema and ema is not None) else model
    model_to_eval.eval()
    all_logits, all_targets = [], []
    for imgs, targets in loader:
        imgs = imgs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with autocast_ctx():
            logits = model_to_eval(imgs)
        all_logits.append(logits.detach().cpu())
        all_targets.append(targets.detach().cpu())
    return torch.cat(all_logits, dim=0), torch.cat(all_targets, dim=0)

@torch.no_grad()
def evaluate(model, loader, device, criterion, autocast_ctx, cfg: Config, use_ema: bool = False, ema: Optional[ModelEmaV2] = None, temperature: Optional[float] = None, threshold: float = 0.5):
    logits, targets = collect_logits_targets(model, loader, device, autocast_ctx, use_ema, ema)
    if temperature is not None:
        logits = apply_temperature(logits, temperature)
    loss = criterion(logits, targets).item()
    metrics = compute_binary_metrics(logits, targets, threshold=threshold)
    return loss, metrics, logits, targets

@torch.no_grad()
def evaluate_with_tta(model, loader, device, criterion, autocast_ctx, cfg: Config, use_ema: bool = False, ema: Optional[ModelEmaV2] = None, temperature: Optional[float] = None, threshold: float = 0.5):
    if not cfg.tta:
        return evaluate(model, loader, device, criterion, autocast_ctx, cfg, use_ema, ema, temperature, threshold)

    model_to_eval = ema.module if (use_ema and ema is not None) else model
    model_to_eval.eval()

    val_loss = 0.0
    all_logits = []
    all_targets = []
    samples = 0

    hflip = transforms.RandomHorizontalFlip(p=1.0)

    for imgs, targets in tqdm(loader, desc="Test(TTA)", unit="batch"):
        imgs = imgs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with autocast_ctx():
            logits1 = model_to_eval(imgs)
        # flip
        imgs_flip = hflip(imgs.detach().cpu()).to(device, non_blocking=True)
        with autocast_ctx():
            logits2 = model_to_eval(imgs_flip)

        logits = 0.5 * (logits1 + logits2)
        if temperature is not None:
            logits = apply_temperature(logits, temperature)

        # compute loss on un-augmented path for reporting
        with autocast_ctx():
            vloss = criterion(logits1 if temperature is None else apply_temperature(logits1, temperature), targets)
        val_loss += vloss.item() * imgs.size(0)

        all_logits.append(logits.detach().cpu())
        all_targets.append(targets.detach().cpu())
        samples += imgs.size(0)

    val_loss = val_loss / max(1, samples)
    all_logits = torch.cat(all_logits, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    metrics = compute_binary_metrics(all_logits, all_targets, threshold=threshold)
    return val_loss, metrics, all_logits, all_targets

# ------------------------------ Robustness ------------------------------

@torch.no_grad()
def robustness_eval(model, base_loader, device, autocast_ctx, cfg: Config, use_ema: bool, ema: Optional[ModelEmaV2], temperature: Optional[float], threshold: float, save_csv: Path):
    """
    Reuses the dataset files but applies extra eval-time transforms on-the-fly.
    """
    if base_loader is None:
        return

    test_ds: datasets.ImageFolder = base_loader.dataset  # type: ignore

    def make_loader(extra_tfm):
        # rebuild a dataset with additional transform after normalization
        # We wrap original transform to add a final extra_tfm in pixel space (pre-norm), so reconstruct:
        eval_tfms = transforms.Compose([
            transforms.Resize(int(cfg.img_size * 1.15)),
            transforms.CenterCrop(cfg.img_size),
            transforms.ToTensor(),
        ])
        norm = transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
        final = transforms.Compose([eval_tfms, extra_tfm, norm])
        ds = datasets.ImageFolder(test_ds.root, transform=final)
        return DataLoader(ds, batch_size=cfg.batch_size, shuffle=False,
                          num_workers=cfg.num_workers, pin_memory=device.type in ("cuda", "xpu"),
                          persistent_workers=cfg.num_workers > 0)

    specs = []
    # JPEG qualities
    for q in [95, 75, 50]:
        tfm = transforms.Lambda(lambda x, q=q: transforms.functional.adjust_jpeg_quality(x, q))
        specs.append((f"jpeg_{q}", tfm))
    # Gaussian blur
    for s in [0.5, 1.0]:
        tfm = transforms.GaussianBlur(kernel_size=3, sigma=s)
        specs.append((f"gblur_{s}", tfm))
    # Brightness/Contrast jitter
    specs.append(("bc_jitter", transforms.ColorJitter(brightness=0.3, contrast=0.3)))

    rows = []
    for name, tfm in specs:
        dl = make_loader(tfm)
        loss, metrics, _, _ = evaluate_with_tta(
            model, dl, device, LabelSmoothingCE(cfg.label_smoothing),
            autocast_ctx, cfg, use_ema=use_ema, ema=ema, temperature=temperature, threshold=threshold
        )
        row = {"shift": name, "loss": loss}
        row.update(metrics)
        rows.append(row)
        print(f"[robustness] {name}: acc={metrics['acc']:.4f} f1={metrics['f1']:.4f} "
              f"prec={metrics['precision']:.4f} rec={metrics['recall']:.4f} "
              + (f"auc={metrics['auc']:.4f}" if 'auc' in metrics else ""))

    # write CSV
    with open(save_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

# ------------------------------ Main ------------------------------

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default=Config.data_root)
    p.add_argument("--results_root", type=str, default=Config.results_root)
    p.add_argument("--ckpt_dir", type=str, default=Config.ckpt_dir)

    p.add_argument("--epochs", type=int, default=Config.epochs)
    p.add_argument("--batch_size", type=int, default=Config.batch_size)
    p.add_argument("--lr", type=float, default=Config.lr)
    p.add_argument("--weight_decay", type=float, default=Config.weight_decay)
    p.add_argument("--warmup_epochs", type=int, default=Config.warmup_epochs)
    p.add_argument("--max_grad_norm", type=float, default=Config.max_grad_norm)
    p.add_argument("--early_stopping_patience", type=int, default=Config.early_stopping_patience)
    p.add_argument("--num_workers", type=int, default=Config.num_workers)
    p.add_argument("--seed", type=int, default=Config.seed)

    p.add_argument("--img_size", type=int, default=Config.img_size)
    p.add_argument("--drop_rate", type=float, default=Config.drop_rate)
    p.add_argument("--drop_path_rate", type=float, default=Config.drop_path_rate)

    p.add_argument("--mixup", type=float, dest="mixup_alpha", default=Config.mixup_alpha)
    p.add_argument("--cutmix", type=float, dest="cutmix_alpha", default=Config.cutmix_alpha)
    p.add_argument("--mixup_stop_epoch", type=int, default=Config.mixup_stop_epoch)
    p.add_argument("--label_smoothing", type=float, default=Config.label_smoothing)
    p.add_argument("--random_erasing_p", type=float, default=Config.random_erasing_p)

    p.add_argument("--use_llrd", action="store_true", default=Config.use_llrd)
    p.add_argument("--llrd_layer_decay", type=float, default=Config.llrd_layer_decay)
    p.add_argument("--head_lr_mult", type=float, default=Config.head_lr_mult)

    p.add_argument("--use_ema", action="store_true", default=Config.use_ema)
    p.add_argument("--ema_decay", type=float, default=Config.ema_decay)

    p.add_argument("--freeze_epochs", type=int, default=Config.freeze_epochs)
    p.add_argument("--no_balance_sampler", action="store_true", default=not Config.balance_sampler)
    p.add_argument("--tta", action="store_true", default=Config.tta)
    p.add_argument("--no_robustness", action="store_true", default=not Config.run_robustness)

    p.add_argument("--resume", action="store_true", default=Config.resume)

    args = p.parse_args()
    cfg = Config(
        data_root=args.data_root,
        results_root=args.results_root,
        ckpt_dir=args.ckpt_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        max_grad_norm=args.max_grad_norm,
        early_stopping_patience=args.early_stopping_patience,
        num_workers=args.num_workers,
        seed=args.seed,
        img_size=args.img_size,
        drop_rate=args.drop_rate,
        drop_path_rate=args.drop_path_rate,
        label_smoothing=args.label_smoothing,
        mixup_alpha=args.mixup_alpha,
        cutmix_alpha=args.cutmix_alpha,
        mixup_stop_epoch=args.mixup_stop_epoch,
        random_erasing_p=args.random_erasing_p,
        use_llrd=args.use_llrd,
        llrd_layer_decay=args.llrd_layer_decay,
        head_lr_mult=args.head_lr_mult,
        use_ema=args.use_ema,
        ema_decay=args.ema_decay,
        freeze_epochs=args.freeze_epochs,
        balance_sampler=not args.no_balance_sampler,
        tta=args.tta,
        run_robustness=not args.no_robustness,
        resume=args.resume
    )

    # Setup
    set_seed(cfg.seed)
    device = choose_device()
    print(f"[train.py] Using device: {device} ({'bf16' if device.type=='xpu' else 'fp16' if device.type=='cuda' else 'fp32'})")

    autocast_ctx, scaler, amp_dtype = get_amp_and_dtype(device)

    # Data
    train_loader, val_loader, test_loader = build_dataloaders(cfg, device)
    print(f"[train.py] Dataset sizes: train={len(train_loader.dataset)}, val={len(val_loader.dataset)}, "
          f"test={len(test_loader.dataset) if test_loader is not None else 0}")

    # Model & optimizer
    model, optimizer = build_model_and_optimizer(cfg, device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train.py] Trainable params: {total_params/1e6:.2f}M")

    # Scheduler
    lr_lambda = warmup_cosine_lambda_factory(cfg.epochs, cfg.warmup_epochs)
    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    # EMA
    ema = ModelEmaV2(model, decay=cfg.ema_decay, device=device) if cfg.use_ema else None

    # Run dirs
    run_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(cfg.results_root) / "vit" / f"run_{run_time}"
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_best_path = Path(cfg.ckpt_dir) / "vit_best.pth"
    ckpt_resume_path = ckpt_best_path

    # Resume
    start_epoch = 1
    best_val_acc = 0.0
    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    if cfg.resume and ckpt_resume_path.exists():
        try:
            state = load_checkpoint(ckpt_resume_path, device)
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state.get("scheduler", {}))
            if scaler is not None and "scaler" in state:
                scaler.load_state_dict(state["scaler"])
            if ema is not None and "ema" in state:
                ema.module.load_state_dict(state["ema"])
            start_epoch = state.get("epoch", 0) + 1
            best_val_acc = state.get("best_val_acc", best_val_acc)
            best_val_loss = state.get("best_val_loss", best_val_loss)
            history = state.get("history", history)
            print(f"[train.py] Resumed from epoch {start_epoch-1}")
        except Exception as e:
            print(f"[train.py] Resume failed: {e}. Starting fresh.")

    # Decide staged schedule for mixup
    mixup_stop = cfg.mixup_stop_epoch if cfg.mixup_stop_epoch > 0 else max(3, int(cfg.epochs * 0.4))
    print(f"[train.py] Mixup/CutMix enabled until epoch {mixup_stop}, then fine-tune without mixup.")

    epochs_no_improve = 0
    temperature = None  # learned after training on val
    tuned_threshold = 0.5

    for epoch in range(start_epoch, cfg.epochs + 1):
        use_mixup = epoch <= mixup_stop and (cfg.mixup_alpha > 0 or cfg.cutmix_alpha > 0)
        criterion, mixup_fn = make_criterion_and_mixup(cfg, use_mixup=use_mixup)

        # Train
        train_loss, train_acc = train_one_epoch(
            model, optimizer, train_loader, device, criterion, mixup_fn, scaler, autocast_ctx, cfg, epoch, ema
        )

        # Validate with current state (no mixup in val)
        val_crit = LabelSmoothingCE(cfg.label_smoothing)
        val_loss, val_metrics, _, _ = evaluate(
            model, val_loader, device, val_crit, autocast_ctx, cfg, use_ema=cfg.use_ema, ema=ema, temperature=None, threshold=0.5
        )
        val_acc = val_metrics["acc"]

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(float(train_acc) if not math.isnan(train_acc) else float("nan"))
        history["val_acc"].append(val_acc)

        print(f"[train.py] Epoch {epoch}/{cfg.epochs} | "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc if not math.isnan(train_acc) else 'mixup'} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, "
              f"LR: {optimizer.param_groups[0]['lr']:.3e}")

        # Early stopping on val loss
        improved = val_loss < best_val_loss - 1e-6
        if improved:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # Save best by val acc
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            payload = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_val_acc": best_val_acc,
                "best_val_loss": best_val_loss,
                "history": history,
            }
            if scaler is not None:
                payload["scaler"] = scaler.state_dict()
            if ema is not None:
                payload["ema"] = ema.module.state_dict()
            save_checkpoint(ckpt_best_path, payload)
            print(f"[train.py] Saved best checkpoint (val_acc={val_acc:.4f}) -> {ckpt_best_path}")

        # Per-epoch lightweight save
        save_checkpoint(run_dir / f"epoch_{epoch}.pth", {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "history": history
        })

        if epochs_no_improve >= cfg.early_stopping_patience:
            print(f"[train.py] Early stopping at epoch {epoch} (no val_loss improvement for {cfg.early_stopping_patience} epochs).")
            break

    # Final save for this run
    final_path = run_dir / "vit_final.pth"
    torch.save(model.state_dict(), str(final_path))
    print(f"[train.py] Final model saved to {final_path}")

    # Save curves
    save_training_plot_and_csv(run_dir, history)

    # -------- Calibration + Threshold tuning on VAL --------
    print("[train.py] Calibrating temperature and tuning threshold on validation logits…")
    # Load best weights for eval
    if ckpt_best_path.exists():
        state = load_checkpoint(ckpt_best_path, device)
        model.load_state_dict(state["model"])
        if cfg.use_ema and "ema" in state:
            if 'ema' in state:
                if ema is None:
                    ema = ModelEmaV2(model, decay=cfg.ema_decay, device=device)
                ema.module.load_state_dict(state["ema"])

    # collect val logits/targets (EMA if used)
    val_logits, val_targets = collect_logits_targets(model, val_loader, device, autocast_ctx, use_ema=cfg.use_ema, ema=ema)
    # Fit temperature for calibration
    try:
        _dev = torch.device("cuda" if torch.cuda.is_available() else ("xpu" if hasattr(torch, "xpu") and torch.xpu.is_available() else "cpu"))
        temperature = fit_temperature(val_logits.to(_dev), val_targets.to(_dev))
        print(f"[train.py] Learned temperature T = {temperature:.3f}")
    except Exception as e:
        print(f"[train.py] Temperature scaling failed ({e}); continuing without calibration.")
        temperature = None

    # Threshold tuning (maximize F1 on val)
    with torch.no_grad():
        v_logits_adj = apply_temperature(val_logits, temperature) if temperature is not None else val_logits
        v_probs = torch.softmax(v_logits_adj, dim=1)[:, 1].numpy()
        v_true = val_targets.numpy()
        # search grid
        thresholds = np.linspace(0.1, 0.9, 81)
        best_f1, best_th = -1.0, 0.5
        for th in thresholds:
            preds = (v_probs >= th).astype(np.int64)
            tp = np.sum((preds == 1) & (v_true == 1))
            fp = np.sum((preds == 1) & (v_true == 0))
            fn = np.sum((preds == 0) & (v_true == 1))
            prec = tp / max(1, tp + fp)
            rec = tp / max(1, tp + fn)
            f1 = 2*prec*rec / max(1e-12, (prec + rec))
            if f1 > best_f1:
                best_f1, best_th = f1, float(th)
        tuned_threshold = best_th
        print(f"[train.py] F1-optimal threshold on val: {tuned_threshold:.3f} (F1={best_f1:.4f})")

    # -------- TEST evaluation (TTA + calibrated + tuned threshold) --------
    if test_loader is not None:
        print("[train.py] Evaluating on TEST set…")
        test_loss_05, test_metrics_05, test_logits, test_targets = evaluate_with_tta(
            model, test_loader, device, LabelSmoothingCE(cfg.label_smoothing),
            autocast_ctx, cfg, use_ema=cfg.use_ema, ema=ema, temperature=temperature, threshold=0.5
        )
        test_loss_tuned, test_metrics_tuned, test_logits_tuned, test_targets_tuned = evaluate_with_tta(
            model, test_loader, device, LabelSmoothingCE(cfg.label_smoothing),
            autocast_ctx, cfg, use_ema=cfg.use_ema, ema=ema, temperature=temperature, threshold=tuned_threshold
        )

        print(f"[train.py] TEST @0.50 | Loss: {test_loss_05:.4f} | "
              f"Acc: {test_metrics_05['acc']:.4f} | Prec: {test_metrics_05['precision']:.4f} | "
              f"Rec: {test_metrics_05['recall']:.4f} | F1: {test_metrics_05['f1']:.4f}"
              + (f" | AUC: {test_metrics_05['auc']:.4f}" if 'auc' in test_metrics_05 else ""))

        print(f"[train.py] TEST @tuned({tuned_threshold:.3f}) | Loss: {test_loss_tuned:.4f} | "
              f"Acc: {test_metrics_tuned['acc']:.4f} | Prec: {test_metrics_tuned['precision']:.4f} | "
              f"Rec: {test_metrics_tuned['recall']:.4f} | F1: {test_metrics_tuned['f1']:.4f}"
              + (f" | AUC: {test_metrics_tuned['auc']:.4f}" if 'auc' in test_metrics_tuned else ""))

        # Save metrics jsons
        out_base = run_dir / "test_metrics"
        out_base.mkdir(parents=True, exist_ok=True)
        with open(out_base / "metrics_thr_0.50.json", "w") as f:
            json.dump(test_metrics_05, f, indent=2)
        with open(out_base / f"metrics_thr_{tuned_threshold:.3f}.json", "w") as f:
            json.dump(test_metrics_tuned, f, indent=2)

        # Plots: CM, ROC/PR, Reliability (use tuned threshold, calibrated probs)
        classes = test_loader.dataset.classes if hasattr(test_loader.dataset, "classes") else ["class0", "class1"]
        tl = apply_temperature(test_logits_tuned, temperature) if temperature is not None else test_logits_tuned
        probs = torch.softmax(tl, dim=1)[:, 1].numpy()
        y_true = test_targets_tuned.numpy()
        y_pred = (probs >= tuned_threshold).astype(np.int64)

        if _HAVE_SK:
            cm = confusion_matrix(y_true, y_pred)
            plot_confusion_matrix(cm, classes, out_base / "confusion_matrix.png")
        plot_roc_pr_reliability(probs, y_true, out_base)

        # -------- Robustness battery --------
        if cfg.run_robustness:
            robustness_csv = out_base / "robustness.csv"
            robustness_eval(model, test_loader, device, autocast_ctx, cfg, use_ema=cfg.use_ema, ema=ema,
                            temperature=temperature, threshold=tuned_threshold, save_csv=robustness_csv)
    else:
        print("[train.py] No test/ directory found; skipped test evaluation and robustness.")

if __name__ == "__main__":
    main()
