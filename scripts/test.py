# scripts/test.py
import os
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import numpy as np
import matplotlib.pyplot as plt

# Optional deps
_HAVE_SK = True
try:
    from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, precision_recall_fscore_support
except Exception:
    _HAVE_SK = False  # noqa

try:
    import seaborn as sns
    _HAVE_SNS = True
except Exception:
    _HAVE_SNS = False

from timm import create_model
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def device_auto():
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_model(arch: str, num_classes: int, drop_rate: float = 0.0, drop_path_rate: float = 0.0, device=None):
    m = create_model(
        arch,
        pretrained=False,
        num_classes=num_classes,
        drop_rate=drop_rate,
        drop_path_rate=drop_path_rate
    )
    if device is not None:
        m = m.to(device)
    return m


def load_weights(model: nn.Module, model_path: str, device) -> None:
    ckpt = torch.load(model_path, map_location=device)

    # If this is a "best checkpoint" payload from the training script
    if isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
        state_dict = ckpt["model"]
    else:
        # Assume it's a raw state dict (e.g., vit_final.pth)
        state_dict = ckpt

    # Strip "module." if present (in case of DDP)
    new_sd = {}
    for k, v in state_dict.items():
        nk = k[len("module."):] if k.startswith("module.") else k
        new_sd[nk] = v

    # Load non-strict to be tolerant to head/EMA/norm key diffs
    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    if missing or unexpected:
        print("[test.py] Note: non-strict load")
        if missing:
            print("  Missing keys:", missing[:10], ("..." if len(missing) > 10 else ""))
        if unexpected:
            print("  Unexpected keys:", unexpected[:10], ("..." if len(unexpected) > 10 else ""))


def build_transforms(img_size: int):
    # Mirror the eval pipeline used in train.py
    eval_tfms = transforms.Compose([
        transforms.Resize(int(img_size * 1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    ])
    return eval_tfms


def make_loader(root: str, img_size: int, batch_size: int, device):
    tfm = build_transforms(img_size)
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Test directory not found: {root}")
    ds = datasets.ImageFolder(root, transform=tfm)
    if len(ds) == 0:
        raise ValueError(f"No images found under {root}")
    pin = device.type in ("cuda", "xpu")
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=pin)
    return ds, dl


@torch.no_grad()
def forward_logits(model: nn.Module, imgs: torch.Tensor, device, tta: bool = False):
    imgs = imgs.to(device, non_blocking=True)
    logits = model(imgs)
    if not tta:
        return logits

    # Simple TTA: horizontal flip
    imgs_flip = torch.flip(imgs, dims=[3])  # flip width
    logits_flip = model(imgs_flip)
    return 0.5 * (logits + logits_flip)


def plot_confmat(cm: np.ndarray, classes, title: str = "Confusion Matrix"):
    plt.figure(figsize=(5, 4))
    if _HAVE_SNS:
        sns.heatmap(cm, annot=True, fmt="d", xticklabels=classes, yticklabels=classes, cmap="Blues")
    else:
        plt.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            plt.text(j, i, f"{int(v)}", ha="center", va="center", color="white" if v > cm.max()/2 else "black")
        plt.xticks(np.arange(len(classes)), classes, rotation=45, ha="right")
        plt.yticks(np.arange(len(classes)), classes)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def summarize_metrics(y_true, y_prob, y_pred, pos_idx=1, tag="TEST"):
    acc = (y_pred == y_true).mean()
    line = f"[{tag}] | Acc: {acc*100:.2f}%"
    if _HAVE_SK:
        try:
            auc = roc_auc_score(y_true, y_prob[:, pos_idx])
            line += f" | AUC: {auc:.4f}"
        except Exception:
            pass
        try:
            prec, rec, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average="binary", pos_label=pos_idx
            )
            line += f" | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f}"
        except Exception:
            pass
    print(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_dir", type=str, default=str(Path(__file__).resolve().parents[1] / "data" / "processed" / "test"))
    ap.add_argument("--model_path", type=str, required=True,
                    help="Path to vit_final.pth or vit_best.pth (from any past run).")
    ap.add_argument("--arch", type=str, default="vit_base_patch16_224",
                    help="timm model name used during training.")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--drop_rate", type=float, default=0.0)
    ap.add_argument("--drop_path_rate", type=float, default=0.0)
    ap.add_argument("--tta", action="store_true", help="Enable simple horizontal-flip TTA.")

    # NEW: calibration & tuned threshold (minimal change)
    ap.add_argument("--temperature", type=float, default=None,
                    help="If set, divide logits by T before softmax (temperature scaling).")
    ap.add_argument("--threshold", type=float, default=None,
                    help="If set, use probs[:,1] >= threshold instead of argmax for predictions.")
    args = ap.parse_args()

    device = device_auto()
    print(f"Using device: {device}")

    # Data
    test_ds, test_loader = make_loader(args.test_dir, args.img_size, args.batch_size, device)
    classes = test_ds.classes
    print(f"Classes detected in test set: {classes} (n={len(test_ds)})")

    pos_idx = 1  # assume classes=['fake','real']; 'real' is positive index 1

    # Model
    model = build_model(args.arch, num_classes=2, drop_rate=args.drop_rate, drop_path_rate=args.drop_path_rate, device=device)
    load_weights(model, args.model_path, device)
    model.eval()

    # Eval loop
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for imgs, labels in test_loader:
            logits = forward_logits(model, imgs, device, tta=args.tta)
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

    all_logits = torch.cat(all_logits, dim=0)

    # Apply temperature scaling if provided
    if args.temperature is not None and args.temperature > 0:
        all_logits = all_logits / args.temperature

    all_labels = torch.cat(all_labels, dim=0).numpy()
    probs = torch.softmax(all_logits, dim=1).numpy()

    # ---- Report RAW @0.50 ----
    preds_050 = (probs[:, pos_idx] >= 0.5).astype(np.int64)
    print()
    summarize_metrics(all_labels, probs, preds_050, pos_idx=pos_idx, tag="TEST @0.50")

    # Confusion matrix + classification report for RAW @0.50
    if _HAVE_SK:
        cm = confusion_matrix(all_labels, preds_050, labels=list(range(len(classes))))
        print("\nConfusion Matrix (@0.50):\n", cm)
        plot_confmat(cm, classes, title="Confusion Matrix (@0.50)")
        print("\nClassification Report (@0.50):")
        print(classification_report(all_labels, preds_050, target_names=classes, digits=4))
    else:
        print("(sklearn not installed) Skipping confusion matrix and report. Run: pip install scikit-learn seaborn")

    # ---- Report TUNED threshold, if provided ----
    if args.threshold is not None:
        preds_tuned = (probs[:, pos_idx] >= float(args.threshold)).astype(np.int64)
        summarize_metrics(all_labels, probs, preds_tuned, pos_idx=pos_idx, tag=f"TEST @tuned({args.threshold:.3f})")

        if _HAVE_SK:
            cm_t = confusion_matrix(all_labels, preds_tuned, labels=list(range(len(classes))))
            print("\nConfusion Matrix (@tuned):\n", cm_t)
            plot_confmat(cm_t, classes, title=f"Confusion Matrix (@tuned τ={args.threshold:.3f})")
            print("\nClassification Report (@tuned):")
            print(classification_report(all_labels, preds_tuned, target_names=classes, digits=4))


if __name__ == "__main__":
    main()
