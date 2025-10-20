# Deepfake Detection with Vision Transformers (ViT)

> High-accuracy deepfake face detection using a fine-tuned ViT-Base/16 model with calibration, TTA, EMA, and advanced training strategy on FaceForensics++ dataset.

---

## Project Overview

This project implements a binary classification model to detect real vs. manipulated (fake) face images extracted from the FaceForensics++ dataset (https://github.com/ondyari/FaceForensics).  
Faces were cropped using MTCNN from sampled frames of real and tampered videos.  
The final detection model is based on Vision Transformer (ViT-Base/16) with Layer-wise Learning Rate Decay, Mixup, EMA weights, Test-Time Augmentation (TTA), and Temperature Scaling.

---

## Final Performance

| Configuration | Accuracy | Precision | Recall | F1-Score | AUC | Threshold |
|--------------|:-------:|:--------:|:-----:|:-------:|:---:|:--------:|
| Raw ViT output (no calibration) | 95.66% | 0.927 | 0.979 | 0.9526 | – | 0.50 |
| + Calibration + TTA + Tuned τ (Final Model) | 96.84% | 0.955 | 0.974 | 0.9648 | 0.9920 | 0.640 |

Final calibrated model achieved:  
>  Accuracy = 96.84%, F1 = 0.9648, ROC-AUC = 0.9920

---


---

## Model Architecture

> Based on **ViT-Base/16** (85.8M params) with:
- Patch Embedding (16×16 patches)
- 12 Transformer Encoder Blocks with MHSA + MLP + Residual Connections
- LayerNorm + Classification Head (2-way softmax)
- **Enhancements:**  
  Layer-wise LR Decay (LLRD) 
  Mixup / CutMix (early training)
  EMA Model Shadow Copy  
  Temperature Scaling (T=0.763)
  TTA (flip-averaged logits)



---

## How to Train
python ./scripts/train.py --batch_size 32 --num_workers 0

- Automatically detects GPU (Intel XPU / CUDA)
- Applies Mixup in early epochs and fine-tunes after epoch 14
- Saves best calibrated model to checkpoints/vit_best.pth
- Training stops early once validation loss plateaus

## How to Evaluate Final Model
python ./scripts/test.py --model_path ./checkpoints/vit_best.pth --tta

Outputs:
- Accuracy, Precision, Recall, F1-score, AUC
- Confusion Matrix
- Classification Report
