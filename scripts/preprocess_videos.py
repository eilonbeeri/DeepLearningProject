import os
import glob
import random
import cv2
import torch
from facenet_pytorch import MTCNN
from tqdm import tqdm
from pathlib import Path
import numpy as np

# Device
device = "xpu" if torch.xpu.is_available() else "cpu"

# Parameters
FRAMES_PER_VIDEO = 32
VAL_RATIO = 0.2
RAW_DATA_DIR = str(Path(__file__).resolve().parents[1] / "data")  # absolute path

# Output
OUTPUT_DIR = os.path.join(RAW_DATA_DIR, "processed")

# Initialize MTCNN face detector
mtcnn = MTCNN(keep_all=False, device=device)

def extract_frames_from_video(video_path, save_dir, frames_per_video=FRAMES_PER_VIDEO):
    cap = cv2.VideoCapture(video_path)
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if nframes == 0:
        cap.release()
        return
    idxs = sorted(random.sample(range(nframes), min(frames_per_video, nframes)))
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face = mtcnn(frame_rgb)
        if face is not None:
            # Convert face tensor to numpy
            face_np = face.permute(1, 2, 0).numpy()
            # Scale to [0,255] if in [0,1]
            if face_np.max() <= 1.0:
                face_np = face_np * 255
            face_np = np.clip(face_np, 0, 255).astype(np.uint8)
            # Skip frames with no face (all zeros)
            if face_np.sum() == 0:
                continue
            cv2.imwrite(os.path.join(save_dir, f"frame{i}.jpg"),
                        cv2.cvtColor(face_np, cv2.COLOR_RGB2BGR))
    cap.release()

def main():
    # Find all videos in original and manipulated sequences
    video_files = []
    for folder in ["original_sequences", "manipulated_sequences"]:
        video_files += glob.glob(os.path.join(RAW_DATA_DIR, folder, "*", "raw", "videos", "*.mp4"))

    print(f"Found {len(video_files)} videos.")

    # Shuffle for random train/val split
    random.shuffle(video_files)

    for video_path in tqdm(video_files, desc="Processing videos"):
        # Determine label
        if "original_sequences" in video_path:
            label = "real"
        else:
            label = "fake"

        # Decide train/val split
        split = "val" if random.random() < VAL_RATIO else "train"
        save_dir = os.path.join(OUTPUT_DIR, split, label, os.path.splitext(os.path.basename(video_path))[0])
        os.makedirs(save_dir, exist_ok=True)

        extract_frames_from_video(video_path, save_dir)

    print(f"✅ Preprocessing complete. Data saved under: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
