"""
config.py
Configurable parameters, data paths, and model configurations for the
Ego-Centric Driving Scene Understanding and Situational Awareness Framework.
"""
import os
import torch

# --- Environment & Hardware ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Storage Directories ---
DATA_DIR = '/content/drive/MyDrive/trainv3/Subject26_2_data'
LOCAL_SCENE_DIR = '/content/local_ims'
OUTPUT_DIR = '/content/drive/MyDrive/scene_segmentations'
VISUAL_DIR = os.path.join(OUTPUT_DIR, 'hazard_evaluation')
GAZE_PROMPTS_PATH = '/content/gaze_prompts.json'
GEMINI_ADVICE_JSON_PATH = os.path.join(OUTPUT_DIR, 'gemini_advice.json')
VIDEO_OUTPUT_PATH = os.path.join(OUTPUT_DIR, 'full_driving_scene_graph.mp4')

# --- Pretrained Model Checkpoints ---
YOLO_MODEL_PATH = '/content/drive/MyDrive/pre_trained_yolov8s.pt'
SAM2_CHECKPOINT = '/content/drive/MyDrive/sam2.1_hiera_large.pt'
SAM2_MODEL_CFG = 'configs/sam2.1/sam2.1_hiera_l.yaml'
SAM2_REPO_ROOT = '/content/segment-anything-2'
VLM_MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"

# --- ADAS & Perception Hyperparameters ---
FRAME_RATE = 30
DISPERSION_THRESHOLD = 50
MIN_DURATION_MS = 100
FOVEAL_RADIUS = 60          # Calibrated range modeling human peripheral attention
ROLLING_WINDOW = 30         # Gantt chart horizontal viewing window
GHOST_LIMIT = 5             # Tracking look-back tolerance window
MEMORY_WINDOW = 20          # Working memory frame window

# --- Camera Intrinsics ---
# Adjust based on specialized calibration specifications
# Format: [cx, cy, fx, fy] where cx=width/2, cy=height/2
INTRINSICS = [384, 256, 500, 500]