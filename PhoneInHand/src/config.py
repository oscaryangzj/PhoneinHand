"""共享输入协议与 R10 推理参数；所有路径相对于交接目录。"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
RECIPE_PATH = ROOT / "configs/R10.json"
MANIFEST_PATH = PROJECT_ROOT / "data/splits/R10.json"
MODEL_DIR = PROJECT_ROOT / "models/R10"
CACHE_DIR = ROOT / "runs/prepared/R10"

FEATURE_COLS = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
LABEL_MAP = {"non_handheld": 0, "handheld": 1}
LABEL_NAMES = ["non_handheld", "handheld"]
SAMPLE_RATE_HZ = 100
INPUT_DIM = 6
NUM_CLASSES = 2
CNN_CHANNELS = 32
KERNEL_SIZE = 5
DILATIONS = (1, 2, 4, 8)
GRAVITY_ALPHA = 0.97
ACC_CHANNELS = 3
WINDOW_SIZE = 128
STRIDE = 1
GRAD_CLIP = 1.0
HANDHELD_EVIDENCE_THRESHOLD = 0.95
NON_HANDHELD_EVIDENCE_THRESHOLD = 0.1
STABLE_EVIDENCE_RATIO = 0.9
CONTACT_TO_FREE_FRAMES = 60
FREE_TO_CONTACT_FRAMES = 35
