# app/config.py
import os


YOLO_MODEL_PATH = os.path.join("model", "best (4).pt")


UPLOAD_DIR = "./uploads/input"
TEMP_DIR = "./uploads/temp"
RESULTS_DIR = "./uploads/results"

CONF_THRESHOLD = 0.5
