# Resource file paths
EMOTION_DIR_PATH = "res/emotions"
HAARCASCADE_FRONTALFACE_DEFAULT_FILE_PATH = "res/haarcascade_frontalface_default.xml"

# Build file paths
BEST_MODEL_DIR_PATH = "build/best_model"
LAST_MODEL_DIR_PATH = "build/last_model"
TRAIN_LOG_FILE_PATH = "build/train_log.csv"

# Other file paths
BACKUP_DIR_PATH = "tmp/backup"
REPORT_DIR_PATH = "reports"

# Image
IMAGE_SIZE = (48, 48)

# Emotion types
LABELS = [
    "Neutral",
    "Happiness",
    "Surprise",
    "Sadness",
    "Anger",
    "Disgust",
    "Fear",
    "Contempt",
    "Unknown",
    "NF"
]

# Training config
TRAIN_EPOCH = 50
BATCH_SIZE = 16
