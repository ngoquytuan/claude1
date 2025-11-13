"""
Configuration settings for Keystroke GAN Demo
All constants and parameters are defined here
"""

# Data Collection Settings
TARGET_TEXT = "security2025"  # Text người dùng phải gõ
NUM_SAMPLES = 100             # Số lần gõ cần thu thập
SAMPLE_RATE = 1000           # Hz - sampling rate cho timestamps

# Authentication Thresholds
AUTH_THRESHOLD = 0.85        # 85% similarity để accept
GAN_SUCCESS_THRESHOLD = 0.90 # 90% để coi như GAN bypass thành công

# Model Parameters
LATENT_DIM = 100            # GAN latent space dimension
EPOCHS_AUTH = 50            # Epochs cho authentication model
EPOCHS_GAN = 1000           # Epochs cho GAN training
BATCH_SIZE = 32

# Feature Dimensions (calculated automatically from TARGET_TEXT)
# Dwell times: len(TARGET_TEXT)
# Flight times: len(TARGET_TEXT) - 1
# Digraph latency: len(TARGET_TEXT) - 1
# Statistical features: 4 (total_time, avg_dwell, avg_flight, typing_speed)
FEATURE_DIM = len(TARGET_TEXT) + (len(TARGET_TEXT) - 1) + (len(TARGET_TEXT) - 1) + 4

# Paths
DATA_RAW_PATH = "data/raw/"
DATA_PROCESSED_PATH = "data/processed/"
MODEL_PATH = "data/models/"
LOG_PATH = "logs/"

# One-Class SVM Parameters
SVM_KERNEL = 'rbf'
SVM_NU = 0.1  # Outlier fraction
SVM_GAMMA = 'scale'

# GAN Parameters
GAN_LEARNING_RATE = 0.0002
GAN_BETA_1 = 0.5
GAN_DROPOUT_RATE = 0.3
GAN_LEAKY_RELU_ALPHA = 0.2

# Demo UI Settings
UI_WINDOW_WIDTH = 1200
UI_WINDOW_HEIGHT = 800
UI_FONT_LARGE = ("Arial", 16, "bold")
UI_FONT_MEDIUM = ("Arial", 12)
UI_FONT_SMALL = ("Arial", 10)
UI_COLOR_SUCCESS = "#2ecc71"
UI_COLOR_FAILURE = "#e74c3c"
UI_COLOR_INFO = "#3498db"
UI_COLOR_WARNING = "#f39c12"

# Logging
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_LEVEL = 'INFO'
