# Keystroke Authentication + GAN Bypass Demo

Hệ thống demo cho mục đích giáo dục, minh họa cách GAN (Generative Adversarial Network) có thể học và bypass hệ thống xác thực dựa trên keystroke dynamics.

## Tổng Quan

Dự án này bao gồm:
- **Hệ thống xác thực** dựa trên keystroke dynamics (One-Class SVM)
- **GAN model** để tạo fake keystroke patterns
- **Demo UI** trực quan cho việc giảng dạy

## Cài Đặt

### 1. Clone Repository

```bash
git clone <repository-url>
cd keystroke-gan-demo
```

### 2. Tạo Virtual Environment (khuyến nghị)

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc
venv\Scripts\activate  # Windows
```

### 3. Cài Đặt Dependencies

```bash
pip install -r requirements.txt
```

## Cấu Trúc Dự Án

```
keystroke-gan-demo/
├── config/
│   └── settings.py              # Cấu hình constants
├── data/
│   ├── raw/                     # Raw keystroke data
│   ├── processed/               # Processed features
│   └── models/                  # Saved models
├── src/
│   ├── data_collection/
│   │   ├── collector.py         # Thu thập keystroke data
│   │   └── feature_extractor.py # Trích xuất features
│   ├── models/
│   │   ├── authentication.py    # Authentication model
│   │   ├── gan.py              # GAN architecture
│   │   └── evaluator.py        # Model evaluation
│   ├── training/
│   │   ├── train_auth.py       # Train authentication
│   │   └── train_gan.py        # Train GAN
│   ├── demo/
│   │   ├── demo_ui.py          # Main demo interface
│   │   ├── scenes.py           # Demo scenes
│   │   └── visualizer.py       # Visualization
│   └── utils/
│       ├── logger.py           # Logging
│       └── helpers.py          # Helper functions
├── run_demo.py                 # Main entry point
└── requirements.txt
```

## Workflow

### Bước 1: Thu Thập Data

Chạy data collector để thu thập keystroke patterns:

```bash
python -m src.data_collection.collector
```

- Gõ text "security2025" chính xác 100 lần
- Data sẽ được lưu vào `data/raw/keystroke_samples_*.json`

### Bước 2: Train Authentication Model

```bash
python -m src.training.train_auth --raw-data data/raw/keystroke_samples_*.json
```

Hoặc nếu đã có processed features:

```bash
python -m src.training.train_auth --data data/processed/features.npy
```

Model sẽ được lưu vào `data/models/auth_model.pkl`

### Bước 3: Train GAN Model

```bash
python -m src.training.train_gan --raw-data data/raw/keystroke_samples_*.json --epochs 1000
```

Hoặc với processed features:

```bash
python -m src.training.train_gan --data data/processed/features.npy --epochs 1000
```

Model sẽ được lưu vào `data/models/gan_model.h5`

### Bước 4: Chạy Demo

```bash
python run_demo.py
```

## Demo Scenarios

Demo bao gồm 4 scenes chính:

### Scene 1: Legitimate User Login
- Giảng viên gõ username/password
- Hệ thống xác thực thành công
- **Kết quả**: ✅ LOGIN SUCCESS

### Scene 2: Impostor Login Attempt
- Sinh viên gõ cùng password
- Pattern khác biệt
- **Kết quả**: ❌ LOGIN DENIED

### Scene 3: GAN Attack
- GAN generate fake patterns
- Thử bypass authentication
- **Kết quả**: ✅ BYPASS SUCCESS (một số attempts)

### Scene 4: Pattern Comparison
- Visualize và so sánh patterns
- Phân tích differences
- Thảo luận defense strategies

## Cấu Hình

Chỉnh sửa `config/settings.py` để thay đổi:

```python
TARGET_TEXT = "security2025"     # Text để gõ
NUM_SAMPLES = 100               # Số samples thu thập
AUTH_THRESHOLD = 0.85           # Ngưỡng authentication
EPOCHS_GAN = 1000              # Epochs cho GAN training
```

## CLI Options

### Train Authentication

```bash
python -m src.training.train_auth --help

Options:
  --data PATH           Path to processed features
  --raw-data PATH       Path to raw data JSON
  --threshold FLOAT     Authentication threshold (default: 0.85)
  --output PATH         Output model path
```

### Train GAN

```bash
python -m src.training.train_gan --help

Options:
  --data PATH           Path to processed features
  --raw-data PATH       Path to raw data JSON
  --epochs INT          Number of epochs (default: 1000)
  --batch-size INT      Batch size (default: 32)
  --latent-dim INT      Latent dimension (default: 100)
  --output PATH         Output model path
```

## Keystroke Features

Hệ thống trích xuất các features sau:

1. **Dwell Time**: Thời gian giữ mỗi phím
2. **Flight Time**: Thời gian giữa 2 lần nhấn phím
3. **Digraph Latency**: Thời gian giữa keydown của 2 phím liên tiếp
4. **Statistical Features**: Total time, average dwell, average flight, typing speed

## Model Architecture

### Authentication Model: One-Class SVM
- Kernel: RBF
- Nu: 0.1 (outlier fraction)
- Learns legitimate user's pattern

### GAN Architecture

**Generator:**
```
Input (100) → Dense(256) → LeakyReLU → BatchNorm → Dropout
            → Dense(512) → LeakyReLU → BatchNorm → Dropout
            → Dense(feature_dim) → Tanh
```

**Discriminator:**
```
Input (feature_dim) → Dense(512) → LeakyReLU → Dropout
                    → Dense(256) → LeakyReLU → Dropout
                    → Dense(1) → Sigmoid
```

## Troubleshooting

### TensorFlow Errors

Nếu gặp lỗi TensorFlow, thử:

```bash
pip install tensorflow==2.13.0
```

### GUI Không Hiển Thị

Đảm bảo tkinter được cài đặt:

```bash
# Ubuntu/Debian
sudo apt-get install python3-tk

# macOS (thường có sẵn)
brew install python-tk
```

### Import Errors

Đảm bảo chạy từ project root và PYTHONPATH đúng:

```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

## Testing

Chạy unit tests:

```bash
pytest tests/
```

Với coverage:

```bash
pytest --cov=src tests/
```

## Kết Luận Demo

**Key Takeaways:**
- ✅ Keystroke dynamics có thể xác thực người dùng
- ⚠️ GAN có thể học và giả mạo behavioral biometrics
- 🛡️ Cần multi-factor authentication
- 🔒 Behavioral biometrics không nên dùng standalone

**Defense Strategies:**
1. Combine với password + 2FA
2. Anomaly detection cho AI-generated patterns
3. Regular model updates với adversarial training
4. Monitor suspicious authentication patterns

## License

MIT License - For educational purposes only

## Contributors

- [Your Name]
- [Team Members]

## References

1. Keystroke Dynamics - Wikipedia
2. GAN (Goodfellow et al., 2014)
3. One-Class SVM (Schölkopf et al., 2001)

## Contact

- Email: [your-email]
- Issues: [GitHub Issues URL]

---

**⚠️ Disclaimer**: This project is for educational purposes only. Do not use for malicious activities.
