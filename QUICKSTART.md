# Quick Start Guide

Hướng dẫn nhanh để chạy demo.

## Setup (5 phút)

```bash
# 1. Clone và cài đặt
git clone <repo-url>
cd keystroke-gan-demo
pip install -r requirements.txt

# 2. Tạo thư mục cần thiết
mkdir -p data/{raw,processed,models} logs
```

## Option 1: Chạy Demo với Sample Data (Khuyến nghị cho lần đầu)

Nếu bạn muốn test nhanh mà không thu thập data:

```bash
# 1. Tạo sample data (simulated)
python -c "
import numpy as np
import json
import os
from datetime import datetime

# Generate sample keystroke data
target_text = 'security2025'
samples = []

for i in range(100):
    keypresses = []
    time = 0
    for j, char in enumerate(target_text):
        dwell = np.random.uniform(0.05, 0.15)
        keypresses.append({
            'char': char,
            'position': j,
            'keydown_time': time,
            'keyup_time': time + dwell
        })
        time += dwell + np.random.uniform(0.1, 0.2)

    samples.append({
        'sample_id': i+1,
        'timestamp': datetime.now().isoformat(),
        'keypresses': keypresses
    })

data = {
    'metadata': {
        'user_id': 'instructor',
        'target_text': target_text,
        'collection_date': datetime.now().isoformat(),
        'num_samples': 100
    },
    'samples': samples
}

os.makedirs('data/raw', exist_ok=True)
with open('data/raw/sample_data.json', 'w') as f:
    json.dump(data, f)

print('✓ Sample data created!')
"

# 2. Train models
python -m src.training.train_auth --raw-data data/raw/sample_data.json
python -m src.training.train_gan --raw-data data/raw/sample_data.json --epochs 500

# 3. Run demo
python run_demo.py
```

## Option 2: Thu Thập Data Thực Tế

```bash
# 1. Thu thập keystroke data
python -m src.data_collection.collector
# Gõ "security2025" đúng 100 lần

# 2. Train authentication model
python -m src.training.train_auth --raw-data data/raw/keystroke_samples_*.json

# 3. Train GAN model (có thể mất 10-30 phút)
python -m src.training.train_gan --raw-data data/raw/keystroke_samples_*.json --epochs 1000

# 4. Chạy demo
python run_demo.py
```

## Sử Dụng Demo UI

1. **Click "Load Models"** để load trained models
2. **Chọn Scene** (1-4)
3. **Click "Run Scene"** để xem demo

### Các Scenes:

- **Scene 1**: Legitimate user login → Success ✅
- **Scene 2**: Impostor attempt → Denied ❌
- **Scene 3**: GAN attack → Bypass! ⚠️
- **Scene 4**: Pattern visualization 📊

## Tips

### Nếu gặp lỗi "Model not found":
```bash
# Kiểm tra models đã được train chưa
ls -la data/models/
# Phải có: auth_model.pkl và gan_model_*.h5
```

### Nếu muốn re-train models:
```bash
# Xóa models cũ
rm data/models/*.pkl data/models/*.h5

# Train lại
python -m src.training.train_auth --raw-data data/raw/*.json
python -m src.training.train_gan --raw-data data/raw/*.json --epochs 1000
```

### Adjust Parameters:

Edit `config/settings.py`:
```python
AUTH_THRESHOLD = 0.85      # Tăng để khó pass hơn
EPOCHS_GAN = 1000         # Giảm cho training nhanh hơn
```

## Training Time

- **Authentication Model**: ~10 seconds
- **GAN Model**:
  - 500 epochs: ~5-10 phút
  - 1000 epochs: ~10-20 phút (khuyến nghị)
  - 2000 epochs: ~20-40 phút

## Minimum Requirements

- Python 3.8+
- 8GB RAM
- 1GB free disk space
- CPU: Any modern processor (GPU không bắt buộc)

## Next Steps

1. ✅ Chạy được demo
2. 📖 Đọc README.md để hiểu chi tiết
3. 🔧 Customize parameters trong config/settings.py
4. 📊 Experiment với different thresholds
5. 🎓 Sử dụng cho teaching/presentation

## Troubleshooting

**Import Error:**
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python run_demo.py
```

**TensorFlow Warning:**
```bash
# Ignore warnings, or install specific version:
pip install tensorflow==2.13.0
```

**GUI không show:**
```bash
# Linux
sudo apt-get install python3-tk

# Mac (usually pre-installed)
brew install python-tk
```

## Demo Script (15 phút)

**Phút 1-3**: Scene 1 - Real user success
**Phút 4-6**: Scene 2 - Impostor denied
**Phút 7-13**: Scene 3 - GAN attack (10 attempts)
**Phút 14-15**: Scene 4 - Visualization & discussion

**Kết luận**: GAN có thể learn behavioral patterns! → Need multi-factor auth!

---

Happy Demo! 🎉
