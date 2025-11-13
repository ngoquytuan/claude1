# Quick Start Guide - Windows

Hướng dẫn nhanh cho Windows.

## Setup (5 phút)

### 1. Cài đặt Python

Tải Python 3.8+ từ https://www.python.org/downloads/

**Quan trọng**: Tick "Add Python to PATH" khi cài đặt!

### 2. Clone và Setup Project

```powershell
# Clone repo
git clone <repo-url>
cd keystroke-gan-demo

# Tạo virtual environment (khuyến nghị)
python -m venv venv

# Activate virtual environment
venv\Scripts\activate

# Cài đặt dependencies
pip install -r requirements.txt
```

## Option 1: Chạy Demo với Sample Data (Khuyến nghị)

### Tạo Sample Data

Tạo file `create_sample_data.py`:

```python
import numpy as np
import json
import os
from datetime import datetime

# Generate sample keystroke data
target_text = 'security2025'
samples = []

print("Generating sample data...")

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

print('Sample data created successfully!')
print('Location: data/raw/sample_data.json')
```

### Chạy Scripts

```powershell
# 1. Tạo sample data
python create_sample_data.py

# 2. Train authentication model
python -m src.training.train_auth --raw-data data\raw\sample_data.json

# 3. Train GAN model (có thể mất 5-15 phút)
python -m src.training.train_gan --raw-data data\raw\sample_data.json --epochs 500

# 4. Chạy demo
python run_demo.py
```

## Option 2: Thu Thập Data Thực

```powershell
# 1. Thu thập keystroke data
python -m src.data_collection.collector
# Gõ "security2025" đúng 100 lần

# 2. Train authentication model
python -m src.training.train_auth --raw-data data\raw\keystroke_samples_*.json

# 3. Train GAN model
python -m src.training.train_gan --raw-data data\raw\keystroke_samples_*.json --epochs 1000

# 4. Chạy demo
python run_demo.py
```

## Tạo Batch Scripts

### setup.bat

```batch
@echo off
echo Creating virtual environment...
python -m venv venv

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Installing dependencies...
pip install -r requirements.txt

echo Creating directories...
if not exist "data\raw" mkdir "data\raw"
if not exist "data\processed" mkdir "data\processed"
if not exist "data\models" mkdir "data\models"
if not exist "logs" mkdir "logs"

echo Setup complete!
pause
```

### create_sample_data.bat

```batch
@echo off
call venv\Scripts\activate.bat
python create_sample_data.py
pause
```

### train_models.bat

```batch
@echo off
call venv\Scripts\activate.bat

echo Training authentication model...
python -m src.training.train_auth --raw-data data\raw\sample_data.json

echo Training GAN model...
python -m src.training.train_gan --raw-data data\raw\sample_data.json --epochs 500

echo Training complete!
pause
```

### run_demo.bat

```batch
@echo off
call venv\Scripts\activate.bat
python run_demo.py
```

## Workflow Nhanh (Windows)

### Cách 1: Dùng Batch Files

```powershell
# 1. Setup
setup.bat

# 2. Tạo sample data
create_sample_data.bat

# 3. Train models
train_models.bat

# 4. Run demo
run_demo.bat
```

### Cách 2: PowerShell

```powershell
# All in one
.\venv\Scripts\activate
python create_sample_data.py
python -m src.training.train_auth --raw-data data\raw\sample_data.json
python -m src.training.train_gan --raw-data data\raw\sample_data.json --epochs 500
python run_demo.py
```

## Troubleshooting Windows

### Lỗi: "python not found"

```powershell
# Thêm Python vào PATH
# Control Panel → System → Advanced → Environment Variables
# Thêm vào PATH: C:\Python39\  và C:\Python39\Scripts\
```

### Lỗi: "No module named tkinter"

Tkinter thường có sẵn với Python trên Windows. Nếu không có:

```powershell
# Cài lại Python và tick "tcl/tk and IDLE" trong installer
```

### Lỗi: "Cannot activate virtual environment"

```powershell
# Nếu gặp lỗi execution policy
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Sau đó activate lại
venv\Scripts\activate
```

### Lỗi TensorFlow

```powershell
# Nếu TensorFlow không cài được
pip install --upgrade pip
pip install tensorflow==2.13.0

# Hoặc dùng CPU version
pip install tensorflow-cpu==2.13.0
```

### Lỗi: "Permission denied"

Chạy PowerShell/Command Prompt as Administrator

### Kiểm Tra Installation

```powershell
python --version
# Should show: Python 3.8+

pip list
# Should show: tensorflow, numpy, scikit-learn, etc.

python -c "import tensorflow; print(tensorflow.__version__)"
# Should show: 2.13.0 or similar
```

## File Paths trên Windows

Code đã được viết cross-platform với `os.path.join()`, nên tự động handle Windows paths:

```python
# Tự động convert thành:
# Windows: data\raw\file.json
# Linux/Mac: data/raw/file.json
```

## Performance Tips Windows

1. **Tắt Windows Defender** khi training (tạm thời):
   - Settings → Update & Security → Windows Security → Virus & threat protection
   - Tạm tắt Real-time protection khi training

2. **Close các app không cần thiết** để có nhiều RAM hơn

3. **Dùng epochs thấp hơn** cho testing:
   ```powershell
   python -m src.training.train_gan --epochs 200  # Nhanh hơn
   ```

## Demo Workflow (5 phút setup)

```powershell
# 1. Open PowerShell trong project folder
cd keystroke-gan-demo

# 2. Activate virtual environment
venv\Scripts\activate

# 3. Tạo và train với sample data
python create_sample_data.py
python -m src.training.train_auth --raw-data data\raw\sample_data.json
python -m src.training.train_gan --raw-data data\raw\sample_data.json --epochs 300

# 4. Run demo
python run_demo.py
```

## System Requirements

- Windows 10/11
- Python 3.8+
- 8GB RAM
- 2GB free disk space
- Intel/AMD CPU (GPU không bắt buộc)

## Video Tutorial

Xem video hướng dẫn chi tiết tại: [Link to video]

---

Happy Demo on Windows! 🎉
