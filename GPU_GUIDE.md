# GPU Training Guide

Hướng dẫn sử dụng GPU cho training.

## 📊 Tổng Quan

### Authentication Model (One-Class SVM)
- ❌ **KHÔNG dùng GPU** (scikit-learn, CPU-only)
- Training: ~5-10 giây
- Không cần tối ưu

### GAN Model (TensorFlow)
- ✅ **TỰ ĐỘNG dùng GPU nếu có**
- Training nhanh hơn **5-10x** với GPU
- Đáng để optimize

## 🔍 Kiểm Tra GPU

### Windows

```batch
REM Chạy script check GPU
check_gpu.bat

REM Hoặc
python check_gpu.py
```

### Linux/Mac

```bash
python check_gpu.py
```

### Trong Python

```python
import tensorflow as tf

# Check GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"Found {len(gpus)} GPU(s)")
    for gpu in gpus:
        print(f"  - {gpu.name}")
else:
    print("No GPU found, using CPU")
```

## ⚙️ Cài Đặt GPU Support

### Windows

1. **Cài NVIDIA Driver**
   - Tải từ: https://www.nvidia.com/drivers
   - Hoặc dùng GeForce Experience

2. **Cài CUDA Toolkit**
   - Tải CUDA 11.8: https://developer.nvidia.com/cuda-11-8-0-download-archive
   - Hoặc CUDA 11.2
   - Thêm vào PATH

3. **Cài cuDNN**
   - Tải từ: https://developer.nvidia.com/cudnn
   - Giải nén vào thư mục CUDA

4. **Cài TensorFlow với GPU**
   ```batch
   pip uninstall tensorflow
   pip install tensorflow[and-cuda]
   ```

### Linux

```bash
# Ubuntu/Debian
# 1. Install NVIDIA driver
sudo ubuntu-drivers autoinstall

# 2. Install CUDA
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2004-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda

# 3. Install TensorFlow
pip install tensorflow[and-cuda]
```

## 📈 Performance Comparison

| Model | CPU (100 samples) | GPU (100 samples) | Speedup |
|-------|------------------|-------------------|---------|
| **Authentication** | ~10 seconds | ~10 seconds | 1x (CPU-only) |
| **GAN (500 epochs)** | ~12-15 minutes | ~2-3 minutes | **5x** |
| **GAN (1000 epochs)** | ~25-30 minutes | ~5-6 minutes | **5x** |
| **GAN (2000 epochs)** | ~50-60 minutes | ~10-12 minutes | **5x** |

## ⚡ Recommended Training Parameters

### Với GPU

```batch
REM Epochs cao hơn cho better quality
python -m src.training.train_gan ^
    --raw-data data\raw\sample_data.json ^
    --epochs 1500 ^
    --batch-size 64

REM Training time: ~7-8 phút
```

### Với CPU

```batch
REM Epochs thấp hơn cho faster training
python -m src.training.train_gan ^
    --raw-data data\raw\sample_data.json ^
    --epochs 500 ^
    --batch-size 32

REM Training time: ~12-15 phút
```

## 🎯 Tối Ưu GPU Memory

### Enable Memory Growth

TensorFlow mặc định chiếm toàn bộ GPU memory. Để tối ưu:

```python
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
```

**Đã tích hợp trong code!** Script `train_gan.py` tự động enable memory growth.

### Giới Hạn Memory

Nếu muốn giới hạn memory usage:

```python
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=4096)]  # 4GB
    )
```

### Mixed Precision Training

Để tăng tốc độ với GPU RTX series:

```python
from tensorflow.keras import mixed_precision

# Enable mixed precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
```

## 🛠️ Troubleshooting

### "Could not load dynamic library 'cudart64_XX.dll'"

**Solution:**
- Cài CUDA Toolkit đúng version
- Thêm CUDA bin folder vào PATH:
  - `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin`

### "Could not load dynamic library 'cudnn64_8.dll'"

**Solution:**
- Cài cuDNN
- Copy DLL files vào CUDA bin folder

### GPU Memory Out of Memory

**Solution:**
```batch
REM Giảm batch size
python -m src.training.train_gan --batch-size 16

REM Hoặc giảm epochs
python -m src.training.train_gan --epochs 300
```

### TensorFlow không detect GPU

**Check:**

```python
import tensorflow as tf
print("TensorFlow version:", tf.__version__)
print("GPU available:", tf.config.list_physical_devices('GPU'))
print("Built with CUDA:", tf.test.is_built_with_cuda())
```

**Solution:**
```batch
REM Reinstall TensorFlow with GPU support
pip uninstall tensorflow tensorflow-cpu
pip install tensorflow[and-cuda]
```

### Performance không tăng với GPU

**Possible reasons:**
1. Model quá nhỏ → Overhead lớn hơn benefit
2. Data loading bottleneck
3. Batch size quá nhỏ

**Solution:**
```batch
REM Tăng batch size
python -m src.training.train_gan --batch-size 128
```

## 📊 Monitor GPU Usage

### Windows

**Task Manager:**
- Ctrl + Shift + Esc
- Tab "Performance" → GPU

**NVIDIA-SMI:**
```batch
nvidia-smi
```

**Watch live:**
```batch
nvidia-smi -l 1
```

### Linux

```bash
# Once
nvidia-smi

# Watch
watch -n 1 nvidia-smi

# Or
nvtop  # Install: sudo apt install nvtop
```

## 🎓 Best Practices

1. **Always check GPU first:**
   ```batch
   check_gpu.bat
   ```

2. **Với GPU: Use higher epochs**
   - 1000-2000 epochs cho best quality
   - Training vẫn nhanh (~5-10 phút)

3. **Với CPU: Use lower epochs**
   - 300-500 epochs để tiết kiệm time
   - Quality vẫn acceptable cho demo

4. **Monitor training:**
   - Check nvidia-smi để ensure GPU được dùng
   - Theo dõi GPU memory usage

5. **Enable memory growth:**
   - Đã tự động enabled trong code
   - Tránh chiếm hết GPU memory

## 💡 FAQ

### Q: Laptop của tôi có GPU nhưng vẫn dùng CPU?

**A:**
- Laptop có thể có integrated GPU (Intel/AMD) → Không support CUDA
- Chỉ NVIDIA GPU mới support TensorFlow
- Check: `nvidia-smi` có hiện GPU không?

### Q: GPU usage chỉ ~30%, chưa full?

**A:**
- Bình thường! Small model không cần 100% GPU
- Tăng batch size để increase utilization
- Hoặc train nhiều models cùng lúc

### Q: Training với GPU vẫn chậm?

**A:**
- Check GPU temperature (thermal throttling?)
- Close các app khác đang dùng GPU
- Update NVIDIA driver

### Q: Có cần GPU cho demo không?

**A:**
- ❌ KHÔNG CẦN
- Demo chỉ load trained models (nhanh)
- GPU chỉ cần cho TRAINING

### Q: Batch size nên là bao nhiêu?

**A:**
- **GPU**: 64 hoặc 128
- **CPU**: 32 (default)
- Experiment để tìm best value

## 📝 Summary

| Aspect | CPU | GPU |
|--------|-----|-----|
| **Setup** | Dễ (mặc định) | Phức tạp (CUDA, cuDNN) |
| **Speed** | Chậm (~15-30 min) | Nhanh (~3-6 min) |
| **Epochs** | 300-500 | 1000-2000 |
| **Batch Size** | 32 | 64-128 |
| **Memory** | RAM | VRAM |
| **Cost** | Free | Requires NVIDIA GPU |
| **Demo** | OK | Better quality |

**Recommendation:**
- ✅ Nếu có NVIDIA GPU → Setup và dùng (đáng!)
- ✅ Nếu chỉ có CPU → Vẫn OK, giảm epochs

---

Happy Training! 🚀
