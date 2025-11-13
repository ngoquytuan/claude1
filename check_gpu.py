#!/usr/bin/env python3
"""
Check GPU availability for TensorFlow
"""

import sys

print("="*60)
print("GPU AVAILABILITY CHECK")
print("="*60)
print()

# Check TensorFlow
try:
    import tensorflow as tf
    print(f"✓ TensorFlow version: {tf.__version__}")
    print()
except ImportError:
    print("✗ TensorFlow not installed")
    print("  Install: pip install tensorflow")
    sys.exit(1)

# Check GPU devices
print("GPU Devices:")
print("-" * 60)

gpus = tf.config.list_physical_devices('GPU')

if gpus:
    print(f"✓ Found {len(gpus)} GPU(s):")
    for i, gpu in enumerate(gpus):
        print(f"  [{i}] {gpu.name}")

        # Get GPU details
        try:
            gpu_details = tf.config.experimental.get_device_details(gpu)
            if gpu_details:
                print(f"      Device: {gpu_details.get('device_name', 'Unknown')}")
                print(f"      Compute Capability: {gpu_details.get('compute_capability', 'Unknown')}")
        except:
            pass

    print()
    print("GPU Memory Configuration:")
    try:
        for gpu in gpus:
            # Try to get memory info
            tf.config.experimental.set_memory_growth(gpu, True)
            print(f"  Memory growth enabled for {gpu.name}")
    except RuntimeError as e:
        print(f"  Note: {e}")

    print()
    print("✓ TensorFlow WILL USE GPU for training!")
    print("  Expected speedup: 5-10x faster than CPU")

else:
    print("✗ No GPU found")
    print()
    print("TensorFlow will use CPU for training.")
    print()
    print("To enable GPU:")
    print("  1. Install CUDA Toolkit (11.2 or 11.8)")
    print("  2. Install cuDNN (8.1+)")
    print("  3. Install: pip install tensorflow[and-cuda]")
    print()
    print("Or use CPU-only version:")
    print("  pip install tensorflow-cpu")

print()
print("="*60)
print("SYSTEM INFO")
print("="*60)

# Python version
print(f"Python version: {sys.version.split()[0]}")

# CPU info
try:
    import platform
    print(f"Platform: {platform.system()} {platform.release()}")
    print(f"Processor: {platform.processor()}")
except:
    pass

# Memory info
try:
    import psutil
    mem = psutil.virtual_memory()
    print(f"RAM: {mem.total / (1024**3):.1f} GB (Available: {mem.available / (1024**3):.1f} GB)")
except:
    pass

print()

# Test computation
print("="*60)
print("QUICK PERFORMANCE TEST")
print("="*60)
print()

import time
import numpy as np

# Create test data
print("Testing matrix multiplication (1000x1000)...")
a = tf.random.normal((1000, 1000))
b = tf.random.normal((1000, 1000))

# Warmup
_ = tf.matmul(a, b)

# Benchmark
start = time.time()
for _ in range(10):
    _ = tf.matmul(a, b)
elapsed = time.time() - start

print(f"  10 iterations: {elapsed:.3f} seconds")
print(f"  Average: {elapsed/10:.3f} seconds per iteration")

if gpus:
    print(f"  Device: GPU ✓")
    print(f"  Performance: FAST 🚀")
else:
    print(f"  Device: CPU")
    print(f"  Performance: OK (GPU would be faster)")

print()
print("="*60)
print("RECOMMENDATIONS FOR YOUR SYSTEM")
print("="*60)
print()

if gpus:
    print("✓ Your system is GPU-ready!")
    print()
    print("Recommended training parameters:")
    print("  - Authentication: No change needed (CPU-only)")
    print("  - GAN epochs: 1000-2000 (will be fast on GPU)")
    print("  - Batch size: 64 or 128 (larger batch with GPU)")
    print()
    print("Estimated training times:")
    print("  - Authentication: ~10 seconds")
    print("  - GAN (1000 epochs): ~3-5 minutes on GPU")
else:
    print("Your system will use CPU.")
    print()
    print("Recommended training parameters:")
    print("  - Authentication: No change needed")
    print("  - GAN epochs: 300-500 (reduce for faster training)")
    print("  - Batch size: 32 (smaller batch for CPU)")
    print()
    print("Estimated training times:")
    print("  - Authentication: ~10 seconds")
    print("  - GAN (500 epochs): ~15-20 minutes on CPU")

print()
print("="*60)
