#!/usr/bin/env python3
"""Quick test to verify GPU is being used."""
import torch
from ultralytics import YOLO
import time

print("="*60)
print("GPU Quick Test")
print("="*60)

# Check CUDA
print(f"\n1. CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Test YOLO
print(f"\n2. Loading YOLO model...")
model = YOLO('/home/itamar/git/mantis/pretrained_models/SegmentAnyToothWeights/segmentanytooth_yolo11_front.pt')

# Check device
print(f"   Model device before .to('cuda'): {next(model.model.parameters()).device}")

# Try to move to GPU
if torch.cuda.is_available():
    model.to('cuda')
    print(f"   Model device after .to('cuda'): {next(model.model.parameters()).device}")

# Test prediction
print(f"\n3. Testing prediction speed...")
import cv2
import numpy as np

# Create dummy image
dummy = np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)

# Warm up
_ = model.predict(dummy, save=False, verbose=False)

# Time it
torch.cuda.synchronize()
start = time.time()
for _ in range(10):
    _ = model.predict(dummy, save=False, verbose=False)
    torch.cuda.synchronize()
elapsed = (time.time() - start) / 10

print(f"   Average inference time: {elapsed*1000:.1f}ms")
print(f"   Throughput: {1/elapsed:.1f} images/sec")

if elapsed > 0.5:
    print(f"\n⚠️  WARNING: YOLO is too slow! Likely running on CPU!")
    print(f"   Expected: <50ms per image on GPU")
    print(f"   Got: {elapsed*1000:.0f}ms per image")
else:
    print(f"\n✅ YOLO speed looks good (GPU)")

print(f"\n4. GPU Memory after test:")
if torch.cuda.is_available():
    print(f"   Allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
    print(f"   Cached: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
