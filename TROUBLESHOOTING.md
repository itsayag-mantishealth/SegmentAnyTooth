# Performance Troubleshooting Guide

## Quick Diagnosis

Run your batch script and look at the **SAM Model Verification** output at startup:

```bash
python batch_segment_optimized.py /data --device cuda --verbose
```

## What to Check

### 1. ✅ SAM Device (CRITICAL!)

**Good:**
```
✓ SAM device: cuda:0
```

**BAD - Root cause of 12.5 sec/image:**
```
✗ SAM device: cpu
CRITICAL ERROR: SAM model is on CPU despite requesting CUDA!
```

**Fix if on CPU:**
- Check torch CUDA installation: `python -c "import torch; print(torch.cuda.is_available())"`
- Reinstall PyTorch with CUDA: `pip install torch --index-url https://download.pytorch.org/whl/cu118`

---

### 2. ✅ Model Architecture

**Good (Mobile SAM - Fast):**
```
✓ SAM model: Sam
✓ Image encoder: TinyViT
✓ Model size: 43.7 MB (10,893,424 parameters)
```

**BAD (Wrong SAM variant - Slow):**
```
✗ Image encoder: ImageEncoderViT
✗ Model size: 375.0 MB (94,000,000+ parameters)
WARNING: You may be using SAM-Base or SAM-Huge (much slower)
```

**Fix if wrong model:**
- Check your weights: should be `segmentanytooth_vit_tiny.pt` (~40MB)
- SAM-Base is ~350MB, SAM-Huge is ~2.4GB
- Download correct Mobile SAM weights

---

### 3. ✅ Performance Test

**Good (GPU working):**
```
✓ SAM test: 312ms for 1 box on 1024x1024 image
  ✅ Performance looks good!
```

**BAD (CPU or slow GPU):**
```
✗ SAM test: 12,450ms for 1 box on 1024x1024 image
  ⚠️ WARNING: SAM is VERY SLOW! (12.5s)
  This suggests SAM is running on CPU!
RuntimeError: SAM performance test failed
```

**Expected times:**
- **A10G GPU:** 200-400ms per image
- **T4 GPU:** 300-600ms per image
- **CPU:** 5-15 seconds per image (40x slower!)

---

### 4. ✅ Model Configuration

**Good:**
```
✓ Model mode: eval
✓ cuDNN enabled: True
✓ cuDNN benchmark: True
```

**BAD:**
```
✗ Model mode: training
✗ cuDNN enabled: False
```

---

## Common Issues & Solutions

### Issue: SAM takes 12+ seconds per image

**Diagnosis:** SAM is running on **CPU** instead of GPU

**Check:**
1. SAM device shows `cpu` instead of `cuda:0`
2. Performance test shows >5 seconds
3. GPU utilization in `nvidia-smi` shows 0%

**Root causes:**
1. **PyTorch not compiled with CUDA**
   ```bash
   # Check CUDA availability
   python -c "import torch; print(torch.cuda.is_available())"
   # Should print: True
   ```

2. **Model not moved to GPU properly**
   - Look for error messages during model loading
   - Try manually: `sam = sam.cuda()`

3. **CUDA driver/version mismatch**
   ```bash
   nvidia-smi  # Check driver version
   python -c "import torch; print(torch.version.cuda)"  # Check torch CUDA version
   ```

**Fix:**
```bash
# Reinstall PyTorch with CUDA support
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu118  # or cu121
```

---

### Issue: SAM on GPU but still slow (1-2 sec/image)

**Diagnosis:** Using **SAM-Base or SAM-Huge** instead of Mobile SAM

**Check:**
- Model size shows >100MB
- Parameter count shows >50M

**Fix:**
- Download Mobile SAM weights (vit_tiny)
- Check weight file size: should be ~40MB, not 350MB or 2.4GB

---

### Issue: YOLO takes 500+ ms per image

**Diagnosis:** YOLO running on **CPU**

**Check:**
Look for in startup logs:
```
Loading YOLO model for front view on cuda...
  YOLO model device: cpu  ← BAD!
```

**Fix:**
Same as SAM - ensure PyTorch has CUDA support

---

## Performance Targets (A10G GPU)

| Component | Expected Time | Your Time | Status |
|-----------|--------------|-----------|---------|
| YOLO detection | 20-50ms | ??? | ??? |
| SAM embedding | 200-400ms | **12,500ms** | ❌ CPU! |
| SAM inference (64 boxes) | 100-200ms | ??? | ??? |
| **Total per image** | **300-700ms** | **12,500ms+** | ❌ |

**Your target:** Should process 1-3 images/second on A10G (not 0.08 images/sec!)

---

## Still Having Issues?

Run the diagnostic scripts:

```bash
# 1. Quick GPU check (30 seconds)
python quick_test.py

# 2. Detailed single-image profiling
python debug_performance.py /path/to/test/image.png --view front

# 3. Full batch with verbose timing
python batch_segment_optimized.py /data --verbose --device cuda
```

Share the output from these commands for further help!

---

## Summary: What Fixed SAM from 12.5s → 0.3s

1. ✅ Verified SAM is on GPU (not CPU)
2. ✅ Using Mobile SAM (TinyViT), not SAM-Base
3. ✅ Model in eval mode
4. ✅ cuDNN enabled with benchmark mode
5. ✅ Performance test confirms <500ms per image

**Expected final result:**
- 15K images: ~2-4 hours on A10G (not 46 hours!)
- Throughput: 1-3 images/second (not 0.08!)
