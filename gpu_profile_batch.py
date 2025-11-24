#!/usr/bin/env python3
"""
GPU profiling and benchmarking utility for batch segmentation.
Helps find optimal batch sizes and identify bottlenecks.
"""
import argparse
import time
import torch
import numpy as np
import cv2
from pathlib import Path
import psutil
import subprocess

try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False
    print("Warning: pynvml not available. Install with: pip install nvidia-ml-py3")
    print("GPU monitoring will be limited.")


class GPUMonitor:
    """Monitor GPU utilization and memory."""

    def __init__(self):
        self.nvml_available = NVML_AVAILABLE
        if self.nvml_available:
            try:
                pynvml.nvmlInit()
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            except Exception as e:
                print(f"NVML initialization failed: {e}")
                self.nvml_available = False

    def get_gpu_stats(self):
        """Get current GPU statistics."""
        if not self.nvml_available:
            return None

        try:
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
            power = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0  # Convert to Watts
            temp = pynvml.nvmlDeviceGetTemperature(self.handle, pynvml.NVML_TEMPERATURE_GPU)

            return {
                'memory_used_gb': mem_info.used / 1e9,
                'memory_total_gb': mem_info.total / 1e9,
                'memory_util_percent': (mem_info.used / mem_info.total) * 100,
                'gpu_util_percent': utilization.gpu,
                'power_watts': power,
                'temperature_c': temp,
            }
        except Exception as e:
            print(f"Error getting GPU stats: {e}")
            return None

    def __del__(self):
        if self.nvml_available:
            try:
                pynvml.nvmlShutdown()
            except:
                pass


def benchmark_sam_batch_sizes(weight_dir: str, test_image_path: str):
    """Benchmark different SAM batch sizes to find optimal setting."""
    from batch_segment_optimized import OptimizedSegmenter

    print("="*70)
    print("SAM Batch Size Benchmark")
    print("="*70)

    # Load test image
    test_image = cv2.imread(test_image_path)
    if test_image is None:
        print(f"Error: Could not load test image: {test_image_path}")
        return

    print(f"Test image shape: {test_image.shape}")

    # Create dummy boxes (simulate YOLO detections)
    # Typical dental image might have 10-30 teeth
    num_boxes_list = [10, 20, 30]

    # Test batch sizes
    batch_sizes = [4, 8, 16, 32, 64, 128]

    monitor = GPUMonitor()

    results = []

    for num_boxes in num_boxes_list:
        print(f"\nTesting with {num_boxes} detections:")
        print("-" * 70)

        # Create dummy boxes
        h, w = test_image.shape[:2]
        boxes = []
        for i in range(num_boxes):
            x1 = np.random.randint(0, w - 100)
            y1 = np.random.randint(0, h - 100)
            x2 = x1 + np.random.randint(50, 100)
            y2 = y1 + np.random.randint(50, 100)
            boxes.append([x1, y1, x2, y2])
        boxes = np.array(boxes)

        for batch_size in batch_sizes:
            try:
                # Initialize segmenter
                segmenter = OptimizedSegmenter(
                    weight_dir=weight_dir,
                    device='cuda',
                    sam_batch_size=batch_size,
                )

                # Warm-up
                _ = segmenter._sam_predict_optimized(boxes[:5], test_image)

                # Benchmark
                torch.cuda.synchronize()
                start_time = time.time()

                for _ in range(3):  # Run 3 times for average
                    _ = segmenter._sam_predict_optimized(boxes, test_image)
                    torch.cuda.synchronize()

                elapsed = (time.time() - start_time) / 3

                # Get GPU stats
                gpu_stats = monitor.get_gpu_stats()

                result = {
                    'num_boxes': num_boxes,
                    'batch_size': batch_size,
                    'time_seconds': elapsed,
                    'throughput': num_boxes / elapsed,
                }

                if gpu_stats:
                    result.update(gpu_stats)

                results.append(result)

                print(f"  Batch size {batch_size:3d}: {elapsed:.3f}s "
                      f"({result['throughput']:.1f} boxes/s)", end='')

                if gpu_stats:
                    print(f" | GPU: {gpu_stats['gpu_util_percent']:3.0f}% "
                          f"| Mem: {gpu_stats['memory_used_gb']:.2f}GB "
                          f"({gpu_stats['memory_util_percent']:.0f}%)")
                else:
                    print()

                # Clean up
                del segmenter
                torch.cuda.empty_cache()

            except torch.cuda.OutOfMemoryError:
                print(f"  Batch size {batch_size:3d}: OOM (out of memory)")
            except Exception as e:
                print(f"  Batch size {batch_size:3d}: Error - {e}")

    # Find optimal batch size
    print("\n" + "="*70)
    print("Recommendations:")
    print("="*70)

    if results:
        # Group by num_boxes
        for num_boxes in num_boxes_list:
            box_results = [r for r in results if r['num_boxes'] == num_boxes]
            if box_results:
                # Find fastest
                best = max(box_results, key=lambda x: x['throughput'])
                print(f"\nFor {num_boxes} teeth:")
                print(f"  Optimal batch size: {best['batch_size']}")
                print(f"  Processing time: {best['time_seconds']:.3f}s")
                print(f"  Throughput: {best['throughput']:.1f} boxes/s")


def profile_full_pipeline(weight_dir: str, test_image_path: str, view: str = 'front'):
    """Profile the entire segmentation pipeline."""
    from batch_segment_optimized import OptimizedSegmenter

    print("="*70)
    print("Full Pipeline Profiling")
    print("="*70)

    monitor = GPUMonitor()

    # Initialize segmenter
    print("\nInitializing models...")
    init_start = time.time()
    segmenter = OptimizedSegmenter(
        weight_dir=weight_dir,
        device='cuda',
        sam_batch_size=32,
    )
    init_time = time.time() - init_start
    print(f"Initialization time: {init_time:.2f}s")

    # Load test image
    test_image = cv2.imread(test_image_path)
    if test_image is None:
        print(f"Error: Could not load test image: {test_image_path}")
        return

    print(f"Test image shape: {test_image.shape}")

    # Profile components
    print("\nProfiling components:")
    print("-" * 70)

    # I/O
    io_start = time.time()
    for _ in range(10):
        _ = cv2.imread(test_image_path)
    io_time = (time.time() - io_start) / 10
    print(f"Image loading (I/O):     {io_time*1000:.1f}ms")

    # Full prediction
    torch.cuda.synchronize()
    pred_start = time.time()
    mask = segmenter.predict(test_image, view, conf_threshold=0.01)
    torch.cuda.synchronize()
    pred_time = time.time() - pred_start
    print(f"Full prediction:         {pred_time*1000:.1f}ms")

    # Save time
    save_start = time.time()
    for _ in range(10):
        cv2.imwrite('/tmp/test_mask.png', mask)
    save_time = (time.time() - save_start) / 10
    print(f"Mask saving (I/O):       {save_time*1000:.1f}ms")

    total_per_image = io_time + pred_time + save_time
    print(f"\nTotal per image:         {total_per_image*1000:.1f}ms")
    print(f"Theoretical throughput:  {1/total_per_image:.1f} images/sec")

    # GPU stats
    gpu_stats = monitor.get_gpu_stats()
    if gpu_stats:
        print(f"\nGPU Statistics:")
        print(f"  Utilization:  {gpu_stats['gpu_util_percent']:.0f}%")
        print(f"  Memory Used:  {gpu_stats['memory_used_gb']:.2f}GB / {gpu_stats['memory_total_gb']:.2f}GB")
        print(f"  Memory Util:  {gpu_stats['memory_util_percent']:.0f}%")
        print(f"  Power:        {gpu_stats['power_watts']:.1f}W")
        print(f"  Temperature:  {gpu_stats['temperature_c']:.0f}°C")

    # Bottleneck analysis
    print(f"\nBottleneck Analysis:")
    print(f"  I/O time:        {(io_time + save_time)/total_per_image*100:.1f}%")
    print(f"  GPU compute:     {pred_time/total_per_image*100:.1f}%")

    if io_time + save_time > pred_time:
        print("\n  ⚠ I/O is the bottleneck! Consider:")
        print("    - Using faster storage (SSD/NVMe)")
        print("    - Preloading images into RAM")
        print("    - Multi-threaded I/O")
    else:
        print("\n  ⚠ GPU compute is the bottleneck! Consider:")
        print("    - Increasing SAM batch size")
        print("    - Using mixed precision (FP16)")
        print("    - Reducing image resolution if acceptable")


def main():
    parser = argparse.ArgumentParser(description='GPU profiling utility for batch segmentation')
    parser.add_argument(
        '--weight-dir',
        type=str,
        default='/home/itamar/git/mantis/pretrained_models/SegmentAnyToothWeights',
        help='Path to model weights'
    )
    parser.add_argument(
        '--test-image',
        type=str,
        required=True,
        help='Path to test image'
    )
    parser.add_argument(
        '--benchmark-sam',
        action='store_true',
        help='Benchmark different SAM batch sizes'
    )
    parser.add_argument(
        '--profile-pipeline',
        action='store_true',
        help='Profile full segmentation pipeline'
    )
    parser.add_argument(
        '--view',
        type=str,
        default='front',
        choices=['upper', 'lower', 'left', 'right', 'front'],
        help='View type for testing'
    )

    args = parser.parse_args()

    # Check CUDA availability
    if not torch.cuda.is_available():
        print("Error: CUDA is not available!")
        return

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"PyTorch Version: {torch.__version__}")
    print()

    if args.benchmark_sam:
        benchmark_sam_batch_sizes(args.weight_dir, args.test_image)

    if args.profile_pipeline:
        profile_full_pipeline(args.weight_dir, args.test_image, args.view)

    if not args.benchmark_sam and not args.profile_pipeline:
        print("Please specify --benchmark-sam or --profile-pipeline")


if __name__ == "__main__":
    main()
