#!/usr/bin/env python3
"""
Performance debugging script to identify bottlenecks.
Run this on a small batch to see where time is spent.
"""
import time
import torch
import cv2
import numpy as np
from pathlib import Path
import argparse

from batch_segment_optimized import OptimizedSegmenter


def time_function(name, func, *args, **kwargs):
    """Time a function and return its result."""
    start = time.time()
    result = func(*args, **kwargs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.time() - start
    print(f"  {name}: {elapsed*1000:.1f}ms")
    return result, elapsed


def debug_single_image(image_path, weight_dir, view='front', conf_threshold=0.01):
    """Debug processing of a single image with detailed timing."""
    print(f"\n{'='*60}")
    print(f"Debugging: {image_path}")
    print(f"{'='*60}")

    total_start = time.time()

    # Check CUDA
    print(f"\n1. CUDA Check:")
    print(f"  CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA Device: {torch.cuda.get_device_name(0)}")
        print(f"  Initial GPU Memory: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")

    # Initialize segmenter
    print(f"\n2. Model Initialization:")
    init_start = time.time()
    segmenter = OptimizedSegmenter(
        weight_dir=weight_dir,
        device='cuda',
        sam_batch_size=64,
    )
    init_time = time.time() - init_start
    print(f"  Total init time: {init_time*1000:.1f}ms")

    # Check models are on GPU
    print(f"\n3. Model Device Check:")
    print(f"  SAM device: {segmenter.sam.device}")
    if torch.cuda.is_available():
        print(f"  GPU Memory after init: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")

    # Load image
    print(f"\n4. Image Loading:")
    image, load_time = time_function("Load image", cv2.imread, str(image_path))
    if image is None:
        print(f"  ERROR: Failed to load image!")
        return
    print(f"  Image shape: {image.shape}")

    # Get YOLO model
    print(f"\n5. YOLO Model Loading:")
    yolo, yolo_load_time = time_function("Get YOLO model", segmenter.get_yolo_model, view)

    # Check YOLO device
    print(f"  YOLO device: {next(yolo.model.parameters()).device}")

    # YOLO prediction
    print(f"\n6. YOLO Detection:")
    yolo_start = time.time()
    results = yolo.predict(image, save=False, conf=conf_threshold, verbose=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    yolo_time = time.time() - yolo_start
    print(f"  YOLO inference: {yolo_time*1000:.1f}ms")

    r = results[0]
    if r.boxes is None or len(r.boxes) == 0:
        print(f"  WARNING: No teeth detected!")
        return

    num_boxes = len(r.boxes)
    print(f"  Detected: {num_boxes} teeth")

    # Get boxes
    print(f"\n7. Box Processing:")
    boxes = r.boxes.xyxy.squeeze(0).cpu().numpy()
    clss = r.boxes.cls.squeeze(0).cpu().numpy().astype(np.int32)
    print(f"  Boxes extracted: {len(boxes)}")

    # Convert image
    print(f"\n8. Image Preprocessing:")
    image_rgb, rgb_time = time_function("RGB conversion", cv2.cvtColor, image, cv2.COLOR_BGR2RGB)

    # SAM prediction
    print(f"\n9. SAM Segmentation:")
    print(f"  Processing {num_boxes} boxes with batch_size=64")
    sam_start = time.time()

    # Break down SAM timing
    print(f"\n  9a. SAM Image Embedding:")
    from sam import SamMobilePredictor
    predictor = SamMobilePredictor(segmenter.sam)

    embed_start = time.time()
    predictor.set_image(image_rgb)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    embed_time = time.time() - embed_start
    print(f"    Image embedding: {embed_time*1000:.1f}ms")

    # SAM inference per batch
    print(f"\n  9b. SAM Box Inference (batched):")
    batch_size = 64
    batch_boxes = np.split(boxes, range(batch_size, len(boxes), batch_size))

    total_sam_inference = 0
    for i, box_batch in enumerate(batch_boxes):
        batch_start = time.time()

        transformed_boxes = predictor.transform.apply_boxes_torch(
            torch.tensor(box_batch), image_rgb.shape[:2]
        ).to(segmenter.device)

        sam_masks, _, _ = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
            hq_token_only=False,
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        batch_time = time.time() - batch_start
        total_sam_inference += batch_time
        print(f"    Batch {i+1} ({len(box_batch)} boxes): {batch_time*1000:.1f}ms")

    total_sam = time.time() - sam_start
    print(f"\n  Total SAM time: {total_sam*1000:.1f}ms")
    print(f"    Embedding: {embed_time*1000:.1f}ms ({embed_time/total_sam*100:.1f}%)")
    print(f"    Inference: {total_sam_inference*1000:.1f}ms ({total_sam_inference/total_sam*100:.1f}%)")

    # Total time
    total_time = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"TOTAL TIME: {total_time:.2f}s ({total_time*1000:.0f}ms)")
    print(f"{'='*60}")

    # Breakdown
    print(f"\nTime Breakdown:")
    print(f"  Model init:      {init_time*1000:6.0f}ms ({init_time/total_time*100:5.1f}%)")
    print(f"  Image load:      {load_time*1000:6.0f}ms ({load_time/total_time*100:5.1f}%)")
    print(f"  YOLO:            {yolo_time*1000:6.0f}ms ({yolo_time/total_time*100:5.1f}%)")
    print(f"  SAM:             {total_sam*1000:6.0f}ms ({total_sam/total_time*100:5.1f}%)")
    print(f"    - Embedding:   {embed_time*1000:6.0f}ms")
    print(f"    - Inference:   {total_sam_inference*1000:6.0f}ms")

    if torch.cuda.is_available():
        print(f"\nGPU Memory:")
        print(f"  Allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        print(f"  Cached: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")

    # Calculate throughput
    print(f"\nProjected Throughput:")
    print(f"  Per image: {total_time:.2f}s")
    print(f"  Images/sec: {1/total_time:.2f}")
    print(f"  15K images would take: {15000*total_time/3600:.1f} hours")


def main():
    parser = argparse.ArgumentParser(description='Debug performance bottlenecks')
    parser.add_argument('image_path', type=str, help='Path to test image')
    parser.add_argument('--weight-dir', type=str,
                       default='/home/itamar/git/mantis/pretrained_models/SegmentAnyToothWeights',
                       help='Path to model weights')
    parser.add_argument('--view', type=str, default='front',
                       choices=['upper', 'lower', 'left', 'right', 'front'],
                       help='View type')
    parser.add_argument('--conf-threshold', type=float, default=0.01,
                       help='YOLO confidence threshold')

    args = parser.parse_args()

    debug_single_image(
        args.image_path,
        args.weight_dir,
        args.view,
        args.conf_threshold
    )


if __name__ == "__main__":
    main()
