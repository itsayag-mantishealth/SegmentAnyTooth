#!/usr/bin/env python3
"""
Highly optimized batch processing with parallel I/O for maximum throughput.
Uses threading for I/O operations while keeping GPU inference in main thread.
"""
import os
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import argparse
import cv2
import numpy as np
import torch
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
import threading

from batch_segment_optimized import (
    OptimizedSegmenter,
    parse_camera_data,
    robust_view_classification,
    find_images_with_cameras,
    get_model_path,
    LEFT_CLASSES,
    suppress_stdout,
)


class ParallelImageLoader:
    """Loads images in parallel using thread pool."""

    def __init__(self, num_workers: int = 4):
        self.num_workers = num_workers
        self.executor = ThreadPoolExecutor(max_workers=num_workers)

    def load_image(self, path: Path) -> Optional[Tuple[Path, np.ndarray]]:
        """Load a single image."""
        try:
            image = cv2.imread(str(path))
            if image is None:
                return None
            return (path, image)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return None

    def load_batch(self, paths: List[Path]) -> List[Tuple[Path, np.ndarray]]:
        """Load a batch of images in parallel."""
        futures = [self.executor.submit(self.load_image, path) for path in paths]
        results = []
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                results.append(result)
        # Sort by original order
        path_to_image = dict(results)
        return [(path, path_to_image[path]) for path in paths if path in path_to_image]

    def shutdown(self):
        self.executor.shutdown(wait=True)


class ParallelMaskSaver:
    """Saves masks in parallel using thread pool."""

    def __init__(self, num_workers: int = 4):
        self.num_workers = num_workers
        self.executor = ThreadPoolExecutor(max_workers=num_workers)

    def save_mask(self, output_path: Path, mask: np.ndarray) -> bool:
        """Save a single mask."""
        try:
            cv2.imwrite(str(output_path), mask)
            return True
        except Exception as e:
            print(f"Error saving {output_path}: {e}")
            return False

    def save_batch(self, output_paths: List[Path], masks: List[np.ndarray]) -> int:
        """Save a batch of masks in parallel."""
        futures = [
            self.executor.submit(self.save_mask, output_path, mask)
            for output_path, mask in zip(output_paths, masks)
        ]
        successful = sum(1 for future in as_completed(futures) if future.result())
        return successful

    def shutdown(self):
        self.executor.shutdown(wait=True)


def process_batch_parallel(
    image_paths: List[Path],
    output_paths: List[Path],
    views: List[str],
    segmenter: OptimizedSegmenter,
    conf_threshold: float,
    loader: ParallelImageLoader,
    saver: ParallelMaskSaver,
) -> int:
    """
    Process a batch of images with parallel I/O.

    Pipeline:
    1. Load all images in parallel (threaded)
    2. Run YOLO detection (GPU, sequential)
    3. Run SAM inference (GPU, batched)
    4. Save all masks in parallel (threaded)

    Args:
        image_paths: List of input image paths
        output_paths: List of output mask paths
        views: List of view types (should all be the same)
        segmenter: Optimized segmenter instance
        conf_threshold: Confidence threshold
        loader: Parallel image loader
        saver: Parallel mask saver

    Returns:
        Number of successfully processed images
    """
    if len(image_paths) == 0:
        return 0

    # Assume all images in batch have same view
    view = views[0]

    # Get YOLO model for this view
    yolo = segmenter.get_yolo_model(view)
    should_flip = view == "left"
    names = yolo.model.names if not should_flip else LEFT_CLASSES

    # Step 1: Load all images in parallel
    loaded_images = loader.load_batch(image_paths)

    if len(loaded_images) == 0:
        return 0

    # Create mapping back to output paths
    path_to_output = {img_path: out_path for img_path, out_path in zip(image_paths, output_paths)}

    # Step 2: Run YOLO detections and prepare for SAM
    batch_data = []

    for img_path, image in loaded_images:
        try:
            output_path = path_to_output[img_path]

            if should_flip:
                image = cv2.flip(image, 1)

            # YOLO detection
            with suppress_stdout():
                r = yolo.predict(
                    image,
                    save=False,
                    save_txt=False,
                    save_conf=False,
                    save_crop=False,
                    project=None,
                    conf=conf_threshold,
                    verbose=False,
                )[0]

            # Skip if no detections
            if r.boxes is None or len(r.boxes) == 0:
                # Queue empty mask for saving
                empty_mask = np.zeros(image.shape[:2], dtype=np.uint8)
                batch_data.append({
                    'mask': empty_mask,
                    'output_path': output_path,
                    'has_detections': False,
                })
                continue

            # Get boxes and classes
            boxes = r.boxes.xyxy.squeeze(0).cpu().numpy()
            clss = r.boxes.cls.squeeze(0).cpu().numpy().astype(np.int32)

            # Sort by class id
            sort_ids = np.argsort(clss)
            clss = clss[sort_ids]
            boxes = boxes[sort_ids]

            if should_flip:
                image_width = image.shape[1]
                image = cv2.flip(image, 1)
                flipped_boxes = boxes.copy()
                flipped_boxes[:, [0, 2]] = image_width - flipped_boxes[:, [2, 0]]
                boxes = flipped_boxes

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            batch_data.append({
                'image': image_rgb,
                'boxes': boxes,
                'clss': clss,
                'output_path': output_path,
                'has_detections': True,
            })

        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    if len(batch_data) == 0:
        return 0

    # Step 3: Run SAM inference for images with detections
    masks_to_save = []
    output_paths_to_save = []

    for data in batch_data:
        if not data['has_detections']:
            masks_to_save.append(data['mask'])
            output_paths_to_save.append(data['output_path'])
            continue

        try:
            # Get SAM predictions
            sam_masks = segmenter._sam_predict_optimized(data['boxes'], data['image'])

            # Build segmentation mask
            predict_mask = np.zeros(data['image'].shape[:2], dtype=np.uint8)
            for cls_id, current_mask in zip(data['clss'], sam_masks):
                fdi_tooth_name = int(names[cls_id][-2:])
                predict_mask[current_mask == 1] = fdi_tooth_name

            masks_to_save.append(predict_mask)
            output_paths_to_save.append(data['output_path'])

        except Exception as e:
            print(f"Error in SAM for {data['output_path']}: {e}")

    # Step 4: Save all masks in parallel
    if len(masks_to_save) > 0:
        successful = saver.save_batch(output_paths_to_save, masks_to_save)
        return successful

    return 0


def main():
    parser = argparse.ArgumentParser(
        description='Parallel batch processing with optimized I/O for maximum A10G utilization'
    )
    parser.add_argument(
        'input_dir',
        type=str,
        help='Root directory containing scan folders with camera_data.txt'
    )
    parser.add_argument(
        '--weight-dir',
        type=str,
        default='/home/itamar/git/mantis/pretrained_models/SegmentAnyToothWeights',
        help='Path to model weights directory'
    )
    parser.add_argument(
        '--conf-threshold',
        type=float,
        default=0.01,
        help='Confidence threshold for YOLO detection (default: 0.01)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to use (default: cuda)'
    )
    parser.add_argument(
        '--sam-batch-size',
        type=int,
        default=64,
        help='SAM batch size (default: 64 for A10G). Use 32 for T4, 128+ for A100.'
    )
    parser.add_argument(
        '--image-batch-size',
        type=int,
        default=16,
        help='Number of images to process together (default: 16 for A10G with parallel I/O)'
    )
    parser.add_argument(
        '--io-workers',
        type=int,
        default=8,
        help='Number of I/O worker threads for loading/saving (default: 8)'
    )
    parser.add_argument(
        '--output-suffix',
        type=str,
        default='_mask',
        help='Output file suffix (default: _mask)'
    )
    parser.add_argument(
        '--skip-existing',
        action='store_true',
        help='Skip images that already have output masks (allows resuming interrupted runs)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show detailed timing for each batch'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='List files without processing'
    )

    args = parser.parse_args()

    # Validate input directory
    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory does not exist: {args.input_dir}")
        sys.exit(1)

    # Find all images with camera data
    print(f"Searching for images in: {args.input_dir}")
    image_data = find_images_with_cameras(args.input_dir, exclude_suffix=args.output_suffix)

    if not image_data:
        print("No images found!")
        sys.exit(1)

    print(f"Found {len(image_data)} images to process")

    # Parse camera data
    print("Loading camera poses...")
    camera_data_cache = {}
    for _, camera_file, _, _ in image_data:
        if camera_file not in camera_data_cache:
            camera_data_cache[camera_file] = parse_camera_data(camera_file)

    # Prepare processing list
    processing_list = []
    view_stats = {'upper': 0, 'lower': 0, 'left': 0, 'right': 0, 'front': 0}

    for img_path, camera_file, camera_side, frame_index in image_data:
        camera_poses = camera_data_cache[camera_file]

        if frame_index not in camera_poses or camera_side not in camera_poses[frame_index]:
            print(f"Warning: No camera data for {img_path}")
            continue

        extrinsics = camera_poses[frame_index][camera_side]
        scan_dir_name = img_path.parent.parent.parent.name
        arch_type = 'upper' if scan_dir_name.startswith('Upper_') else 'lower'
        view = robust_view_classification(extrinsics, arch_type)
        view_stats[view] += 1

        output_path = img_path.parent / f"{frame_index}{args.output_suffix}.png"

        # Skip if output already exists and --skip-existing is set
        if args.skip_existing and output_path.exists():
            continue

        processing_list.append((img_path, output_path, view))

    # Report skipped files
    total_found = len(image_data)
    skipped = total_found - len(processing_list) if args.skip_existing else 0
    if args.skip_existing and skipped > 0:
        print(f"\nSkipped {skipped} images with existing masks")
        print(f"Processing {len(processing_list)} remaining images")

    print(f"\nView classification statistics:")
    for view_type, count in sorted(view_stats.items()):
        print(f"  {view_type}: {count}")

    if args.dry_run:
        print("\nFiles to be processed:")
        for input_path, output_path, view in processing_list[:20]:
            print(f"  {input_path} -> {output_path} (view: {view})")
        if len(processing_list) > 20:
            print(f"  ... and {len(processing_list) - 20} more files")
        print(f"\nTotal: {len(processing_list)} files")
        sys.exit(0)

    # Initialize optimized segmenter
    print(f"\nInitializing parallel processing pipeline...")
    print(f"  Device: {args.device}")
    print(f"  SAM batch size: {args.sam_batch_size}")
    print(f"  Image batch size: {args.image_batch_size}")
    print(f"  I/O workers: {args.io_workers}")
    print(f"  Confidence threshold: {args.conf_threshold}")

    if args.device == 'cuda':
        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    segmenter = OptimizedSegmenter(
        weight_dir=args.weight_dir,
        device=args.device,
        sam_batch_size=args.sam_batch_size,
    )

    # Initialize parallel I/O
    loader = ParallelImageLoader(num_workers=args.io_workers)
    saver = ParallelMaskSaver(num_workers=args.io_workers)

    # Process images in batches
    print(f"\nProcessing {len(processing_list)} images with parallel I/O...")
    successful = 0

    image_paths, output_paths, views = zip(*processing_list)

    # Group images by view
    view_groups = {}
    for i, (img_path, out_path, view) in enumerate(zip(image_paths, output_paths, views)):
        if view not in view_groups:
            view_groups[view] = []
        view_groups[view].append((i, img_path, out_path))

    with tqdm(total=len(processing_list), desc="Processing images") as pbar:
        for view, group in view_groups.items():
            # Process this view in batches
            for batch_start in range(0, len(group), args.image_batch_size):
                batch_end = min(batch_start + args.image_batch_size, len(group))
                batch_items = group[batch_start:batch_end]

                batch_paths = [item[1] for item in batch_items]
                batch_outputs = [item[2] for item in batch_items]
                batch_views = [view] * len(batch_items)

                result = process_batch_parallel(
                    batch_paths,
                    batch_outputs,
                    batch_views,
                    segmenter,
                    args.conf_threshold,
                    loader,
                    saver,
                )
                successful += result
                pbar.update(len(batch_items))

    # Cleanup
    loader.shutdown()
    saver.shutdown()

    # Summary
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"  Successful: {successful}")
    print(f"  Failed: {len(processing_list) - successful}")
    print(f"  Total: {len(processing_list)}")

    if args.device == 'cuda' and torch.cuda.is_available():
        print(f"\nGPU Memory Usage:")
        print(f"  Allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        print(f"  Cached: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")

    print(f"\nOutput files saved in frames_cleanpass/{{left,right}} directories")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
