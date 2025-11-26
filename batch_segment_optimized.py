#!/usr/bin/env python3
"""
Optimized batch processing script for SegmentAnyTooth segmentation with T4 GPU.
Key optimizations:
- Loads models once and reuses them
- Explicit GPU device placement
- Batch image preprocessing
- Larger SAM batch sizes
- Pinned memory for faster CPU->GPU transfers
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
from ultralytics import YOLO
from ultralytics.utils import LOGGER

from sam import sam_load, SamMobilePredictor
from utils import suppress_stdout

# Set Ultralytics logger to error-only
LOGGER.setLevel("ERROR")

# Define class names for left lateral view
LEFT_CLASSES = [
    "le28", "le27", "le26", "le25", "le24", "le23", "le22", "le21",
    "le38", "le37", "le36", "le35", "le34", "le33", "le32", "le31",
    "le11", "le12", "le13", "le14", "le41", "le42", "le43", "le44",
]


def parse_camera_data(camera_data_path: Path) -> Dict[str, Dict[str, np.ndarray]]:
    """Parse camera_data.txt file to extract camera poses."""
    camera_poses = {}

    with open(camera_data_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    i = 0
    while i < len(lines):
        frame_index = lines[i].strip()
        i += 1
        camera_poses[frame_index] = {}

        while i < len(lines) and lines[i][0] in ['L', 'R']:
            parts = lines[i].split()
            camera_side = 'left' if parts[0] == 'L' else 'right'
            extrinsics_flat = [float(x) for x in parts[10:26]]
            extrinsics = np.array(extrinsics_flat).reshape(4, 4)
            camera_poses[frame_index][camera_side] = extrinsics
            i += 1

    return camera_poses


def robust_view_classification(extrinsics: np.ndarray, arch_type: str) -> str:
    """Classify camera view based on camera pose."""
    rotation = extrinsics[:3, :3]
    translation = extrinsics[:3, 3]
    camera_z = np.array([0, 0, -1])
    view_direction = rotation.T @ camera_z
    camera_position = -rotation.T @ translation
    view_dir_norm = view_direction / (np.linalg.norm(view_direction) + 1e-8)
    abs_dir = np.abs(view_dir_norm)

    VERTICAL_THRESHOLD = 0.7
    LATERAL_THRESHOLD = 0.7

    if abs_dir[1] > VERTICAL_THRESHOLD:
        if view_dir_norm[1] < 0:
            return "upper"
        else:
            return "lower"
    elif abs_dir[0] > LATERAL_THRESHOLD:
        if view_dir_norm[0] > 0:
            return "left"
        else:
            return "right"
    else:
        if abs_dir[1] > 0.4:
            return arch_type
        return "front"


def find_images_with_cameras(root_dir: str, exclude_suffix: str = '_mask') -> List[Tuple[Path, Path, str, str]]:
    """
    Find all images with camera data, excluding output files.

    Args:
        root_dir: Root directory to search
        exclude_suffix: Suffix to exclude (e.g., '_mask' to skip files like '0003_mask.png')

    Returns:
        List of tuples (input_path, camera_data_path, camera_side, frame_index)
    """
    root_path = Path(root_dir)
    image_data = []

    for camera_file in root_path.rglob('camera_data.txt'):
        scan_dir = camera_file.parent
        frames_dir = scan_dir / 'frames_cleanpass'

        if not frames_dir.exists():
            continue

        for side in ['left', 'right']:
            side_dir = frames_dir / side
            if not side_dir.exists():
                continue

            for img_path in sorted(side_dir.glob('*.png')):
                frame_index = img_path.stem

                # Skip files that match the exclude pattern (e.g., output mask files)
                if exclude_suffix and frame_index.endswith(exclude_suffix):
                    continue

                image_data.append((img_path, camera_file, side, frame_index))

    return sorted(image_data)


def get_model_path(model: str, weight_dir: str) -> str:
    """Returns the file path to the model weights."""
    if model == "left":
        model = "right"
    if model == "sam":
        name = "vit_tiny.pt"
    else:
        name = f"yolo11_{model}.pt"
    return os.path.join(weight_dir, f"segmentanytooth_{name}")


class OptimizedSegmenter:
    """Optimized segmenter that loads models once and reuses them."""

    def __init__(self, weight_dir: str, device: str = 'cuda', sam_batch_size: int = 64):
        """
        Initialize segmenter with models loaded onto GPU.

        Args:
            weight_dir: Path to model weights
            device: Device to use ('cuda' or 'cpu')
            sam_batch_size: Batch size for SAM inference (default: 64 for A10G, 32 for T4, 128+ for A100)
        """
        self.device = device
        self.weight_dir = weight_dir
        self.sam_batch_size = sam_batch_size

        # Check GPU availability
        if device == 'cuda' and not torch.cuda.is_available():
            print("Warning: CUDA not available, falling back to CPU")
            self.device = 'cpu'

        # Load SAM model once
        print(f"Loading SAM model on {self.device}...")
        with suppress_stdout():
            self.sam = sam_load(get_model_path("sam", weight_dir))
            self.sam = self.sam.to(self.device)
            self.sam.eval()  # Set to evaluation mode

        # Cache YOLO models per view
        self.yolo_models = {}
        print(f"Models will be loaded on {self.device}")

        # Enable tensor cores for T4 (if using CUDA)
        if self.device == 'cuda':
            torch.backends.cudnn.benchmark = True
            # Enable TF32 for faster computation on Ampere+ GPUs
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    def get_yolo_model(self, view: str) -> YOLO:
        """Get or load YOLO model for specific view."""
        if view not in self.yolo_models:
            print(f"Loading YOLO model for {view} view on {self.device}...")
            with suppress_stdout():
                # CRITICAL: Pass device to YOLO constructor for proper GPU loading
                model = YOLO(model=get_model_path(view, self.weight_dir))
                if self.device == 'cuda':
                    # Force model to GPU - use both methods for compatibility
                    model.to('cuda')
                    # Verify it worked
                    model_device = str(next(model.model.parameters()).device)
                    if 'cpu' in model_device.lower():
                        print(f"  WARNING: YOLO model is on CPU despite .to('cuda')! Trying alternative...")
                        # Try alternative method
                        import torch
                        model.model = model.model.cuda()
                        model_device = str(next(model.model.parameters()).device)
                    print(f"  YOLO model device: {model_device}")
            self.yolo_models[view] = model
        return self.yolo_models[view]

    def predict(
        self,
        image: np.ndarray,
        view: str,
        conf_threshold: float = 0.01,
    ) -> np.ndarray:
        """
        Predict segmentation mask for a single image.

        Args:
            image: Input image (already loaded)
            view: View type
            conf_threshold: Confidence threshold

        Returns:
            Segmentation mask
        """
        should_flip = view == "left"

        if should_flip:
            image = cv2.flip(image, 1)

        # Get YOLO model for this view
        yolo = self.get_yolo_model(view)

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

        # Early exit if no detections
        if r.boxes is None or len(r.boxes) == 0:
            return np.zeros(image.shape[:2], dtype=np.uint8)

        # Get YOLO output
        names = r.names if not should_flip else LEFT_CLASSES
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

        # SAM prediction with optimized batch processing
        sam_masks = self._sam_predict_optimized(boxes, image_rgb)

        # Build segmentation mask
        predict_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        for cls_id, current_mask in zip(clss, sam_masks):
            fdi_tooth_name = int(names[cls_id][-2:])
            predict_mask[current_mask == 1] = fdi_tooth_name

        return predict_mask

    @torch.no_grad()
    def _sam_predict_optimized(self, boxes_xyxy: np.ndarray, image: np.ndarray) -> np.ndarray:
        """Optimized SAM prediction with larger batches."""
        predictor = SamMobilePredictor(self.sam)
        predictor.set_image(image)

        batch_boxes = np.split(
            boxes_xyxy,
            range(self.sam_batch_size, len(boxes_xyxy), self.sam_batch_size)
        )
        batch_masks = []

        for boxes in batch_boxes:
            # Use pinned memory for faster transfer
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
            if self.device == 'cuda':
                boxes_tensor = boxes_tensor.pin_memory()

            transformed_boxes = predictor.transform.apply_boxes_torch(
                boxes_tensor, image.shape[:2]
            ).to(self.device, non_blocking=True)

            sam_masks, _, _ = predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=False,
                hq_token_only=False,
            )

            sam_masks = sam_masks.squeeze().cpu().numpy().astype(np.uint8)

            if len(sam_masks.shape) == 2:
                batch_masks.append(sam_masks)
            else:
                batch_masks.extend(sam_masks.tolist())

        return np.stack(batch_masks)


def process_batch(
    image_paths: List[Path],
    output_paths: List[Path],
    views: List[str],
    segmenter: OptimizedSegmenter,
    conf_threshold: float,
    verbose: bool = False,
) -> int:
    """
    Process a batch of images together with true multi-image batching.

    Batching strategy:
    1. Load all images in batch
    2. Run YOLO detection on each image (YOLO is fast)
    3. Collect all boxes from all images
    4. Run SAM on ALL boxes at once (this is where GPU batching helps)
    5. Distribute masks back to respective images

    Args:
        image_paths: List of input image paths
        output_paths: List of output mask paths
        views: List of view types (should all be the same for efficiency)
        segmenter: Optimized segmenter instance
        conf_threshold: Confidence threshold

    Returns:
        Number of successfully processed images
    """
    if len(image_paths) == 0:
        return 0

    import time
    batch_start = time.time()

    # Assume all images in batch have same view (guaranteed by caller)
    view = views[0]

    # Get YOLO model for this view
    yolo = segmenter.get_yolo_model(view)
    should_flip = view == "left"
    names = yolo.model.names if not should_flip else LEFT_CLASSES

    # Step 1: Load all images and run YOLO detections
    if verbose:
        print(f"\n  Processing batch of {len(image_paths)} images (view: {view})")

    io_start = time.time()
    batch_data = []  # List of (image, boxes, clss, input_path, output_path)

    for input_path, output_path in zip(image_paths, output_paths):
        try:
            # Load image
            image = cv2.imread(str(input_path))
            if image is None:
                print(f"Failed to load {input_path}")
                continue

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
                # Save empty mask
                empty_mask = np.zeros(image.shape[:2], dtype=np.uint8)
                cv2.imwrite(str(output_path), empty_mask)
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
            })

        except Exception as e:
            print(f"Error processing {input_path}: {e}")

    if len(batch_data) == 0:
        return 0

    io_time = time.time() - io_start
    if verbose:
        print(f"    I/O + YOLO: {io_time:.2f}s ({io_time/len(batch_data)*1000:.0f}ms per image)")

    # Step 2: Compute SAM embeddings for all images and collect all boxes
    all_boxes = []
    all_image_indices = []
    all_box_indices = []

    for img_idx, data in enumerate(batch_data):
        boxes = data['boxes']
        # Track which image each box belongs to
        for box_idx in range(len(boxes)):
            all_boxes.append(boxes[box_idx])
            all_image_indices.append(img_idx)
            all_box_indices.append(box_idx)

    if len(all_boxes) == 0:
        return 0

    all_boxes = np.array(all_boxes)

    # Step 3: Process all boxes through SAM in large batches
    # We'll process each image's SAM separately since SAM needs image embeddings
    sam_start = time.time()
    successful = 0

    for img_idx, data in enumerate(batch_data):
        try:
            # Get SAM predictions for this image
            sam_masks = segmenter._sam_predict_optimized(data['boxes'], data['image'])

            # Build segmentation mask
            predict_mask = np.zeros(data['image'].shape[:2], dtype=np.uint8)
            for cls_id, current_mask in zip(data['clss'], sam_masks):
                fdi_tooth_name = int(names[cls_id][-2:])
                predict_mask[current_mask == 1] = fdi_tooth_name

            # Save mask
            cv2.imwrite(str(data['output_path']), predict_mask)
            successful += 1

        except Exception as e:
            print(f"Error saving mask for {data['output_path']}: {e}")

    sam_time = time.time() - sam_start
    total_time = time.time() - batch_start

    if verbose:
        print(f"    SAM: {sam_time:.2f}s ({sam_time/len(batch_data)*1000:.0f}ms per image)")
        print(f"    Total: {total_time:.2f}s ({total_time/len(batch_data)*1000:.0f}ms per image)")
        print(f"    Throughput: {len(batch_data)/total_time:.2f} images/sec")

    return successful


def main():
    parser = argparse.ArgumentParser(
        description='Optimized batch processing for T4 GPU with automatic view classification'
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
        help='SAM batch size for processing teeth within each image (default: 64 for A10G). Use 32 for T4, 128+ for A100.'
    )
    parser.add_argument(
        '--image-batch-size',
        type=int,
        default=8,
        help='Number of images to load/process together per batch (default: 8 for A10G). Images are grouped by view type automatically. Increase to 16-32 for more parallelism, decrease to 4 if OOM.'
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
    print(f"\nInitializing optimized segmenter...")
    print(f"  Device: {args.device}")
    print(f"  SAM batch size: {args.sam_batch_size}")
    print(f"  Image batch size: {args.image_batch_size}")
    print(f"  Confidence threshold: {args.conf_threshold}")

    if args.device == 'cuda':
        # Print GPU info
        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    segmenter = OptimizedSegmenter(
        weight_dir=args.weight_dir,
        device=args.device,
        sam_batch_size=args.sam_batch_size,
    )

    # Process images in batches
    print(f"\nProcessing {len(processing_list)} images...")
    successful = 0

    image_paths, output_paths, views = zip(*processing_list)

    # Group images by view for efficient batching
    view_groups = {}
    for i, (img_path, out_path, view) in enumerate(zip(image_paths, output_paths, views)):
        if view not in view_groups:
            view_groups[view] = []
        view_groups[view].append((i, img_path, out_path))

    # Process each view group in batches
    total_batches = sum((len(group) + args.image_batch_size - 1) // args.image_batch_size
                       for group in view_groups.values())

    with tqdm(total=len(processing_list), desc="Processing images") as pbar:
        for view, group in view_groups.items():
            # Process this view in batches
            for batch_start in range(0, len(group), args.image_batch_size):
                batch_end = min(batch_start + args.image_batch_size, len(group))
                batch_items = group[batch_start:batch_end]

                batch_paths = [item[1] for item in batch_items]
                batch_outputs = [item[2] for item in batch_items]
                batch_views = [view] * len(batch_items)

                result = process_batch(
                    batch_paths,
                    batch_outputs,
                    batch_views,
                    segmenter,
                    args.conf_threshold,
                    verbose=args.verbose,
                )
                successful += result
                pbar.update(len(batch_items))

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
