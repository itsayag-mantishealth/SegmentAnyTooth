#!/usr/bin/env python3
"""
Batch processing script for SegmentAnyTooth segmentation.
Automatically determines the view type from camera poses in camera_data.txt.
Processes images in frames_cleanpass/{left,right} directories.
"""
import os
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import argparse
import cv2
import numpy as np
from tqdm import tqdm
from segmentanytooth import predict


def parse_camera_data(camera_data_path: Path) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Parse camera_data.txt file to extract camera poses.

    Format:
    {frame_index}
    L {intrinsics 9 values} {extrinsics 16 values (4x4 matrix in row-major)}
    R {intrinsics 9 values} {extrinsics 16 values (4x4 matrix in row-major)}

    Args:
        camera_data_path: Path to camera_data.txt

    Returns:
        Dict mapping frame index to camera side ('left'/'right') to extrinsics matrix (4x4)
    """
    camera_poses = {}

    with open(camera_data_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    i = 0
    while i < len(lines):
        # Read frame index
        frame_index = lines[i].strip()
        i += 1

        camera_poses[frame_index] = {}

        # Read L and R camera lines
        while i < len(lines) and lines[i][0] in ['L', 'R']:
            parts = lines[i].split()
            camera_side = 'left' if parts[0] == 'L' else 'right'

            # Parse extrinsics (4x4 matrix) - starts after intrinsics (9 values)
            # Intrinsics: fx 0 cx 0 fy cy 0 0 1 (9 values)
            # Extrinsics: 16 values (4x4 matrix in row-major order)
            extrinsics_flat = [float(x) for x in parts[10:26]]
            extrinsics = np.array(extrinsics_flat).reshape(4, 4)

            camera_poses[frame_index][camera_side] = extrinsics
            i += 1

    return camera_poses


def robust_view_classification(extrinsics: np.ndarray, arch_type: str) -> str:
    """
    Classify camera view based on camera pose (extrinsics matrix).

    The extrinsics matrix is a 4x4 transformation matrix:
    [R | t]
    [0 | 1]
    where R is 3x3 rotation and t is 3x1 translation.

    Args:
        extrinsics: 4x4 camera extrinsics matrix (world to camera transform)
        arch_type: 'upper' or 'lower' from directory name

    Returns:
        view: one of ["upper", "lower", "left", "right", "front"]
    """
    # Extract rotation matrix and translation
    rotation = extrinsics[:3, :3]
    translation = extrinsics[:3, 3]

    # Camera view direction in world space (camera looks down -Z axis)
    # Transform camera -Z direction to world coordinates
    camera_z = np.array([0, 0, -1])
    view_direction = rotation.T @ camera_z  # Inverse rotation (R^T for orthogonal matrix)

    # Camera position in world space
    camera_position = -rotation.T @ translation

    # Normalize
    view_dir_norm = view_direction / (np.linalg.norm(view_direction) + 1e-8)

    # Calculate dominant viewing axis
    abs_dir = np.abs(view_dir_norm)

    # Define thresholds
    VERTICAL_THRESHOLD = 0.7  # Strong vertical component
    LATERAL_THRESHOLD = 0.7   # Strong lateral component

    # Classification logic
    # Check if view is predominantly vertical (Y-axis)
    if abs_dir[1] > VERTICAL_THRESHOLD:
        # Looking down (negative Y) -> upper teeth
        # Looking up (positive Y) -> lower teeth
        if view_dir_norm[1] < 0:
            return "upper"
        else:
            return "lower"

    # Check if view is predominantly lateral (X-axis)
    elif abs_dir[0] > LATERAL_THRESHOLD:
        # Looking from right to left (positive X) -> left view
        # Looking from left to right (negative X) -> right view
        if view_dir_norm[0] > 0:
            return "left"
        else:
            return "right"

    # Otherwise, assume front view (Z-axis dominant or oblique)
    else:
        # For oblique views, use arch_type as a hint
        # If clearly from above/below, use arch_type
        if abs_dir[1] > 0.4:
            return arch_type  # Use 'upper' or 'lower' from directory
        return "front"


def find_images_with_cameras(root_dir: str) -> List[Tuple[Path, Path, str, str]]:
    """
    Find all images in frames_cleanpass/{left,right} directories with camera data.

    Args:
        root_dir: Root directory to search

    Returns:
        List of tuples (input_path, camera_data_path, camera_side, frame_index)
    """
    root_path = Path(root_dir)
    image_data = []

    # Find all directories with camera_data.txt
    for camera_file in root_path.rglob('camera_data.txt'):
        scan_dir = camera_file.parent
        frames_dir = scan_dir / 'frames_cleanpass'

        if not frames_dir.exists():
            continue

        # Check for left and right subdirectories
        for side in ['left', 'right']:
            side_dir = frames_dir / side
            if not side_dir.exists():
                continue

            # Find all PNG images
            for img_path in sorted(side_dir.glob('*.png')):
                # Extract frame index from filename (e.g., "0001.png" -> "0001")
                frame_index = img_path.stem
                image_data.append((img_path, camera_file, side, frame_index))

    return sorted(image_data)


def process_image(
    input_path: Path,
    output_path: Path,
    weight_dir: str,
    view: str,
    conf_threshold: float,
) -> bool:
    """
    Process a single image with SegmentAnyTooth.

    Args:
        input_path: Input image path
        output_path: Output mask path
        weight_dir: Path to model weights
        view: View type (upper/lower/left/right/front)
        conf_threshold: Confidence threshold for detection

    Returns:
        True if successful, False otherwise
    """
    try:
        # Run prediction
        mask = predict(
            image_path=str(input_path),
            view=view,
            weight_dir=weight_dir,
            conf_threshold=conf_threshold,
        )

        # Save mask
        cv2.imwrite(str(output_path), mask)

        return True

    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Batch process dental images with automatic view classification from camera poses'
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
        help='Confidence threshold for YOLO detection (default: 0.01 for 3D renders)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='List files that would be processed without actually processing them'
    )
    parser.add_argument(
        '--output-suffix',
        type=str,
        default='_mask',
        help='Output file suffix (default: _mask, resulting in {index}_mask.png)'
    )

    args = parser.parse_args()

    # Validate input directory
    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory does not exist: {args.input_dir}")
        sys.exit(1)

    # Find all images with camera data
    print(f"Searching for images in frames_cleanpass/{{left,right}} directories under: {args.input_dir}")
    image_data = find_images_with_cameras(args.input_dir)

    if not image_data:
        print("No images found in frames_cleanpass/{left,right} directories with camera_data.txt!")
        sys.exit(1)

    print(f"Found {len(image_data)} images to process")

    # Parse all camera data files
    print("Loading camera poses...")
    camera_data_cache = {}
    for _, camera_file, _, _ in image_data:
        if camera_file not in camera_data_cache:
            camera_data_cache[camera_file] = parse_camera_data(camera_file)

    # Prepare processing list with view classification
    processing_list = []
    view_stats = {'upper': 0, 'lower': 0, 'left': 0, 'right': 0, 'front': 0}

    for img_path, camera_file, camera_side, frame_index in image_data:
        # Get camera pose
        camera_poses = camera_data_cache[camera_file]

        if frame_index not in camera_poses or camera_side not in camera_poses[frame_index]:
            print(f"Warning: No camera data for {img_path} (frame {frame_index}, side {camera_side})")
            continue

        extrinsics = camera_poses[frame_index][camera_side]

        # Determine arch type from directory name (Upper_xxx or Lower_xxx)
        scan_dir_name = img_path.parent.parent.parent.name
        arch_type = 'upper' if scan_dir_name.startswith('Upper_') else 'lower'

        # Classify view
        view = robust_view_classification(extrinsics, arch_type)
        view_stats[view] += 1

        # Create output path: same directory, {frame_index}_mask.png
        output_path = img_path.parent / f"{frame_index}{args.output_suffix}.png"

        processing_list.append((img_path, output_path, view))

    print(f"\nView classification statistics:")
    for view_type, count in sorted(view_stats.items()):
        print(f"  {view_type}: {count}")

    # Dry run mode - just list files
    if args.dry_run:
        print("\nFiles to be processed:")
        for input_path, output_path, view in processing_list[:20]:  # Show first 20
            print(f"  {input_path} -> {output_path} (view: {view})")
        if len(processing_list) > 20:
            print(f"  ... and {len(processing_list) - 20} more files")
        print(f"\nTotal: {len(processing_list)} files")
        sys.exit(0)

    # Process images
    print(f"\nProcessing with settings:")
    print(f"  Confidence threshold: {args.conf_threshold}")
    print(f"  Output format: {{frame_index}}{args.output_suffix}.png")
    print()

    successful = 0
    failed = 0

    for input_path, output_path, view in tqdm(processing_list, desc="Processing images"):
        success = process_image(
            input_path=input_path,
            output_path=output_path,
            weight_dir=args.weight_dir,
            view=view,
            conf_threshold=args.conf_threshold,
        )

        if success:
            successful += 1
        else:
            failed += 1

    # Summary
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Total: {len(processing_list)}")
    print(f"\nOutput files saved as {{frame_index}}{args.output_suffix}.png in frames_cleanpass/{{left,right}} directories")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
