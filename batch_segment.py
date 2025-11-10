#!/usr/bin/env python3
"""
Batch processing script for SegmentAnyTooth segmentation.
Recursively processes all images in frames_cleanpass/ directories.
"""
import os
import sys
from pathlib import Path
from typing import List, Tuple
import argparse
import cv2
import numpy as np
from tqdm import tqdm
from segmentanytooth import predict


def find_images_in_frames_cleanpass(root_dir: str) -> List[Tuple[Path, Path]]:
    """
    Recursively find all images in frames_cleanpass directories.

    Args:
        root_dir: Root directory to search

    Returns:
        List of tuples (input_path, output_path)
    """
    root_path = Path(root_dir)
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
    image_pairs = []

    # Find all frames_cleanpass directories
    for frames_dir in root_path.rglob('frames_cleanpass'):
        # Find all image files in this directory and subdirectories
        for img_path in frames_dir.rglob('*'):
            if img_path.suffix.lower() in image_extensions and img_path.is_file():
                # Create output path in same location with "_seg" suffix
                output_path = img_path.parent / f"{img_path.stem}_seg{img_path.suffix}"
                image_pairs.append((img_path, output_path))

    return sorted(image_pairs)


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
        output_path: Output mask path (with _seg suffix)
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

        # Save mask in same location as input with _seg suffix
        # Mask is saved at original scale (pixel values = FDI tooth numbers)
        cv2.imwrite(str(output_path), mask)

        return True

    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Batch process dental images with SegmentAnyTooth'
    )
    parser.add_argument(
        'input_dir',
        type=str,
        help='Root directory to search for frames_cleanpass folders'
    )
    parser.add_argument(
        '--weight-dir',
        type=str,
        default='/home/itamar/git/mantis/pretrained_models/SegmentAnyToothWeights',
        help='Path to model weights directory'
    )
    parser.add_argument(
        '--view',
        type=str,
        choices=['upper', 'lower', 'left', 'right', 'front'],
        default='front',
        help='View type for all images (default: front)'
    )
    parser.add_argument(
        '--conf-threshold',
        type=float,
        default=0.01,
        help='Confidence threshold for YOLO detection (default: 0.01 for 3D renders, 0.25 for real photos)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='List files that would be processed without actually processing them'
    )

    args = parser.parse_args()

    # Validate input directory
    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory does not exist: {args.input_dir}")
        sys.exit(1)

    # Find all images
    print(f"Searching for images in frames_cleanpass directories under: {args.input_dir}")
    image_pairs = find_images_in_frames_cleanpass(args.input_dir)

    if not image_pairs:
        print("No images found in frames_cleanpass directories!")
        sys.exit(1)

    print(f"Found {len(image_pairs)} images to process")

    # Dry run mode - just list files
    if args.dry_run:
        print("\nFiles to be processed:")
        for input_path, output_path in image_pairs:
            print(f"  {input_path} -> {output_path}")
        print(f"\nTotal: {len(image_pairs)} files")
        sys.exit(0)

    # Process images
    print(f"\nProcessing with settings:")
    print(f"  View: {args.view}")
    print(f"  Confidence threshold: {args.conf_threshold}")
    print(f"  Output: Same location as input with '_seg' suffix")
    print()

    successful = 0
    failed = 0

    for input_path, output_path in tqdm(image_pairs, desc="Processing images"):
        success = process_image(
            input_path=input_path,
            output_path=output_path,
            weight_dir=args.weight_dir,
            view=args.view,
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
    print(f"  Total: {len(image_pairs)}")
    print(f"\nOutput files saved with '_seg' suffix in same directories as input")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
