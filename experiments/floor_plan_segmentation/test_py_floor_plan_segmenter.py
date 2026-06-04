#!/usr/bin/env python3
"""Offline test harness for py_floor_plan_segmenter on ROS SLAM maps.

This script deliberately stays outside the ROS node/package path. It is meant
to answer one question first: does py_floor_plan_segmenter produce useful room
regions on our occupancy maps?

The external project documents a CLI entry point:

    python -m py_floor_plan_segmenter -i <input_folder> -p <output_folder>

It expects a floor-plan image named rank.png in the input folder. We generate
that image from a ROS map YAML + PGM pair, run the CLI, then convert the most
likely segmentation image into standard experiment artifacts.
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import yaml


DEFAULT_MAP_YAML = Path("warehouse_slam_map.yaml")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run py_floor_plan_segmenter as an offline backend on a ROS "
            "SLAM occupancy map."
        )
    )
    parser.add_argument(
        "--map-yaml",
        type=Path,
        default=DEFAULT_MAP_YAML,
        help=(
            "Path to ROS map YAML metadata. Defaults to "
            "warehouse_slam_map.yaml in the workspace root."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/floor_plan_segmentation/outputs"),
        help="Directory for generated experiment artifacts.",
    )
    parser.add_argument(
        "--keep-work-dir",
        action="store_true",
        help="Keep temporary py_floor_plan_segmenter input/output folders.",
    )
    return parser.parse_args()


def load_ros_map(map_yaml: Path) -> Tuple[np.ndarray, Dict]:
    """Load the ROS map YAML and associated PGM/PNG image.

    ROS map YAML files store the image path relative to the YAML file. The image
    is usually a PGM where dark pixels are occupied, light pixels are free, and
    gray pixels are unknown.
    """
    map_yaml = map_yaml.expanduser().resolve()
    if not map_yaml.exists():
        raise FileNotFoundError(f"Map YAML not found: {map_yaml}")

    with map_yaml.open("r", encoding="utf-8") as stream:
        metadata = yaml.safe_load(stream)

    image_path = Path(metadata["image"])
    if not image_path.is_absolute():
        image_path = map_yaml.parent / image_path
    image_path = image_path.resolve()

    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Could not load map image: {image_path}")

    metadata["_resolved_yaml"] = str(map_yaml)
    metadata["_resolved_image"] = str(image_path)
    return image, metadata


def ros_map_to_rank_image(map_image: np.ndarray) -> np.ndarray:
    """Convert a ROS occupancy image to the segmenter's rank.png format.

    We keep the floor-plan convention expected by many segmentation tools:
    walls/occupied cells are black, free space is white, and unknown space stays
    gray so it is visually distinct. For common ROS maps:

    - dark pixels below 100 are occupied -> 0
    - bright pixels above 220 are free -> 255
    - everything else is unknown -> 127
    """
    rank = np.full(map_image.shape, 127, dtype=np.uint8)
    rank[map_image < 100] = 0
    rank[map_image > 220] = 255
    return rank


def ensure_segmenter_available():
    """Fail early with a helpful message if the package is not installed."""
    result = subprocess.run(
        [sys.executable, "-m", "py_floor_plan_segmenter", "--help"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "py_floor_plan_segmenter is not installed or not importable. "
            "See experiments/floor_plan_segmentation/README.md for install "
            "instructions."
        )


def run_segmenter(input_dir: Path, segmenter_output_dir: Path):
    """Run the external py_floor_plan_segmenter CLI."""
    command = [
        sys.executable,
        "-m",
        "py_floor_plan_segmenter",
        "-i",
        str(input_dir),
        "-p",
        str(segmenter_output_dir),
    ]
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "py_floor_plan_segmenter failed.\n"
            f"Command: {' '.join(command)}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    return result


def collect_output_images(output_dir: Path) -> List[Path]:
    """Find PNG/JPG outputs created by the segmenter."""
    patterns = ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff")
    images: List[Path] = []
    for pattern in patterns:
        images.extend(output_dir.rglob(pattern))
    return sorted(images)


def choose_segmentation_image(images: List[Path]) -> Path:
    """Pick the most likely segmentation result image.

    The external repo's exact output names may change, so use a mild heuristic:
    prefer filenames mentioning segmentation/rooms/regions/result, then fall
    back to the last image produced.
    """
    if not images:
        raise RuntimeError(
            "py_floor_plan_segmenter completed, but no image outputs were "
            "found. Cannot create region masks."
        )

    preferred_tokens = ("segment", "room", "region", "result", "label")
    for image_path in images:
        name = image_path.name.lower()
        if any(token in name for token in preferred_tokens):
            return image_path
    return images[-1]


def masks_from_segmentation_image(segmentation_path: Path) -> np.ndarray:
    """Convert a segmentation output image into boolean region masks.

    If the image is color, each non-background color becomes one region. If the
    image is grayscale, each nonzero/non-unknown intensity becomes one region.
    The output shape is (region_count, height, width).
    """
    image = cv2.imread(str(segmentation_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"Could not load segmentation image: {segmentation_path}")

    if image.ndim == 2:
        labels = image
        ignored = {0, 127, 255}
        values = [
            int(value)
            for value in np.unique(labels)
            if int(value) not in ignored
        ]
        if not values:
            # Some tools write a binary room/free-space mask rather than a
            # labeled image. In that case, turn connected non-background blobs
            # into region masks instead of returning an empty result.
            binary = np.zeros_like(labels, dtype=np.uint8)
            binary[labels > 0] = 255
            label_count, component_labels = cv2.connectedComponents(
                binary,
                connectivity=4,
            )
            return np.array(
                [
                    component_labels == label
                    for label in range(1, label_count)
                ],
                dtype=bool,
            )
        return np.array([labels == value for value in values], dtype=bool)

    if image.shape[2] == 4:
        image = image[:, :, :3]

    flat = image.reshape((-1, image.shape[2]))
    colors = np.unique(flat, axis=0)
    masks = []
    for color in colors:
        # Ignore common background colors: black, white, and mid-gray.
        if np.all(color == 0) or np.all(color == 255) or np.all(color == 127):
            continue
        mask = np.all(image == color, axis=2)
        if int(mask.sum()) == 0:
            continue
        masks.append(mask)
    return np.array(masks, dtype=bool)


def summarize_masks(
    masks: np.ndarray,
    metadata: Dict,
) -> List[Dict]:
    """Compute simple image/map-space summaries for region masks."""
    resolution = float(metadata.get("resolution", 1.0))
    origin = metadata.get("origin", [0.0, 0.0, 0.0])
    origin_x = float(origin[0])
    origin_y = float(origin[1])

    summary = []
    for index, mask in enumerate(masks, start=1):
        ys, xs = np.where(mask)
        if len(xs) == 0:
            continue

        centroid_x = float(xs.mean())
        centroid_y = float(ys.mean())
        world_x = origin_x + (centroid_x + 0.5) * resolution
        world_y = origin_y + (centroid_y + 0.5) * resolution

        summary.append({
            "region_id": index,
            "area_pixels": int(len(xs)),
            "area_m2": float(len(xs) * resolution * resolution),
            "centroid_pixel": [centroid_x, centroid_y],
            "centroid_world": [world_x, world_y],
            "bbox_pixel": [
                int(xs.min()),
                int(ys.min()),
                int(xs.max()),
                int(ys.max()),
            ],
        })
    return summary


def create_segmentation_overlay(
    original_map: np.ndarray,
    masks: np.ndarray,
) -> np.ndarray:
    """Overlay region colors on top of the original map image."""
    overlay = cv2.cvtColor(original_map, cv2.COLOR_GRAY2BGR)
    colors = [
        (230, 25, 75),
        (60, 180, 75),
        (255, 225, 25),
        (0, 130, 200),
        (245, 130, 48),
        (145, 30, 180),
        (70, 240, 240),
        (240, 50, 230),
        (210, 245, 60),
        (250, 190, 190),
    ]

    for index, mask in enumerate(masks):
        color = np.array(colors[index % len(colors)], dtype=np.uint8)
        overlay[mask] = (
            0.45 * overlay[mask].astype(np.float32) +
            0.55 * color.astype(np.float32)
        ).astype(np.uint8)

    return overlay


def main():
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    work_dir = output_dir / "_work"
    input_dir = work_dir / "input"
    segmenter_output_dir = work_dir / "segmenter_output"

    ensure_segmenter_available()
    map_image, metadata = load_ros_map(args.map_yaml)

    output_dir.mkdir(parents=True, exist_ok=True)
    if work_dir.exists():
        shutil.rmtree(work_dir)
    input_dir.mkdir(parents=True)
    segmenter_output_dir.mkdir(parents=True)

    rank_image = ros_map_to_rank_image(map_image)

    cv2.imwrite(str(output_dir / "original_map.png"), map_image)
    cv2.imwrite(str(input_dir / "rank.png"), rank_image)

    result = run_segmenter(input_dir, segmenter_output_dir)

    output_images = collect_output_images(segmenter_output_dir)
    segmentation_image = choose_segmentation_image(output_images)
    masks = masks_from_segmentation_image(segmentation_image)
    summary = summarize_masks(masks, metadata)
    overlay = create_segmentation_overlay(map_image, masks)

    cv2.imwrite(str(output_dir / "segmentation_overlay.png"), overlay)
    np.save(output_dir / "region_masks.npy", masks)

    summary_payload = {
        "map_yaml": metadata["_resolved_yaml"],
        "map_image": metadata["_resolved_image"],
        "segmenter_output_image": str(segmentation_image),
        "region_count": len(summary),
        "regions": summary,
        "segmenter_stdout": result.stdout,
        "segmenter_stderr": result.stderr,
    }
    with (output_dir / "region_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary_payload, stream, indent=2)

    if not args.keep_work_dir:
        shutil.rmtree(work_dir)

    print(f"Saved original map: {output_dir / 'original_map.png'}")
    print(f"Saved overlay: {output_dir / 'segmentation_overlay.png'}")
    print(f"Saved masks: {output_dir / 'region_masks.npy'}")
    print(f"Saved summary: {output_dir / 'region_summary.json'}")
    print(f"Detected regions: {len(summary)}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
