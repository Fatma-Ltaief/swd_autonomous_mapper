# Offline Floor Plan Segmentation Experiment

This folder tests `py_floor_plan_segmenter` as an offline room segmentation backend for ROS SLAM occupancy maps.

It does not modify:

- `room_active_slam`
- Gazebo
- Nav2
- Sionna
- radio nodes

## Install `py_floor_plan_segmenter`

The external repo is not vendored into this workspace. Install it in a virtual environment or your ROS Python environment:

```bash
python3 -m pip install git+https://github.com/sharif1093/py_floor_plan_segmenter.git
```

If pip installation fails because of missing system packages, install OpenCV and common Python build tools first:

```bash
sudo apt update
sudo apt install python3-pip python3-opencv python3-yaml
```

Then retry the pip command.

## Run The Experiment

From the workspace root:

```bash
cd ~/autonomous_slam_baseline
python3 experiments/floor_plan_segmentation/test_py_floor_plan_segmenter.py \
  --map-yaml warehouse_slam_map.yaml
```

You can point it at another ROS map YAML:

```bash
python3 experiments/floor_plan_segmentation/test_py_floor_plan_segmenter.py \
  --map-yaml /path/to/map.yaml \
  --output-dir experiments/floor_plan_segmentation/outputs/my_map
```

## Outputs

The script writes:

- `original_map.png`: the input ROS map image.
- `segmentation_overlay.png`: colored region masks over the original map.
- `region_masks.npy`: boolean masks with shape `(region_count, height, width)`.
- `region_summary.json`: region areas, centroids, bounding boxes, source map paths, and segmenter logs.

## Notes

The script creates a temporary `rank.png` because the external package's CLI expects that name inside an input folder. `rank.png` uses:

- occupied cells: black (`0`)
- free cells: white (`255`)
- unknown cells: gray (`127`)

This is an offline experiment first. If the segmentation quality is useful, the next step is to wrap the backend behind a clean interface before integrating it into `room_active_slam`.
