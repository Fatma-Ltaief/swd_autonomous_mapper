#!/usr/bin/env python3
"""Offline/static debugger for R-ALC region frontier logic.

This script mirrors the wall-aware region/frontier logic without publishing any
navigation goals. It can read a saved ROS map YAML/PGM or sample /map and
/ralc/current_region once from a running ROS graph.
"""

import argparse
import json
import math
import sys
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml


FREE = 0
UNKNOWN = -1
OCCUPIED = 100


PALETTE = {
    "free": (235, 235, 235),
    "occupied": (20, 20, 20),
    "unknown": (120, 145, 135),
    "region": (30, 230, 30),
    "reachable": (255, 220, 60),
    "frontier": (255, 80, 10),
    "global_frontier": (230, 40, 190),
    "rejected": (150, 150, 150),
    "candidate": (0, 145, 255),
    "selected": (0, 255, 255),
    "bad_goal": (0, 0, 255),
    "robot": (0, 80, 255),
}


class SimpleGrid:
    def __init__(
        self,
        data: np.ndarray,
        resolution: float,
        origin_x: float,
        origin_y: float,
        frame_id: str = "map",
    ):
        self.data = data.astype(np.int16)
        self.height, self.width = data.shape
        self.resolution = float(resolution)
        self.origin_x = float(origin_x)
        self.origin_y = float(origin_y)
        self.frame_id = frame_id


def parse_args():
    parser = argparse.ArgumentParser(
        description="Debug wall-aware R-ALC region/frontier logic."
    )
    parser.add_argument("--map-yaml", help="Saved ROS map YAML path.")
    parser.add_argument("--map-topic", default="/map", help="Live map topic.")
    parser.add_argument(
        "--region-topic",
        default="/ralc/current_region",
        help="Live current region JSON topic.",
    )
    parser.add_argument("--region-json", help="Region JSON string or file path.")
    parser.add_argument("--xmin", type=float)
    parser.add_argument("--xmax", type=float)
    parser.add_argument("--ymin", type=float)
    parser.add_argument("--ymax", type=float)
    parser.add_argument("--robot-x", type=float)
    parser.add_argument("--robot-y", type=float)
    parser.add_argument("--robot-frame", default="base_link")
    parser.add_argument("--map-frame", default="map")
    parser.add_argument("--output", default="debug_region.png")
    parser.add_argument("--json-output", help="Output JSON summary path.")
    parser.add_argument("--timeout-sec", type=float, default=5.0)
    parser.add_argument("--min-cluster-size", type=int, default=3)
    parser.add_argument("--min-actionable-cluster-size", type=int, default=15)
    parser.add_argument("--occupied-margin", type=float, default=0.10)
    parser.add_argument("--unknown-margin", type=float, default=0.10)
    parser.add_argument("--min-goal-distance", type=float, default=0.35)
    parser.add_argument(
        "--goal-shifts",
        type=float,
        nargs="+",
        default=[0.35, 0.55, 0.80, 1.10, 1.40],
        help="Candidate shifts from frontier centroid toward robot.",
    )
    parser.add_argument(
        "--completion-threshold",
        type=float,
        default=0.03,
        help="Reachable unknown ratio threshold for completion.",
    )
    return parser.parse_args()


def load_saved_map(yaml_path: str) -> SimpleGrid:
    yaml_path = Path(yaml_path).expanduser()
    with yaml_path.open("r", encoding="utf-8") as stream:
        metadata = yaml.safe_load(stream)

    image_path = Path(metadata["image"])
    if not image_path.is_absolute():
        image_path = yaml_path.parent / image_path
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Could not read map image: {image_path}")

    negate = int(metadata.get("negate", 0))
    if negate:
        image = 255 - image
    occupied_thresh = float(metadata.get("occupied_thresh", 0.65))
    free_thresh = float(metadata.get("free_thresh", 0.196))
    # ROS map YAML thresholds are probabilities; PGM pixels encode occupancy as
    # 0=occupied/dark and 255=free/light when negate is false.
    occupancy_probability = (255.0 - image.astype(np.float32)) / 255.0
    data = np.full(image.shape, UNKNOWN, dtype=np.int16)
    data[occupancy_probability >= occupied_thresh] = OCCUPIED
    data[occupancy_probability <= free_thresh] = FREE

    origin = metadata.get("origin", [0.0, 0.0, 0.0])
    return SimpleGrid(
        data=data,
        resolution=float(metadata["resolution"]),
        origin_x=float(origin[0]),
        origin_y=float(origin[1]),
    )


def load_live_inputs(args):
    try:
        import rclpy
        from nav_msgs.msg import OccupancyGrid
        from rclpy.duration import Duration
        from rclpy.node import Node
        from rclpy.time import Time
        from std_msgs.msg import String
        from tf2_ros import Buffer, TransformException, TransformListener
    except Exception as exc:
        raise RuntimeError(
            "Live ROS input requires rclpy/nav_msgs/std_msgs/tf2_ros."
        ) from exc

    class CaptureNode(Node):
        def __init__(self):
            super().__init__("debug_region_frontiers_capture")
            self.map_msg = None
            self.region_msg = None
            self.tf_buffer = Buffer()
            self.tf_listener = TransformListener(self.tf_buffer, self)
            self.create_subscription(OccupancyGrid, args.map_topic, self.map_cb, 10)
            self.create_subscription(String, args.region_topic, self.region_cb, 10)

        def map_cb(self, msg):
            self.map_msg = msg

        def region_cb(self, msg):
            self.region_msg = msg.data

        def lookup_robot_xy(self):
            try:
                transform = self.tf_buffer.lookup_transform(
                    args.map_frame,
                    args.robot_frame,
                    Time(),
                    timeout=Duration(seconds=0.2),
                )
            except TransformException:
                return None
            t = transform.transform.translation
            return float(t.x), float(t.y)

    rclpy.init(args=None)
    node = CaptureNode()
    deadline = time.monotonic() + args.timeout_sec
    robot_xy = None
    try:
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
            if args.robot_x is None or args.robot_y is None:
                robot_xy = node.lookup_robot_xy()
            if node.map_msg is not None:
                has_region = (
                    args.region_json is not None or
                    manual_region_available(args) or
                    node.region_msg is not None
                )
                has_robot = (
                    (args.robot_x is not None and args.robot_y is not None) or
                    robot_xy is not None
                )
                if has_region and has_robot:
                    break
    finally:
        node.destroy_node()
        rclpy.shutdown()

    if node.map_msg is None:
        raise RuntimeError(f"No map received from {args.map_topic}")

    msg = node.map_msg
    data = np.array(msg.data, dtype=np.int16).reshape(
        (msg.info.height, msg.info.width)
    )
    grid = SimpleGrid(
        data=data,
        resolution=msg.info.resolution,
        origin_x=msg.info.origin.position.x,
        origin_y=msg.info.origin.position.y,
        frame_id=msg.header.frame_id or args.map_frame,
    )
    region_json = args.region_json or node.region_msg
    if args.robot_x is None or args.robot_y is None:
        if robot_xy is None:
            raise RuntimeError(
                f"No TF {args.map_frame}->{args.robot_frame}; pass --robot-x/--robot-y."
            )
        args.robot_x, args.robot_y = robot_xy
    return grid, region_json


def manual_region_available(args):
    return None not in (args.xmin, args.xmax, args.ymin, args.ymax)


def load_region(args, region_json):
    if manual_region_available(args):
        return {
            "region_id": 0,
            "xmin": args.xmin,
            "xmax": args.xmax,
            "ymin": args.ymin,
            "ymax": args.ymax,
            "status": "MANUAL",
        }
    if not region_json:
        raise RuntimeError(
            "Region required: use --region-topic, --region-json, or manual bounds."
        )
    path = Path(region_json).expanduser()
    if path.exists():
        region_json = path.read_text(encoding="utf-8")
    region = json.loads(region_json)
    required = ("xmin", "xmax", "ymin", "ymax")
    if any(key not in region for key in required):
        raise RuntimeError(f"Region JSON missing one of {required}: {region}")
    return region


def world_to_cell(grid: SimpleGrid, x: float, y: float):
    cx = int((x - grid.origin_x) / grid.resolution)
    cy = int((y - grid.origin_y) / grid.resolution)
    if cx < 0 or cy < 0 or cx >= grid.width or cy >= grid.height:
        return None
    return cx, cy


def cell_to_world(grid: SimpleGrid, x: float, y: float):
    return (
        grid.origin_x + (float(x) + 0.5) * grid.resolution,
        grid.origin_y + (float(y) + 0.5) * grid.resolution,
    )


def region_bounds_cells(grid: SimpleGrid, region: Dict):
    xmin = int(math.floor((float(region["xmin"]) - grid.origin_x) / grid.resolution))
    xmax = int(math.ceil((float(region["xmax"]) - grid.origin_x) / grid.resolution))
    ymin = int(math.floor((float(region["ymin"]) - grid.origin_y) / grid.resolution))
    ymax = int(math.ceil((float(region["ymax"]) - grid.origin_y) / grid.resolution))
    return (
        max(0, min(grid.width, xmin)),
        max(0, min(grid.width, xmax)),
        max(0, min(grid.height, ymin)),
        max(0, min(grid.height, ymax)),
    )


def nearest_free_cell_in_bounds(grid, start, bounds, radius_cells=12):
    xmin, xmax, ymin, ymax = bounds
    if start and xmin <= start[0] < xmax and ymin <= start[1] < ymax:
        if grid.data[start[1], start[0]] == FREE:
            return start
    if start is None:
        return None
    best = None
    best_dist = None
    for radius in range(1, radius_cells + 1):
        for y in range(max(ymin, start[1] - radius), min(ymax, start[1] + radius + 1)):
            for x in range(max(xmin, start[0] - radius), min(xmax, start[0] + radius + 1)):
                if grid.data[y, x] != FREE:
                    continue
                dist = math.hypot(x - start[0], y - start[1])
                if best_dist is None or dist < best_dist:
                    best = (x, y)
                    best_dist = dist
        if best is not None:
            return best
    return None


def flood_reachable_free(grid, region, robot_x, robot_y):
    bounds = region_bounds_cells(grid, region)
    xmin, xmax, ymin, ymax = bounds
    reachable = np.zeros_like(grid.data, dtype=bool)
    start = nearest_free_cell_in_bounds(
        grid,
        world_to_cell(grid, robot_x, robot_y),
        bounds,
    )
    if start is None:
        return reachable, bounds

    queue = deque([start])
    reachable[start[1], start[0]] = True
    while queue:
        x, y = queue.popleft()
        for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if nx < xmin or nx >= xmax or ny < ymin or ny >= ymax:
                continue
            if reachable[ny, nx] or grid.data[ny, nx] != FREE:
                continue
            reachable[ny, nx] = True
            queue.append((nx, ny))
    return reachable, bounds


def unknown_adjacent_to(mask, unknown):
    adjacent = np.zeros_like(mask, dtype=bool)
    adjacent[1:, :] |= mask[:-1, :]
    adjacent[:-1, :] |= mask[1:, :]
    adjacent[:, 1:] |= mask[:, :-1]
    adjacent[:, :-1] |= mask[:, 1:]
    return unknown & adjacent


def frontier_mask_from_free(free_mask, unknown_mask):
    unknown_neighbor = np.zeros_like(unknown_mask, dtype=bool)
    unknown_neighbor[1:, :] |= unknown_mask[:-1, :]
    unknown_neighbor[:-1, :] |= unknown_mask[1:, :]
    unknown_neighbor[:, 1:] |= unknown_mask[:, :-1]
    unknown_neighbor[:, :-1] |= unknown_mask[:, 1:]
    return free_mask & unknown_neighbor


def connected_components(mask, min_size):
    visited = np.zeros_like(mask, dtype=bool)
    clusters = []
    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):
            if visited[y, x] or not mask[y, x]:
                continue
            cells = []
            queue = deque([(x, y)])
            visited[y, x] = True
            while queue:
                cx, cy = queue.popleft()
                cells.append((cx, cy))
                for ny in range(cy - 1, cy + 2):
                    for nx in range(cx - 1, cx + 2):
                        if nx == cx and ny == cy:
                            continue
                        if nx < 0 or ny < 0 or nx >= mask.shape[1] or ny >= mask.shape[0]:
                            continue
                        if visited[ny, nx] or not mask[ny, nx]:
                            continue
                        visited[ny, nx] = True
                        queue.append((nx, ny))
            if len(cells) >= min_size:
                clusters.append(np.array(cells, dtype=np.int32))
    return clusters


def has_margin(grid, cell, margin_m, forbidden):
    radius = max(1, int(math.ceil(margin_m / grid.resolution)))
    cx, cy = cell
    for y in range(max(0, cy - radius), min(grid.height, cy + radius + 1)):
        for x in range(max(0, cx - radius), min(grid.width, cx + radius + 1)):
            if forbidden == "occupied" and grid.data[y, x] > 50:
                return False
            if forbidden == "unknown" and grid.data[y, x] == UNKNOWN:
                return False
    return True


def shift_toward_robot(frontier, robot_x, robot_y, shift):
    dx = robot_x - frontier[0]
    dy = robot_y - frontier[1]
    dist = math.hypot(dx, dy)
    if dist < 1e-6:
        return frontier
    shift = min(shift, max(0.0, dist - 0.05))
    return frontier[0] + shift * dx / dist, frontier[1] + shift * dy / dist


def evaluate_clusters(grid, clusters, reachable, robot_x, robot_y, args):
    selected = None
    candidates = []
    rejected = []
    stats = {
        "actionable_clusters": 0,
        "rejected_too_small": 0,
        "rejected_safety": 0,
        "rejected_unreachable": 0,
        "rejected_unknown": 0,
        "rejected_blacklisted": 0,
    }

    for index, cluster in enumerate(clusters, start=1):
        centroid_cell = (float(cluster[:, 0].mean()), float(cluster[:, 1].mean()))
        centroid_world = cell_to_world(grid, centroid_cell[0], centroid_cell[1])
        if len(cluster) < args.min_actionable_cluster_size:
            stats["rejected_too_small"] += 1
            rejected.append((centroid_world, "too_small"))
            continue

        cluster_best = None
        cluster_rejected = []
        for shift in args.goal_shifts:
            goal = shift_toward_robot(centroid_world, robot_x, robot_y, shift)
            cell = world_to_cell(grid, goal[0], goal[1])
            reason = None
            if cell is None or not reachable[cell[1], cell[0]]:
                reason = "unreachable"
            elif math.hypot(goal[0] - robot_x, goal[1] - robot_y) < args.min_goal_distance:
                reason = "too_close"
            elif not has_margin(grid, cell, args.occupied_margin, "occupied"):
                reason = "safety"
            elif not has_margin(grid, cell, args.unknown_margin, "unknown"):
                reason = "unknown"

            if reason:
                cluster_rejected.append((goal, reason))
                continue

            distance = math.hypot(goal[0] - robot_x, goal[1] - robot_y)
            score = distance - 0.02 * len(cluster)
            item = {
                "cluster_id": index,
                "cluster_size": int(len(cluster)),
                "centroid": centroid_world,
                "goal": goal,
                "score": score,
                "distance": distance,
            }
            if cluster_best is None or score < cluster_best["score"]:
                cluster_best = item

        if cluster_best is None:
            if cluster_rejected:
                reason = cluster_rejected[-1][1]
                if reason == "safety":
                    stats["rejected_safety"] += 1
                elif reason == "unknown":
                    stats["rejected_unknown"] += 1
                else:
                    stats["rejected_unreachable"] += 1
                rejected.extend(cluster_rejected)
            continue

        stats["actionable_clusters"] += 1
        candidates.append(cluster_best)
        if selected is None or cluster_best["score"] < selected["score"]:
            selected = cluster_best

    return selected, candidates, rejected, stats


def draw_mask(overlay, mask, color, alpha):
    layer = overlay.copy()
    layer[mask] = color
    cv2.addWeighted(layer, alpha, overlay, 1.0 - alpha, 0, dst=overlay)


def draw_region_rectangle(image, grid, region):
    p1 = world_to_cell(grid, float(region["xmin"]), float(region["ymin"]))
    p2 = world_to_cell(grid, float(region["xmax"]), float(region["ymax"]))
    if p1 is None or p2 is None:
        return
    cv2.rectangle(image, p1, p2, PALETTE["region"], 2)


def draw_world_point(image, grid, point, color, radius=4, thickness=-1):
    cell = world_to_cell(grid, point[0], point[1])
    if cell is None:
        return
    cv2.circle(image, cell, radius, color, thickness)


def create_visualization(
    grid,
    region,
    reachable,
    frontier,
    global_frontier,
    clusters,
    selected,
    candidates,
    rejected,
    robot_xy,
):
    image = np.zeros((grid.height, grid.width, 3), dtype=np.uint8)
    image[grid.data == FREE] = PALETTE["free"]
    image[grid.data == UNKNOWN] = PALETTE["unknown"]
    image[grid.data > 50] = PALETTE["occupied"]

    outside_frontier = global_frontier & ~frontier
    draw_mask(image, outside_frontier, PALETTE["global_frontier"], 0.85)
    draw_mask(image, reachable, PALETTE["reachable"], 0.35)
    draw_mask(image, frontier, PALETTE["frontier"], 0.95)

    for cluster in clusters:
        centroid = cell_to_world(
            grid,
            float(cluster[:, 0].mean()),
            float(cluster[:, 1].mean()),
        )
        draw_world_point(image, grid, centroid, PALETTE["frontier"], 3)

    for goal, _reason in rejected:
        draw_world_point(image, grid, goal, PALETTE["bad_goal"], 3)
    for candidate in candidates:
        draw_world_point(image, grid, candidate["goal"], PALETTE["candidate"], 4)
    if selected:
        draw_world_point(image, grid, selected["goal"], PALETTE["selected"], 6)
    draw_world_point(image, grid, robot_xy, PALETTE["robot"], 6)

    draw_region_rectangle(image, grid, region)
    return cv2.flip(image, 0)


def analyze(grid, region, robot_x, robot_y, args):
    free = grid.data == FREE
    unknown = grid.data == UNKNOWN
    reachable, bounds = flood_reachable_free(grid, region, robot_x, robot_y)
    frontier = frontier_mask_from_free(reachable, unknown)
    global_frontier = frontier_mask_from_free(free, unknown)
    clusters = connected_components(frontier, args.min_cluster_size)
    selected, candidates, rejected, goal_stats = evaluate_clusters(
        grid, clusters, reachable, robot_x, robot_y, args
    )

    xmin, xmax, ymin, ymax = bounds
    rectangle = grid.data[ymin:ymax, xmin:xmax]
    rectangle_total = int(rectangle.size)
    raw_unknown = int(np.count_nonzero(rectangle == UNKNOWN))
    reachable_unknown = unknown_adjacent_to(reachable, unknown)
    reachable_unknown_count = int(np.count_nonzero(reachable_unknown[ymin:ymax, xmin:xmax]))
    reachable_ratio = (
        float(reachable_unknown_count / rectangle_total)
        if rectangle_total else 0.0
    )
    blocked_unknown = max(0, raw_unknown - reachable_unknown_count)
    frontier_cells = int(np.count_nonzero(frontier))
    actionable = int(goal_stats["actionable_clusters"])
    completion_allowed = (
        reachable_ratio <= args.completion_threshold and actionable == 0
    )
    if completion_allowed:
        completion_reason = "NO_REACHABLE_ACTIONABLE_FRONTIER"
    elif reachable_ratio > args.completion_threshold and actionable == 0:
        completion_reason = "REACHABLE_UNKNOWN_REMAINS_BUT_NO_ACTIONABLE_FRONTIER"
    else:
        completion_reason = "ACTIONABLE_FRONTIERS_REMAIN"

    summary = {
        "region_id": region.get("region_id"),
        "robot_x": robot_x,
        "robot_y": robot_y,
        "rectangle_total_cells": rectangle_total,
        "reachable_free_cells": int(np.count_nonzero(reachable[ymin:ymax, xmin:xmax])),
        "reachable_unknown_cells": reachable_unknown_count,
        "reachable_unknown_ratio": reachable_ratio,
        "blocked_unknown_cells": blocked_unknown,
        "frontier_cells": frontier_cells,
        "clusters": len(clusters),
        "actionable_clusters": actionable,
        **goal_stats,
        "selected_goal": (
            None if selected is None else {
                "x": selected["goal"][0],
                "y": selected["goal"][1],
                "cluster_id": selected["cluster_id"],
                "cluster_size": selected["cluster_size"],
                "score": selected["score"],
            }
        ),
        "completion_allowed": completion_allowed,
        "completion_reason": completion_reason,
    }
    return summary, create_visualization(
        grid,
        region,
        reachable,
        frontier,
        global_frontier,
        clusters,
        selected,
        candidates,
        rejected,
        (robot_x, robot_y),
    )


def main():
    args = parse_args()
    if args.map_yaml:
        grid = load_saved_map(args.map_yaml)
        region_json = args.region_json
    else:
        grid, region_json = load_live_inputs(args)

    region = load_region(args, region_json)
    if args.robot_x is None or args.robot_y is None:
        raise RuntimeError("Robot pose required: use TF live mode or --robot-x/--robot-y.")

    summary, visualization = analyze(grid, region, args.robot_x, args.robot_y, args)
    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), visualization)

    json_output = (
        Path(args.json_output).expanduser()
        if args.json_output else output.with_suffix(".json")
    )
    json_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Wrote image: {output}")
    print(f"Wrote JSON: {json_output}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"debug_region_frontiers.py: {exc}", file=sys.stderr)
        sys.exit(1)
