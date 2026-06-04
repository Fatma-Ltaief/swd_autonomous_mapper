#! /usr/bin/env python3
# Copyright 2019 Samsung Research America
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import time
from collections import deque
from dataclasses import dataclass

from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav2_msgs.action import FollowWaypoints
from nav2_msgs.srv import ManageLifecycleNodes
from nav2_msgs.srv import GetCostmap
from nav2_msgs.msg import Costmap
from nav_msgs.msg  import OccupancyGrid

import rclpy
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSReliabilityPolicy
from rclpy.qos import QoSProfile
from rclpy.time import Time
from tf2_ros import Buffer, TransformException, TransformListener

from enum import Enum

import numpy as np

import math

OCC_THRESHOLD = 10
MIN_FRONTIER_SIZE = 5

class Costmap2d():
    class CostValues(Enum):
        FreeSpace = 0
        InscribedInflated = 253
        LethalObstacle = 254
        NoInformation = 255
    
    def __init__(self, map):
        self.map = map

    def getCost(self, mx, my):
        return self.map.data[self.__getIndex(mx, my)]

    def inBounds(self, mx, my):
        return 0 <= mx < self.getSizeX() and 0 <= my < self.getSizeY()

    def getSize(self):
        return (self.map.metadata.size_x, self.map.metadata.size_y)

    def getSizeX(self):
        return self.map.metadata.size_x

    def getSizeY(self):
        return self.map.metadata.size_y

    def getResolution(self):
        return self.map.metadata.resolution

    def mapToWorld(self, mx, my):
        wx = self.map.metadata.origin.position.x + (mx + 0.5) * self.map.metadata.resolution
        wy = self.map.metadata.origin.position.y + (my + 0.5) * self.map.metadata.resolution

        return (wx, wy)

    def worldToMap(self, wx, wy):
        if (wx < self.map.metadata.origin.position.x or wy < self.map.metadata.origin.position.y):
            raise Exception("World coordinates out of bounds")

        mx = int((wx - self.map.metadata.origin.position.x) / self.map.metadata.resolution)
        my = int((wy - self.map.metadata.origin.position.y) / self.map.metadata.resolution)

        if not self.inBounds(mx, my):
            raise Exception("Out of bounds")

        return (mx, my)

    def bounds(self):
        min_x = self.map.metadata.origin.position.x
        min_y = self.map.metadata.origin.position.y
        max_x = min_x + self.getSizeX() * self.map.metadata.resolution
        max_y = min_y + self.getSizeY() * self.map.metadata.resolution

        return (min_x, min_y, max_x, max_y)

    def isOccupied(self, mx, my):
        if not self.inBounds(mx, my):
            return True

        cost = self.getCost(mx, my)
        return cost >= Costmap2d.CostValues.InscribedInflated.value

    def isUnknown(self, mx, my):
        if not self.inBounds(mx, my):
            return True

        return self.getCost(mx, my) == Costmap2d.CostValues.NoInformation.value

    def __getIndex(self, mx, my):
        return my * self.map.metadata.size_x + mx

class OccupancyGrid2d():
    class CostValues(Enum):
        FreeSpace = 0
        InscribedInflated = 100
        LethalObstacle = 100
        NoInformation = -1

    def __init__(self, map):
        self.map = map

    def getCost(self, mx, my):
        return self.map.data[self.__getIndex(mx, my)]

    def inBounds(self, mx, my):
        return 0 <= mx < self.getSizeX() and 0 <= my < self.getSizeY()

    def getSize(self):
        return (self.map.info.width, self.map.info.height)

    def getSizeX(self):
        return self.map.info.width

    def getSizeY(self):
        return self.map.info.height

    def getResolution(self):
        return self.map.info.resolution

    def mapToWorld(self, mx, my):
        wx = self.map.info.origin.position.x + (mx + 0.5) * self.map.info.resolution
        wy = self.map.info.origin.position.y + (my + 0.5) * self.map.info.resolution

        return (wx, wy)

    def worldToMap(self, wx, wy):
        if (wx < self.map.info.origin.position.x or wy < self.map.info.origin.position.y):
            raise Exception("World coordinates out of bounds")

        mx = int((wx - self.map.info.origin.position.x) / self.map.info.resolution)
        my = int((wy - self.map.info.origin.position.y) / self.map.info.resolution)
        
        if not self.inBounds(mx, my):
            raise Exception("Out of bounds")

        return (mx, my)

    def bounds(self):
        min_x = self.map.info.origin.position.x
        min_y = self.map.info.origin.position.y
        max_x = min_x + self.getSizeX() * self.map.info.resolution
        max_y = min_y + self.getSizeY() * self.map.info.resolution

        return (min_x, min_y, max_x, max_y)

    def isOccupied(self, mx, my):
        if not self.inBounds(mx, my):
            return True

        return self.getCost(mx, my) >= OccupancyGrid2d.CostValues.LethalObstacle.value

    def isUnknown(self, mx, my):
        if not self.inBounds(mx, my):
            return True

        return self.getCost(mx, my) == OccupancyGrid2d.CostValues.NoInformation.value

    def __getIndex(self, mx, my):
        return my * self.map.info.width + mx

class FrontierCache():
    cache = {}

    def getPoint(self, x, y):
        idx = self.__cantorHash(x, y)

        if idx in self.cache:
            return self.cache[idx]

        self.cache[idx] = FrontierPoint(x, y)
        return self.cache[idx]

    def __cantorHash(self, x, y):
        return (((x + y) * (x + y + 1)) / 2) + y

    def clear(self):
        self.cache = {}

class FrontierPoint():
    def __init__(self, x, y):
        self.classification = 0
        self.mapX = x
        self.mapY = y

@dataclass
class FrontierCandidate:
    centroid: tuple
    goal: tuple
    size: int
    goal_cell: tuple

def centroid(arr):
    arr = np.array(arr)
    length = arr.shape[0]
    sum_x = np.sum(arr[:, 0])
    sum_y = np.sum(arr[:, 1])
    return sum_x/length, sum_y/length

def findFree(mx, my, costmap):
    fCache = FrontierCache()

    bfs = deque([fCache.getPoint(mx, my)])

    while len(bfs) > 0:
        loc = bfs.popleft()

        if costmap.getCost(loc.mapX, loc.mapY) == OccupancyGrid2d.CostValues.FreeSpace.value:
            return (loc.mapX, loc.mapY)

        for n in getNeighbors(loc, costmap, fCache):
            if n.classification & PointClassification.MapClosed.value == 0:
                n.classification = n.classification | PointClassification.MapClosed.value
                bfs.append(n)

    return (mx, my)

def nearestFreeCellToCentroid(frontier, centroid_world, costmap, fCache):
    free_cells = {}

    for point in frontier:
        for neighbor in getNeighbors(point, costmap, fCache):
            if costmap.getCost(neighbor.mapX, neighbor.mapY) == OccupancyGrid2d.CostValues.FreeSpace.value:
                free_cells[(neighbor.mapX, neighbor.mapY)] = neighbor

    if not free_cells:
        return None

    def distance_to_centroid(cell):
        wx, wy = costmap.mapToWorld(cell[0], cell[1])
        return math.hypot(wx - centroid_world[0], wy - centroid_world[1])

    return min(free_cells.keys(), key=distance_to_centroid)

def getFrontier(robot_x, robot_y, costmap, logger, min_frontier_size=MIN_FRONTIER_SIZE):
    fCache = FrontierCache()

    fCache.clear()

    try:
        mx, my = costmap.worldToMap(robot_x, robot_y)
    except Exception as e:
        logger.warn(
            f'Robot pose outside map bounds; cannot search frontiers: {e}'
        )
        return []

    freePoint = findFree(mx, my, costmap)
    start = fCache.getPoint(freePoint[0], freePoint[1])
    start.classification = PointClassification.MapOpen.value
    mapPointQueue = [start]

    frontiers = []

    while len(mapPointQueue) > 0:
        p = mapPointQueue.pop(0)

        if p.classification & PointClassification.MapClosed.value != 0:
            continue

        if isFrontierPoint(p, costmap, fCache):
            p.classification = p.classification | PointClassification.FrontierOpen.value
            frontierQueue = [p]
            newFrontier = []

            while len(frontierQueue) > 0:
                q = frontierQueue.pop(0)

                if q.classification & (PointClassification.MapClosed.value | PointClassification.FrontierClosed.value) != 0:
                    continue

                if isFrontierPoint(q, costmap, fCache):
                    newFrontier.append(q)

                    for w in getNeighbors(q, costmap, fCache):
                        if w.classification & (PointClassification.FrontierOpen.value | PointClassification.FrontierClosed.value | PointClassification.MapClosed.value) == 0:
                            w.classification = w.classification | PointClassification.FrontierOpen.value
                            frontierQueue.append(w)

                q.classification = q.classification | PointClassification.FrontierClosed.value

            
            newFrontierCords = []
            for x in newFrontier:
                x.classification = x.classification | PointClassification.MapClosed.value
                newFrontierCords.append(costmap.mapToWorld(x.mapX, x.mapY))

            if len(newFrontier) > min_frontier_size:
                frontier_centroid = centroid(newFrontierCords)
                goal_cell = nearestFreeCellToCentroid(
                    newFrontier,
                    frontier_centroid,
                    costmap,
                    fCache
                )

                if goal_cell is None:
                    logger.warn(
                        f'Rejecting frontier centroid '
                        f'x={frontier_centroid[0]:.2f}, '
                        f'y={frontier_centroid[1]:.2f}: no adjacent free goal'
                    )
                    continue

                frontiers.append(FrontierCandidate(
                    centroid=frontier_centroid,
                    goal=costmap.mapToWorld(goal_cell[0], goal_cell[1]),
                    size=len(newFrontier),
                    goal_cell=goal_cell
                ))

        for v in getNeighbors(p, costmap, fCache):
            if v.classification & (PointClassification.MapOpen.value | PointClassification.MapClosed.value) == 0:
                if any(costmap.getCost(x.mapX, x.mapY) == OccupancyGrid2d.CostValues.FreeSpace.value for x in getNeighbors(v, costmap, fCache)):
                    v.classification = v.classification | PointClassification.MapOpen.value
                    mapPointQueue.append(v)

        p.classification = p.classification | PointClassification.MapClosed.value

    return frontiers
        

def getNeighbors(point, costmap, fCache):
    neighbors = []

    for x in range(point.mapX - 1, point.mapX + 2):
        for y in range(point.mapY - 1, point.mapY + 2):
            if 0 <= x < costmap.getSizeX() and 0 <= y < costmap.getSizeY():
                neighbors.append(fCache.getPoint(x, y))

    return neighbors

def isFrontierPoint(point, costmap, fCache):
    if costmap.getCost(point.mapX, point.mapY) != OccupancyGrid2d.CostValues.NoInformation.value:
        return False

    hasFree = False
    for n in getNeighbors(point, costmap, fCache):
        cost = costmap.getCost(n.mapX, n.mapY)

        if cost > OCC_THRESHOLD:
            return False

        if cost == OccupancyGrid2d.CostValues.FreeSpace.value:
            hasFree = True

    return hasFree

class PointClassification(Enum):
    MapOpen = 1
    MapClosed = 2
    FrontierOpen = 4
    FrontierClosed = 8

class WaypointFollowerTest(Node):

    def __init__(self):
        super().__init__(node_name='nav2_waypoint_tester', namespace='')
        self.waypoints = None
        self.readyToMove = True
        self.robot_x = None
        self.robot_y = None
        self.lastWaypoint = None
        self.declare_parameter('min_frontier_size', MIN_FRONTIER_SIZE)
        self.declare_parameter('safety_margin', 0.15)
        self.declare_parameter('occupied_safety_margin', 0.20)
        self.declare_parameter('unknown_safety_margin', 0.05)
        self.declare_parameter('frontier_blacklist_duration', 45.0)
        self.declare_parameter('goal_aliasing_distance', 0.6)
        self.declare_parameter('max_planner_candidates', 8)
        self.declare_parameter('planner_timeout', 5.0)
        self.declare_parameter('max_goal_cost', 252)

        self.min_frontier_size = int(
            self.get_parameter('min_frontier_size').value
        )
        self.safety_margin = float(
            self.get_parameter('safety_margin').value
        )
        self.occupied_safety_margin = float(
            self.get_parameter('occupied_safety_margin').value
        )
        self.unknown_safety_margin = float(
            self.get_parameter('unknown_safety_margin').value
        )
        self.frontier_blacklist_duration = float(
            self.get_parameter('frontier_blacklist_duration').value
        )
        self.goal_aliasing_distance = float(
            self.get_parameter('goal_aliasing_distance').value
        )
        self.max_planner_candidates = int(
            self.get_parameter('max_planner_candidates').value
        )
        self.planner_timeout = float(
            self.get_parameter('planner_timeout').value
        )
        self.max_goal_cost = int(
            self.get_parameter('max_goal_cost').value
        )
        self.blacklisted_goals = {}
        self.global_costmap = None
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.action_client = ActionClient(self, FollowWaypoints, 'follow_waypoints')
        self.initial_pose_pub = self.create_publisher(PoseWithCovarianceStamped,
                                                      'initialpose', 10)

        self.costmapClient = self.create_client(GetCostmap, '/global_costmap/get_costmap')
        if not self.costmapClient.wait_for_service(timeout_sec=1.0):
            self.warn_msg(
                '/global_costmap/get_costmap not available yet; '
                'frontier goals will use /map validation until it appears'
            )
        self.goal_handle = None

        map_qos = QoSProfile(
          durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
          reliability=QoSReliabilityPolicy.RELIABLE,
          history=QoSHistoryPolicy.KEEP_LAST,
          depth=1)

        # self.costmapSub = self.create_subscription(Costmap, '/global_costmap/costmap_raw', self.costmapCallback, map_qos)
        self.costmapSub = self.create_subscription(OccupancyGrid, '/map', self.occupancyGridCallback, map_qos)
        self.costmap = None

        self.get_logger().info('Running Waypoint Test')
        self.get_logger().info(
            f'Frontier safety_margin={self.safety_margin:.2f} m, '
            f'occupied_safety_margin={self.occupied_safety_margin:.2f} m, '
            f'unknown_safety_margin={self.unknown_safety_margin:.2f} m, '
            f'min_frontier_size={self.min_frontier_size}, '
            f'blacklist_duration={self.frontier_blacklist_duration:.1f} s'
        )

    def occupancyGridCallback(self, msg):
        self.costmap = OccupancyGrid2d(msg)

    def moveToFrontiers(self):
        while rclpy.ok():
            robot_pose = self.lookupRobotPoseInMap()
            if robot_pose is None:
                self.warn_msg(
                    'TF map->base_footprint/base_link unavailable; '
                    'waiting before frontier search'
                )
                rclpy.spin_once(self, timeout_sec=1.0)
                continue

            self.robot_x, self.robot_y = robot_pose
            if not self.logRobotMapCell(self.robot_x, self.robot_y):
                self.warn_msg(
                    'Robot TF pose is outside /map bounds; waiting for a '
                    'new map/TF update before frontier search'
                )
                rclpy.spin_once(self, timeout_sec=1.0)
                continue

            frontiers = getFrontier(
                self.robot_x,
                self.robot_y,
                self.costmap,
                self.get_logger(),
                self.min_frontier_size
            )
            self.global_costmap = self.getGlobalCostmap()
            self.pruneBlacklist()

            if len(frontiers) == 0:
                self.info_msg('No More Frontiers')
                return

            candidates = self.validFrontierCandidates(frontiers)

            if not candidates:
                self.info_msg(
                    'No valid frontier goals after map/costmap/safety checks'
                )
                return

            goal_completed = False
            for candidate in candidates:
                if self.sendFrontierGoal(candidate):
                    goal_completed = True
                    break

                self.blacklistGoal(candidate.goal, 'Nav2 goal failed')

            if not goal_completed:
                self.warn_msg(
                    'All valid frontier goals failed or were rejected; '
                    'recomputing frontiers after blacklist update'
                )
                continue

            # Recompute frontiers from the latest map and pose after each move.

    def costmapCallback(self, msg):
        self.costmap = Costmap2d(msg)

        unknowns = 0
        for x in range(0, self.costmap.getSizeX()):
            for y in range(0, self.costmap.getSizeY()):
                if self.costmap.getCost(x, y) == 255:
                    unknowns = unknowns + 1
        self.get_logger().info(f'Unknowns {unknowns}')
        self.get_logger().info('Got global costmap update')

    def dumpCostmap(self):
        costmapReq = GetCostmap.Request()
        self.get_logger().info('Requesting Costmap')
        costmap = self.costmapClient.call(costmapReq)
        self.get_logger().info(f'costmap resolution {costmap.specs.resolution}')

    def getGlobalCostmap(self):
        if not self.costmapClient.service_is_ready():
            if not self.costmapClient.wait_for_service(timeout_sec=0.2):
                self.warn_msg(
                    '/global_costmap/get_costmap unavailable; '
                    'skipping global costmap bounds check'
                )
                return None

        request = GetCostmap.Request()
        future = self.costmapClient.call_async(request)
        rclpy.spin_until_future_complete(
            self,
            future,
            timeout_sec=self.planner_timeout
        )

        if not future.done():
            self.warn_msg(
                'Timed out fetching /global_costmap/costmap; '
                'skipping global costmap bounds check'
            )
            return None

        response = future.result()
        if response is None:
            self.warn_msg(
                'Failed to fetch /global_costmap/costmap; '
                'skipping global costmap bounds check'
            )
            return None

        return Costmap2d(response.map)

    def validFrontierCandidates(self, frontiers):
        ordered = sorted(
            frontiers,
            key=lambda candidate: self.distanceFromRobot(candidate.goal),
            reverse=True
        )

        valid = []
        for candidate in ordered:
            self.shiftCandidateIntoKnownFreeSpace(candidate)
            reason = self.rejectionReason(candidate)
            if reason is not None:
                self.logCandidate(candidate, reason)
                continue

            self.logCandidate(candidate, 'accepted')
            valid.append(candidate)

            if len(valid) >= self.max_planner_candidates:
                break

        return valid

    def rejectionReason(self, candidate):
        goal_x, goal_y = candidate.goal

        if self.isBlacklisted(candidate.goal):
            return 'temporarily blacklisted'

        if not self.isInsideGridWorld(self.costmap, goal_x, goal_y):
            return 'outside /map bounds'

        try:
            map_mx, map_my = self.costmap.worldToMap(goal_x, goal_y)
        except Exception as e:
            return f'outside /map bounds: {e}'

        if self.costmap.getCost(map_mx, map_my) != OccupancyGrid2d.CostValues.FreeSpace.value:
            return 'goal is not in known free /map space'

        map_violation = self.safetyViolation(self.costmap, map_mx, map_my)
        if map_violation is not None:
            return f'{map_violation} in /map'

        if self.global_costmap is None:
            return None

        if not self.isInsideGridWorld(self.global_costmap, goal_x, goal_y):
            return 'outside /global_costmap/costmap bounds'

        try:
            cost_mx, cost_my = self.global_costmap.worldToMap(goal_x, goal_y)
        except Exception as e:
            return f'outside /global_costmap/costmap bounds: {e}'

        if self.global_costmap.isUnknown(cost_mx, cost_my):
            return 'goal is unknown in /global_costmap'

        goal_cost = self.global_costmap.getCost(cost_mx, cost_my)
        if goal_cost >= self.max_goal_cost:
            return (
                f'global costmap goal cost {goal_cost} >= '
                f'max_goal_cost {self.max_goal_cost}'
            )

        return None

    def isInsideGridWorld(self, grid, wx, wy):
        min_x, min_y, max_x, max_y = grid.bounds()
        return min_x <= wx < max_x and min_y <= wy < max_y

    def safetyViolation(self, grid, mx, my):
        occupied = self.isTooCloseToCellType(
            grid,
            mx,
            my,
            self.occupied_safety_margin,
            'occupied'
        )
        if occupied is not None:
            return occupied

        unknown = self.isTooCloseToCellType(
            grid,
            mx,
            my,
            self.unknown_safety_margin,
            'unknown'
        )
        if unknown is not None:
            return unknown

        return None

    def isTooCloseToCellType(self, grid, mx, my, margin, cell_type):
        if margin <= 0.0:
            return None

        margin_cells = max(1, math.ceil(margin / grid.getResolution()))

        for x in range(mx - margin_cells, mx + margin_cells + 1):
            for y in range(my - margin_cells, my + margin_cells + 1):
                if not grid.inBounds(x, y):
                    if cell_type == 'occupied':
                        return (
                            f'within {margin:.2f} m of occupied cells '
                            '(map edge)'
                        )
                    return (
                        f'within {margin:.2f} m of unknown cells '
                        '(map edge)'
                    )

                distance = math.hypot(x - mx, y - my) * grid.getResolution()
                if distance > margin:
                    continue

                if cell_type == 'occupied' and grid.isOccupied(x, y):
                    return f'within {margin:.2f} m of occupied cells'

                if cell_type == 'unknown' and grid.isUnknown(x, y):
                    return f'within {margin:.2f} m of unknown cells'

        return None

    def shiftCandidateIntoKnownFreeSpace(self, candidate):
        shifted_goal = self.findShiftedGoal(candidate)
        if shifted_goal is None:
            return

        if shifted_goal == candidate.goal_cell:
            return

        old_goal = candidate.goal
        candidate.goal_cell = shifted_goal
        candidate.goal = self.costmap.mapToWorld(shifted_goal[0], shifted_goal[1])
        self.info_msg(
            f'Shifted frontier goal from x={old_goal[0]:.2f}, '
            f'y={old_goal[1]:.2f} to x={candidate.goal[0]:.2f}, '
            f'y={candidate.goal[1]:.2f} inside known free space'
        )

    def findShiftedGoal(self, candidate):
        start_mx, start_my = candidate.goal_cell
        start_violation = self.combinedSafetyViolation(start_mx, start_my)
        if start_violation is None:
            return candidate.goal_cell

        max_shift = max(1.0, self.occupied_safety_margin * 5.0)
        step_distance = self.costmap.getResolution()
        radius_cells = max(1, math.ceil(max_shift / step_distance))
        options = []

        for x in range(start_mx - radius_cells, start_mx + radius_cells + 1):
            for y in range(start_my - radius_cells, start_my + radius_cells + 1):
                if not self.costmap.inBounds(x, y):
                    continue

                shift_distance = math.hypot(
                    x - start_mx,
                    y - start_my
                ) * self.costmap.getResolution()
                if shift_distance > max_shift:
                    continue

                if self.costmap.getCost(x, y) != OccupancyGrid2d.CostValues.FreeSpace.value:
                    continue

                if self.combinedSafetyViolation(x, y) is not None:
                    continue

                wx, wy = self.costmap.mapToWorld(x, y)
                robot_distance = math.hypot(wx - self.robot_x, wy - self.robot_y)
                frontier_distance = math.hypot(
                    wx - candidate.centroid[0],
                    wy - candidate.centroid[1]
                )
                clearance = self.nearestOccupiedDistance(self.costmap, x, y, max_shift)
                options.append((
                    -clearance,
                    shift_distance,
                    frontier_distance,
                    -robot_distance,
                    x,
                    y,
                ))

        if options:
            options.sort()
            return (options[0][4], options[0][5])

        self.warn_msg(
            f'Could not shift frontier goal x={candidate.goal[0]:.2f}, '
            f'y={candidate.goal[1]:.2f} enough to satisfy safety filters; '
            f'original issue: {start_violation}'
        )
        return None

    def combinedSafetyViolation(self, map_mx, map_my):
        map_violation = self.safetyViolation(self.costmap, map_mx, map_my)
        if map_violation is not None:
            return f'{map_violation} in /map'

        if self.global_costmap is None:
            return None

        wx, wy = self.costmap.mapToWorld(map_mx, map_my)
        if not self.isInsideGridWorld(self.global_costmap, wx, wy):
            return 'outside /global_costmap/costmap bounds'

        try:
            cost_mx, cost_my = self.global_costmap.worldToMap(wx, wy)
        except Exception as e:
            return f'outside /global_costmap/costmap bounds: {e}'

        if self.global_costmap.isUnknown(cost_mx, cost_my):
            return 'goal is unknown in /global_costmap'

        goal_cost = self.global_costmap.getCost(cost_mx, cost_my)
        if goal_cost >= self.max_goal_cost:
            return (
                f'global costmap goal cost {goal_cost} >= '
                f'max_goal_cost {self.max_goal_cost}'
            )

        return None

    def nearestOccupiedDistance(self, grid, mx, my, search_radius):
        radius_cells = max(1, math.ceil(search_radius / grid.getResolution()))
        nearest = search_radius

        for x in range(mx - radius_cells, mx + radius_cells + 1):
            for y in range(my - radius_cells, my + radius_cells + 1):
                if not grid.inBounds(x, y):
                    continue

                distance = math.hypot(x - mx, y - my) * grid.getResolution()
                if distance > nearest:
                    continue

                if grid.isOccupied(x, y):
                    nearest = distance

        return nearest

    def sendFrontierGoal(self, candidate):
        while not self.action_client.wait_for_server(timeout_sec=1.0):
            self.info_msg(
                "'FollowWaypoints' action server not available, waiting..."
            )

        self.setWaypoints([candidate.goal])

        action_request = FollowWaypoints.Goal()
        action_request.poses = self.waypoints

        self.info_msg(
            f'Sending frontier goal x={candidate.goal[0]:.2f}, '
            f'y={candidate.goal[1]:.2f}'
        )
        send_goal_future = self.action_client.send_goal_async(action_request)
        try:
            rclpy.spin_until_future_complete(self, send_goal_future)
            self.goal_handle = send_goal_future.result()
        except Exception as e:
            self.error_msg('Service call failed %r' % (e,))
            return False

        if self.goal_handle is None or not self.goal_handle.accepted:
            self.error_msg('Goal rejected')
            return False

        self.info_msg('Goal accepted')

        get_result_future = self.goal_handle.get_result_async()

        self.info_msg("Waiting for 'FollowWaypoints' action to complete")
        try:
            rclpy.spin_until_future_complete(self, get_result_future)
            result_wrapper = get_result_future.result()
        except Exception as e:
            self.error_msg('Service call failed %r' % (e,))
            return False

        if result_wrapper is None:
            self.error_msg('Goal result was empty')
            return False

        status = result_wrapper.status
        result = result_wrapper.result

        if status != GoalStatus.STATUS_SUCCEEDED:
            self.warn_msg('Goal failed with status code: {0}'.format(status))
            return False

        if len(result.missed_waypoints) > 0:
            self.warn_msg(
                'Goal failed to process all waypoints, missed {0} wps.'.format(
                    len(result.missed_waypoints)
                )
            )
            return False

        self.info_msg('Goal succeeded!')
        return True

    def blacklistGoal(self, goal, reason):
        key = self.blacklistKey(goal)
        expires_at = time.monotonic() + self.frontier_blacklist_duration
        self.blacklisted_goals[key] = expires_at
        self.warn_msg(
            f'Blacklisting frontier goal x={goal[0]:.2f}, y={goal[1]:.2f} '
            f'for {self.frontier_blacklist_duration:.1f} s: {reason}'
        )

    def isBlacklisted(self, goal):
        return self.blacklistKey(goal) in self.blacklisted_goals

    def pruneBlacklist(self):
        now = time.monotonic()
        expired = [
            key for key, expires_at in self.blacklisted_goals.items()
            if expires_at <= now
        ]

        for key in expired:
            del self.blacklisted_goals[key]

    def blacklistKey(self, goal):
        return (
            round(goal[0] / self.goal_aliasing_distance),
            round(goal[1] / self.goal_aliasing_distance)
        )

    def distanceFromRobot(self, goal):
        return math.hypot(
            goal[0] - self.robot_x,
            goal[1] - self.robot_y
        )

    def formatBounds(self, grid):
        min_x, min_y, max_x, max_y = grid.bounds()
        return (
            f'x=[{min_x:.2f}, {max_x:.2f}), '
            f'y=[{min_y:.2f}, {max_y:.2f})'
        )

    def logCandidate(self, candidate, status):
        costmap_bounds = 'unavailable'
        if self.global_costmap is not None:
            costmap_bounds = self.formatBounds(self.global_costmap)

        self.info_msg(
            f'Candidate frontier goal x={candidate.goal[0]:.2f}, '
            f'y={candidate.goal[1]:.2f}; '
            f'centroid x={candidate.centroid[0]:.2f}, '
            f'y={candidate.centroid[1]:.2f}; '
            f'size={candidate.size}; '
            f'map bounds {self.formatBounds(self.costmap)}; '
            f'costmap bounds {costmap_bounds}; '
            f'{status}'
        )

    def setInitialPose(self, pose):
        self.init_pose = PoseWithCovarianceStamped()
        self.init_pose.pose.pose.position.x = pose[0]
        self.init_pose.pose.pose.position.y = pose[1]
        self.init_pose.header.frame_id = 'map'
        self.publishInitialPose()
        time.sleep(5)

    def lookupRobotPoseInMap(self):
        for source_frame in ('base_footprint', 'base_link'):
            try:
                transform = self.tf_buffer.lookup_transform(
                    'map',
                    source_frame,
                    Time(),
                    timeout=Duration(seconds=0.5)
                )
                robot_x = transform.transform.translation.x
                robot_y = transform.transform.translation.y
                self.info_msg(
                    f'Robot TF pose in map from {source_frame}: '
                    f'robot_x={robot_x:.3f}, robot_y={robot_y:.3f}'
                )
                return robot_x, robot_y
            except TransformException as e:
                self.warn_msg(
                    f'TF lookup map->{source_frame} not ready: {e}'
                )

        return None

    def logRobotMapCell(self, robot_x, robot_y):
        origin_x = self.costmap.map.info.origin.position.x
        origin_y = self.costmap.map.info.origin.position.y
        resolution = self.costmap.map.info.resolution
        width = self.costmap.map.info.width
        height = self.costmap.map.info.height
        mx = int((robot_x - origin_x) / resolution)
        my = int((robot_y - origin_y) / resolution)
        inside = 0 <= mx < width and 0 <= my < height

        self.info_msg(
            f'Robot map-cell debug: robot_x={robot_x:.3f}, '
            f'robot_y={robot_y:.3f}, map_origin_x={origin_x:.3f}, '
            f'map_origin_y={origin_y:.3f}, map_width={width}, '
            f'map_height={height}, map_resolution={resolution:.3f}, '
            f'mx={mx}, my={my}, inside_bounds={inside}'
        )

        return inside


    def setWaypoints(self, waypoints):
        self.waypoints = []
        for wp in waypoints:
            msg = PoseStamped()
            msg.header.frame_id = 'map'
            msg.pose.position.x = wp[0]
            msg.pose.position.y = wp[1]
            yaw = self.waypointYaw(wp)
            msg.pose.orientation.z = math.sin(yaw / 2.0)
            msg.pose.orientation.w = math.cos(yaw / 2.0)
            self.info_msg(
                f'Waypoint heading set to yaw={yaw:.2f} rad for '
                f'x={wp[0]:.2f}, y={wp[1]:.2f}'
            )
            self.waypoints.append(msg)

    def waypointYaw(self, waypoint):
        if self.robot_x is None or self.robot_y is None:
            return 0.0

        return math.atan2(
            waypoint[1] - self.robot_y,
            waypoint[0] - self.robot_x
        )

    def run(self, block):
        if not self.waypoints:
            rclpy.error_msg('Did not set valid waypoints before running test!')
            return False

        while not self.action_client.wait_for_server(timeout_sec=1.0):
            self.info_msg("'FollowWaypoints' action server not available, waiting...")

        action_request = FollowWaypoints.Goal()
        action_request.poses = self.waypoints

        self.info_msg('Sending goal request...')
        send_goal_future = self.action_client.send_goal_async(action_request)
        try:
            rclpy.spin_until_future_complete(self, send_goal_future)
            self.goal_handle = send_goal_future.result()
        except Exception as e:
            self.error_msg('Service call failed %r' % (e,))

        if not self.goal_handle.accepted:
            self.error_msg('Goal rejected')
            return False

        self.info_msg('Goal accepted')
        if not block:
            return True

        get_result_future = self.goal_handle.get_result_async()

        self.info_msg("Waiting for 'FollowWaypoints' action to complete")
        try:
            rclpy.spin_until_future_complete(self, get_result_future)
            status = get_result_future.result().status
            result = get_result_future.result().result
        except Exception as e:
            self.error_msg('Service call failed %r' % (e,))

        if status != GoalStatus.STATUS_SUCCEEDED:
            self.info_msg('Goal failed with status code: {0}'.format(status))
            return False
        if len(result.missed_waypoints) > 0:
            self.info_msg('Goal failed to process all waypoints,'
                          ' missed {0} wps.'.format(len(result.missed_waypoints)))
            return False

        self.info_msg('Goal succeeded!')
        return True

    def publishInitialPose(self):
        self.initial_pose_pub.publish(self.init_pose)

    def shutdown(self):
        self.info_msg('Shutting down')

        self.action_client.destroy()
        self.info_msg('Destroyed FollowWaypoints action client')

        transition_service = 'lifecycle_manager_navigation/manage_nodes'
        mgr_client = self.create_client(ManageLifecycleNodes, transition_service)
        while not mgr_client.wait_for_service(timeout_sec=1.0):
            self.info_msg(transition_service + ' service not available, waiting...')

        req = ManageLifecycleNodes.Request()
        req.command = ManageLifecycleNodes.Request().SHUTDOWN
        future = mgr_client.call_async(req)
        try:
            rclpy.spin_until_future_complete(self, future)
            future.result()
        except Exception as e:
            self.error_msg('%s service call failed %r' % (transition_service, e,))

        self.info_msg('{} finished'.format(transition_service))

        transition_service = 'lifecycle_manager_localization/manage_nodes'
        mgr_client = self.create_client(ManageLifecycleNodes, transition_service)
        while not mgr_client.wait_for_service(timeout_sec=1.0):
            self.info_msg(transition_service + ' service not available, waiting...')

        req = ManageLifecycleNodes.Request()
        req.command = ManageLifecycleNodes.Request().SHUTDOWN
        future = mgr_client.call_async(req)
        try:
            rclpy.spin_until_future_complete(self, future)
            future.result()
        except Exception as e:
            self.error_msg('%s service call failed %r' % (transition_service, e,))

        self.info_msg('{} finished'.format(transition_service))

    def cancel_goal(self):
        cancel_future = self.goal_handle.cancel_goal_async()
        rclpy.spin_until_future_complete(self, cancel_future)

    def info_msg(self, msg: str):
        self.get_logger().info(msg)

    def warn_msg(self, msg: str):
        self.get_logger().warn(msg)

    def error_msg(self, msg: str):
        self.get_logger().error(msg)


def main(argv=sys.argv[1:]):
    rclpy.init()

    # wait a few seconds to make sure entire stacks are up
    #time.sleep(10)

    test = WaypointFollowerTest()
    #test.dumpCostmap()

    while test.costmap == None:
        test.info_msg('Getting initial map')
        rclpy.spin_once(test, timeout_sec=1.0)

    test.moveToFrontiers()

    rclpy.spin(test)
    # result = test.run(True)
    # assert result

    # # preempt with new point
    # test.setWaypoints([starting_pose])
    # result = test.run(False)
    # time.sleep(2)
    # test.setWaypoints([wps[1]])
    # result = test.run(False)

    # # cancel
    # time.sleep(2)
    # test.cancel_goal()

    # # a failure case
    # time.sleep(2)
    # test.setWaypoints([[100.0, 100.0]])
    # result = test.run(True)
    # assert not result
    # result = not result

    # test.shutdown()
    # test.info_msg('Done Shutting Down.')

    # if not result:
    #     test.info_msg('Exiting failed')
    #     exit(1)
    # else:
    #     test.info_msg('Exiting passed')
    #     exit(0)


if __name__ == '__main__':
    main()
