#!/usr/bin/env python3

import argparse
import logging
import math
from pathlib import Path
from xml.etree import ElementTree as ET

try:
    import cv2
except ImportError as exc:
    raise SystemExit(
        'OpenCV is required. Install it with: pip install opencv-python'
    ) from exc

try:
    import numpy as np
except ImportError as exc:
    raise SystemExit(
        'NumPy is required. Install it with: pip install numpy'
    ) from exc

try:
    import yaml
except ImportError as exc:
    raise SystemExit(
        'PyYAML is required. Install it with: pip install pyyaml'
    ) from exc


LOGGER = logging.getLogger('slam_map_to_sionna_scene')


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Convert a ROS 2 SLAM occupancy-grid map into a simplified 2.5D '
            'Sionna RT scene prototype.'
        )
    )
    parser.add_argument('--map_yaml', required=True, help='Path to map.yaml')
    parser.add_argument(
        '--output_dir',
        required=True,
        help='Directory where OBJ meshes and scene.xml will be written')
    parser.add_argument('--wall_height', type=float, default=2.5)
    parser.add_argument('--wall_thickness', type=float, default=0.10)
    parser.add_argument('--floor_thickness', type=float, default=0.05)
    return parser.parse_args()


def load_map_yaml(path):
    path = Path(path).expanduser().resolve()
    LOGGER.info('Reading map YAML: %s', path)
    with path.open('r', encoding='utf-8') as stream:
        data = yaml.safe_load(stream)

    required = ('image', 'resolution', 'origin', 'occupied_thresh',
                'free_thresh')
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f'Map YAML is missing required keys: {missing}')

    image_path = Path(data['image'])
    if not image_path.is_absolute():
        image_path = path.parent / image_path

    return {
        'yaml_path': path,
        'image_path': image_path.resolve(),
        'resolution': float(data['resolution']),
        'origin': tuple(float(value) for value in data['origin']),
        'occupied_thresh': float(data['occupied_thresh']),
        'free_thresh': float(data['free_thresh']),
        'negate': int(data.get('negate', 0)),
        'mode': data.get('mode', 'trinary'),
    }


def load_pgm_image(path):
    LOGGER.info('Loading PGM image: %s', path)
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f'Could not read map image: {path}')
    LOGGER.info('Loaded map image with shape: %s', image.shape)
    return image


def occupancy_probability(gray_image, negate):
    image_float = gray_image.astype(np.float32) / 255.0
    if negate:
        return image_float
    return 1.0 - image_float


def build_wall_mask(gray_image, map_info, wall_thickness):
    occupied = occupancy_probability(gray_image, map_info['negate'])
    wall_mask = (occupied >= map_info['occupied_thresh']).astype(np.uint8) * 255

    occupied_cells = int(np.count_nonzero(wall_mask))
    LOGGER.info('Initial occupied wall cells: %d', occupied_cells)

    # Close small gaps and remove isolated speckles before contour extraction.
    cleanup_kernel = np.ones((3, 3), dtype=np.uint8)
    wall_mask = cv2.morphologyEx(wall_mask, cv2.MORPH_CLOSE, cleanup_kernel,
                                 iterations=1)
    wall_mask = cv2.morphologyEx(wall_mask, cv2.MORPH_OPEN, cleanup_kernel,
                                 iterations=1)

    # Use the requested wall thickness as a prototype minimum thickness. This
    # is intentionally simple: it grows the mask in 2D before extrusion.
    dilation_radius = max(0, int(math.ceil(
        (wall_thickness / map_info['resolution']) / 2.0)))
    if dilation_radius > 0:
        kernel_size = 2 * dilation_radius + 1
        thickness_kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        wall_mask = cv2.dilate(wall_mask, thickness_kernel, iterations=1)
        LOGGER.info(
            'Dilated wall mask by %d px to approximate %.3f m thickness',
            dilation_radius,
            wall_thickness)

    LOGGER.info('Cleaned wall cells: %d', int(np.count_nonzero(wall_mask)))
    return wall_mask


def extract_simplified_contours(wall_mask):
    contours, _ = cv2.findContours(
        wall_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    LOGGER.info('Found %d raw wall contours', len(contours))

    simplified = []
    for contour in contours:
        area_px = cv2.contourArea(contour)
        if area_px < 4.0:
            continue

        epsilon = max(1.0, 0.01 * cv2.arcLength(contour, closed=True))
        approx = cv2.approxPolyDP(contour, epsilon, closed=True)
        points = approx.reshape(-1, 2)
        if len(points) >= 3:
            simplified.append(points)

    LOGGER.info('Kept %d simplified wall contours', len(simplified))
    return simplified


def pixel_to_world(point, image_height, map_info):
    x_px, y_px = float(point[0]), float(point[1])
    resolution = map_info['resolution']
    origin_x, origin_y, origin_yaw = map_info['origin'][:3]

    # Image coordinates start at top-left. ROS map coordinates start at the
    # bottom-left map origin, so y is flipped.
    local_x = x_px * resolution
    local_y = (image_height - y_px) * resolution

    cos_yaw = math.cos(origin_yaw)
    sin_yaw = math.sin(origin_yaw)
    world_x = origin_x + local_x * cos_yaw - local_y * sin_yaw
    world_y = origin_y + local_x * sin_yaw + local_y * cos_yaw
    return world_x, world_y


def extrude_contours_to_mesh(contours, image_height, map_info, wall_height):
    vertices = []
    faces = []

    for contour in contours:
        base_index = len(vertices) + 1
        bottom = []
        top = []

        for point in contour:
            world_x, world_y = pixel_to_world(point, image_height, map_info)
            bottom.append((world_x, world_y, 0.0))
            top.append((world_x, world_y, wall_height))

        vertices.extend(bottom)
        vertices.extend(top)
        count = len(contour)

        for index in range(count):
            next_index = (index + 1) % count
            faces.append((
                base_index + index,
                base_index + next_index,
                base_index + count + next_index,
                base_index + count + index,
            ))

        # Add simple caps. OBJ supports n-gon faces, which is good enough for
        # this prototype and keeps the mesh human-readable.
        faces.append(tuple(base_index + index for index in reversed(range(count))))
        faces.append(tuple(base_index + count + index for index in range(count)))

    LOGGER.info('Wall mesh: %d vertices, %d faces', len(vertices), len(faces))
    return vertices, faces


def floor_mesh(image_shape, map_info, floor_thickness):
    height_px, width_px = image_shape
    corners_px = (
        (0, height_px),
        (width_px, height_px),
        (width_px, 0),
        (0, 0),
    )
    top = [
        (*pixel_to_world(point, height_px, map_info), 0.0)
        for point in corners_px
    ]
    bottom = [(x, y, -floor_thickness) for x, y, _ in top]
    vertices = top + bottom
    faces = [
        (1, 2, 3, 4),
        (8, 7, 6, 5),
        (1, 5, 6, 2),
        (2, 6, 7, 3),
        (3, 7, 8, 4),
        (4, 8, 5, 1),
    ]
    LOGGER.info('Floor mesh covers %.2f m x %.2f m',
                width_px * map_info['resolution'],
                height_px * map_info['resolution'])
    return vertices, faces


def write_obj(path, vertices, faces, object_name):
    LOGGER.info('Writing OBJ: %s', path)
    with Path(path).open('w', encoding='utf-8') as stream:
        stream.write(f'# Generated by slam_map_to_sionna_scene.py\n')
        stream.write(f'o {object_name}\n')
        for x, y, z in vertices:
            stream.write(f'v {x:.6f} {y:.6f} {z:.6f}\n')
        for face in faces:
            stream.write('f ' + ' '.join(str(index) for index in face) + '\n')


def add_material(parent, material_id, color):
    bsdf = ET.SubElement(parent, 'bsdf', {
        'type': 'diffuse',
        'id': material_id,
    })
    ET.SubElement(bsdf, 'rgb', {
        'name': 'reflectance',
        'value': color,
    })


def add_obj_shape(parent, obj_file, material_id):
    shape = ET.SubElement(parent, 'shape', {'type': 'obj'})
    ET.SubElement(shape, 'string', {
        'name': 'filename',
        'value': obj_file,
    })
    ET.SubElement(shape, 'ref', {
        'id': material_id,
        'name': 'bsdf',
    })


def write_scene_xml(path, wall_obj, floor_obj):
    LOGGER.info('Writing Sionna/Mitsuba-style scene XML: %s', path)
    scene = ET.Element('scene', {'version': '3.0.0'})
    add_material(scene, 'wall_material', '0.75, 0.75, 0.75')
    add_material(scene, 'floor_material', '0.45, 0.45, 0.45')
    add_obj_shape(scene, wall_obj, 'wall_material')
    add_obj_shape(scene, floor_obj, 'floor_material')

    tree = ET.ElementTree(scene)
    ET.indent(tree, space='  ')
    tree.write(path, encoding='utf-8', xml_declaration=True)


def convert(map_yaml, output_dir, wall_height, wall_thickness,
            floor_thickness):
    map_info = load_map_yaml(map_yaml)
    image = load_pgm_image(map_info['image_path'])
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info('Writing output scene directory: %s', output_dir)

    wall_mask = build_wall_mask(image, map_info, wall_thickness)
    contours = extract_simplified_contours(wall_mask)

    wall_vertices, wall_faces = extrude_contours_to_mesh(
        contours, image.shape[0], map_info, wall_height)
    floor_vertices, floor_faces = floor_mesh(
        image.shape, map_info, floor_thickness)

    wall_obj = output_dir / 'walls.obj'
    floor_obj = output_dir / 'floor.obj'
    scene_xml = output_dir / 'scene.xml'
    mask_png = output_dir / 'wall_mask_debug.png'

    write_obj(wall_obj, wall_vertices, wall_faces, 'slam_walls')
    write_obj(floor_obj, floor_vertices, floor_faces, 'slam_floor')
    cv2.imwrite(str(mask_png), wall_mask)
    write_scene_xml(scene_xml, wall_obj.name, floor_obj.name)

    LOGGER.info('Done. Outputs:')
    LOGGER.info('  walls: %s', wall_obj)
    LOGGER.info('  floor: %s', floor_obj)
    LOGGER.info('  scene: %s', scene_xml)
    LOGGER.info('  debug mask: %s', mask_png)


def main():
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
    args = parse_args()
    convert(
        map_yaml=args.map_yaml,
        output_dir=args.output_dir,
        wall_height=args.wall_height,
        wall_thickness=args.wall_thickness,
        floor_thickness=args.floor_thickness,
    )


if __name__ == '__main__':
    main()
