#!/usr/bin/env python3
import argparse
import os
import struct


def _binary_vertices(data):
    if len(data) < 84:
        return None

    triangle_count = struct.unpack_from('<I', data, 80)[0]
    expected_size = 84 + triangle_count * 50
    if expected_size != len(data):
        return None

    vertices = []
    offset = 84
    for _ in range(triangle_count):
        offset += 12
        for _ in range(3):
            vertices.append(struct.unpack_from('<fff', data, offset))
            offset += 12
        offset += 2

    return vertices


def _ascii_vertices(text):
    vertices = []
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) == 4 and parts[0].lower() == 'vertex':
            vertices.append(tuple(float(value) for value in parts[1:]))
    return vertices


def read_vertices(path):
    with open(path, 'rb') as stl_file:
        data = stl_file.read()

    vertices = _binary_vertices(data)
    if vertices is not None:
        return vertices, 'binary'

    text = data.decode('utf-8', errors='ignore')
    vertices = _ascii_vertices(text)
    if vertices:
        return vertices, 'ascii'

    raise ValueError(f'Could not parse STL vertices from {path}')


def print_bounds(path):
    vertices, stl_type = read_vertices(path)
    mins = [min(vertex[axis] for vertex in vertices) for axis in range(3)]
    maxs = [max(vertex[axis] for vertex in vertices) for axis in range(3)]
    sizes = [maxs[axis] - mins[axis] for axis in range(3)]
    center = [(mins[axis] + maxs[axis]) / 2.0 for axis in range(3)]

    print(path)
    print(f'  type: {stl_type}')
    print(f'  vertices: {len(vertices)}')
    print(f'  min:    {mins[0]:.6f} {mins[1]:.6f} {mins[2]:.6f}')
    print(f'  max:    {maxs[0]:.6f} {maxs[1]:.6f} {maxs[2]:.6f}')
    print(f'  size:   {sizes[0]:.6f} {sizes[1]:.6f} {sizes[2]:.6f}')
    print(f'  center: {center[0]:.6f} {center[1]:.6f} {center[2]:.6f}')


def main():
    parser = argparse.ArgumentParser(description='Print STL bounding boxes.')
    parser.add_argument('stl_files', nargs='+')
    args = parser.parse_args()

    for index, path in enumerate(args.stl_files):
        if index:
            print()
        print_bounds(os.path.abspath(path))


if __name__ == '__main__':
    main()
