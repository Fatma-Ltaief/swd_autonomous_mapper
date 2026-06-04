from glob import glob
import os

from setuptools import find_packages, setup


package_name = 'room_active_slam'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
         glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='fatma',
    maintainer_email='fatma@example.com',
    description=(
        'First prototype room/region analyzer for active SLAM using an '
        'occupancy grid.'
    ),
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'room_map_analyzer_node = '
            'room_active_slam.room_map_analyzer_node:main',
        ],
    },
)
