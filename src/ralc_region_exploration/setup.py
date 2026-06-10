from glob import glob
import os

from setuptools import find_packages, setup


package_name = 'ralc_region_exploration'

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
    description='Prototype R-ALC-inspired rectangular region exploration.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'region_manager_node = '
            'ralc_region_exploration.region_manager_node:main',
            'ralc_frontier_planner_node = '
            'ralc_region_exploration.ralc_frontier_planner_node:main',
            'ralc_executive_node = '
            'ralc_region_exploration.ralc_executive_node:main',
            'ralc_alc_planner_node = '
            'ralc_region_exploration.ralc_alc_planner_node:main',
            'ralc_pgs_planner_node = '
            'ralc_region_exploration.ralc_pgs_planner_node:main',
            'ralc_exploration_manager_node = '
            'ralc_region_exploration.ralc_exploration_manager_node:main',
            'ralc_slam_backend_node = '
            'ralc_region_exploration.ralc_slam_backend_node:main',
            'scan_min_range_filter_node = '
            'ralc_region_exploration.scan_min_range_filter_node:main',
        ],
    },
)
