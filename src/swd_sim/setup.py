from setuptools import setup
from glob import glob
import os

package_name = 'swd_sim'


def package_files(directory):
    paths = []
    for path, _, filenames in os.walk(directory):
        files = [os.path.join(path, filename) for filename in filenames]
        if files:
            paths.append((os.path.join('share', package_name, path), files))
    return paths


setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),

        (os.path.join('share', package_name, 'launch'),
         sorted(set(glob('launch/*.launch.py') + glob('launch/*.py')))),
        (os.path.join('share', package_name, 'config'),
         glob('config/*')),
        (os.path.join('share', package_name, 'worlds'),
         glob('worlds/*')),
    ] + package_files('models'),
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='fatma',
    maintainer_email='fatma.ltaief@imt-nord-europe.fr',
    description='Simulation package',
    license='TODO',
    entry_points={
        'console_scripts': [
            'wander_node = swd_sim.wander_node:main',
        ],
    },
)
