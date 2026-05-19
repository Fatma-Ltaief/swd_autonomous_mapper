from setuptools import setup
from glob import glob
import os

package_name = 'swd_sim'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),

        # THIS LINE IS THE KEY
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='fatma',
    maintainer_email='fatma.ltaief@imt-nord-europe.fr',
    description='Simulation package',
    license='TODO',
    entry_points={},
)