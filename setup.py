import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'asv_perception'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        # Ament index
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Launch files
        (os.path.join('share', package_name, 'launch'),
            glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
        # Config files
        (os.path.join('share', package_name, 'config'),
            glob(os.path.join('config', '*.yaml')) +
            glob(os.path.join('config', '*.rviz'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='shubham',
    maintainer_email='shubhambarge.dev@gmail.com',
    description='Coastal hazard detection and autonomous patrol for ASVs '
                'using LiDAR + camera fusion on VRX/WAM-V.',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'hazard_detector_node = asv_perception.hazard_detector_node:main',
            'lidar_processor_node = asv_perception.lidar_processor_node:main',
            'sensor_fusion_node = asv_perception.sensor_fusion_node:main',
            'reactive_planner_node = asv_perception.reactive_planner_node:main',
        ],
    },
)
