from setuptools import find_packages, setup
from glob import glob

package_name = 'speckflow_control'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.py')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='dfordequan',
    maintainer_email='dekchuen@gmail.com',
    description='Optical flow-based drone control with Speck SNN',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'flight_control = speckflow_control.flight_control_node:main',
        ],
    },
)
