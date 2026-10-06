from setuptools import find_packages, setup

package_name = 'speck_flow'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='dfordequan',
    maintainer_email='dekchuen@gmail.com',
    description='Speck DVS optical flow estimation using spiking neural networks.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'speck_flow_node = speck_flow.speck_flow_node:main',
        ],
    },
)
