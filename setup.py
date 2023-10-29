from setuptools import setup

package_name = 'fonbot_ros'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Stefan',
    maintainer_email='sjovanovic0831@gmail.com',
    description='Main ROS2 package for FONBot - robot assistant made at FON',
    license='GPL-2.0 license',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'listener = fonbot_ros.subscriber_member_function:main',
            'camera = fonbot_ros.subscriber_camera:main',
        ],
    },
)
