from setuptools import setup

package_name = 'fonbot_package'
setup(
name=package_name,
# data_files=[
#         ('share/ament_index/resource_index/packages',
#             ['resource/' + package_name]),
#         ('share/' + package_name, ['package.xml']),
#     ],
maintainer='Stefan',
maintainer_email='sjovanovic0831@gmail.com',
description='Main package for FONBot - robot assistant made at FON',
license='GPL-2.0 license',

entry_points={
        'console_scripts': [
                'listener = fonbot_package.subscriber_member_function:main',
        ],
},

)