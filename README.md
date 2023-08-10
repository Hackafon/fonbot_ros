# FONBot ROS
Main ROS2 package for FONBot - robot assistant made at FON

## Setup
To get started, install ROS2 on your machine. I recommend that you do that using [Robostack bundling](https://robostack.github.io/).
Once you have ROS2 installed, clone the repository in /src directory of your ROS2 workspace.<br>

## Starting ROS2 environment
If you installed ROS2 using Robostack, start miniforge console and type the following command:
- ```mamba activate <name of ros2 environment>```

## Build
In order to build the package place yourself in your ROS2 workspace and run one of the following commands:
- ```colcon build --symlink-install```
- ```colcon build --packages-select <name of the package> --symlink-install```

Once the build has finished, source the package with the following command (make sure you are in root directory of your ROS2 workspace):
- Windows: ```call install\setup.bat```
- Linux: ```source install\setup.sh```

## Running nodes
In order to run ROS2 node type the following command:
- ```ros2 run <name of the package> <name of the executable>```<br>
In order to run launch file type the following command:
- ```ros2 launch <name of the package> <launch file>```

## Other useful commands
- Listing all ROS2 packages: ```ros2 pkg list```
- Listing all ROS2 topics: ```ros2 topic list```
- Generating URDF with xacro (download xacro from [this repository](https://github.com/ros/xacro/tree/ros2)): ```ros2 run xacro xacro -o <path for urdf> <path to xacro file>```

## RViz visualization
RViz is a ROS widget that enables us to view our URDF files as 3D models. In order to use RViz, make sure you have joint_state_publisher and joint_state_publisher_gui packages installed in your ROS2 workspace. You can download this packages from [this repository](https://github.com/ros/joint_state_publisher/tree/ros2) (check earlier section on how to build ROS2 package). Every time you start rviz, you should also run joint_state_publisher gui node (check earlier section on how to run nodes). <br>
To run RViz, type one of the following commands:
- Running RViz: ```rviz2```
- Running specific RViz file: ```rviz2 -d <path to rviz file>```

## Robot model
![](https://github.com/StefanJo3107/fonbot_ros/blob/master/models/FonbotRViz.gif)
