# Copyright 2016 Open Source Robotics Foundation, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2


class ImagePublisher(Node):
    def __init__(self):
        super().__init__('publisher_camera')
        self.publisher_ = self.create_publisher(Image,'camera_driver',10)
        timer_period = 0.1
      
        self.timer = self.create_timer(timer_period, self.timer_callback)

        self.cap = cv2.VideoCapture(0)

        self.br = CvBridge()
    
    def timer_callback(self):
        ret, frame = self.cap.read()

        if ret:
            frame = cv2.flip(frame, 0)
            self.publisher_.publish(self.br.cv2_to_imgmsg(frame))
            self.get_logger().info('Publishing the frame')
        else:
            self.get_logger().info("Can't receive the frame...")




def main(args=None):
    rclpy.init(args=args)

    publisher_camera=ImagePublisher()

    rclpy.spin(publisher_camera)

    publisher_camera.destroy_node()
    rclpy.shutdown()
    

if __name__=='__main__':
    main()
