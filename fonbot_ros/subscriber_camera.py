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
import numpy as np
import cv2
import numpy as np
import dlib

face_detector = dlib.get_frontal_face_detector() 

shape_predictor_path = "models/face_landmarks.dat" 
resnet_predictor_path = "models/resnet_model_v1.dat"
pose_predictor_68_point = dlib.shape_predictor(shape_predictor_path)
face_encoder = dlib.face_recognition_model_v1(resnet_predictor_path)

def _trim_css_to_bounds(css, image_shape):
    return max(css[0], 0), min(css[1], image_shape[1]), min(css[2], image_shape[0]), max(css[3], 0)

def compare_faces(known_face_encodings, face_encoding_to_check, tolerance=0.6):
    return list(distance(known_face_encodings, face_encoding_to_check) <= tolerance)

def distance(face_encodings:list, face_to_compare:np.ndarray)->np.ndarray:
    """
    Computes the Euclidean distance between face encodings.

    Parameters:
    face_encodings (List[numpy.ndarray]): List of face encodings.
    face_to_compare (numpy.ndarray): The face encoding to compare against.

    Returns:
    numpy.ndarray: Array of distances between face encodings and the target face.
    """
    if len(face_encodings) == 0:
        return np.empty((0))

    return np.linalg.norm(face_encodings - face_to_compare, axis=1)

def face_locations(img: np.ndarray, number_of_times_to_upsample: int = 1) -> list:
    """
    Detects faces in an image and returns the locations of detected faces.

    Parameters:
    img (numpy.ndarray): The input image.
    number_of_times_to_upsample (int): The number of times to upsample the image.

    Returns:
    List[dlib.rectangle]: A list of rectangles representing the detected face locations.
    """
    return [_trim_css_to_bounds((face.top(), face.right(), face.bottom(), face.left()), img.shape) for face in face_detector(img, number_of_times_to_upsample)]

def _raw_face_landmarks(face_image: np.ndarray, face_locations: list = None) -> list:
    """
    Detects facial landmarks in an image.

    Parameters:
    face_image (numpy.ndarray): The face image.
    face_locations (List[dlib.rectangle]): Optional list of face locations.

    Returns:
    List[dlib.full_object_detection]: A list of facial landmark points for each detected face.
    """
    if face_locations is None:
        face_locations = face_detector(face_image)
    else:
        face_locations = [dlib.rectangle(face_location[3], face_location[0], face_location[1], face_location[2]) for face_location in face_locations]

    return [pose_predictor_68_point(face_image, face_location) for face_location in face_locations]


def face_encodings(face_image: np.ndarray, known_face_locations: list = None, num_jitters: int = 1)->list:
    """
    Computes face encodings (feature vectors) for the input face image.

    Parameters:
    face_image (numpy.ndarray): The input face image.
    known_face_locations (List[dlib.rectangle]): Optional list of known face locations.
    num_jitters (int): Number of jitters for encoding calculation.

    Returns:
    List[numpy.ndarray]: A list of computed face encodings.
    """
    raw_landmarks = _raw_face_landmarks(face_image, known_face_locations)
    return [np.array(face_encoder.compute_face_descriptor(face_image, raw_landmark_set, num_jitters)) for raw_landmark_set in raw_landmarks]


##################################### CONFIGURE ############################################################################
face_image_01 = cv2.imread("Database/placeholder_01.jpg")[:,:,::-1]
face_image_02 = cv2.imread("Database/placeholder_02.jpg")[:,:,::-1]

known_face_encodings = [face_encodings(np.array(face_image_01))[0], face_encodings(np.array(face_image_02))[0]]
known_names = ["name_of_person_image_01", "name_of_person_image_02"]
#############################################################################################################################

class MinimalSubscriber(Node):
    def __init__(self, known_face_encodings, known_face_names):
        super().__init__('minimal_subscriber')
        self.subscription = self.create_subscription(
            Image, 
            'camera_driver',
            self.listener_callback,
            10)
        self.subscription  
        self.known_face_encodings = known_face_encodings
        self.known_face_names = known_face_names
        # self.unknown_counter = 0

    @staticmethod
    def get_image_arr(data: np.array, n: int = 3)->list:
        """
        Breaks list in list of lists with n elements 
        ex. from [value_01,value_02,value_03,value_04,value_05,value_06,...] to 
        [[value_01,value_02,value_03],[value_04,value_05,value_06],...]

        Parameters:
        data (np.array): The input list.
        n (int): Number of elements in each new list.

        Returns:
        list: A list of lists with n elements.
        """
        if n == 1920:
            x = [data[i * n:(i + 1) * n] for i in range((len(data) + n - 1) // n )]
            return x
        
        temp = [data[i * n:(i + 1) * n] for i in range((len(data) + n - 1) // n )]
        return MinimalSubscriber.get_image_arr(temp, n=1920)
    
    def listener_callback(self, msg):
        
        processed_frame = np.rot90(MinimalSubscriber.get_image_arr(np.array(msg.data)),2)
        
        scaled_frame_bgr = cv2.resize(processed_frame, (0, 0), fx=0.25, fy=0.25)
        scaled_frame_rgb = np.ascontiguousarray(scaled_frame_bgr[:,:,::-1])
        
        face_loc = face_locations(scaled_frame_rgb)
        
        face_enc = face_encodings(scaled_frame_rgb, face_loc)
        face_names = []
        
        for face_encoding in face_enc:
            matches = compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown"

            # If a match was found in known_face_encodings, just use the first one.
            if True in matches:
                first_match_index = matches.index(True)
                name = self.known_face_names[first_match_index]
            # can be used for clustering
            # if name=="Unknown":
            #     cv2.imwrite(f'image{self.unknown_counter}.jpg', temp[:,:,::-1])
            #     self.unknown_counter+=1
            face_names.append(name)
            
        print(face_names)


def main(args=None):
    rclpy.init(args=args)
    
    minimal_subscriber = MinimalSubscriber(known_face_encodings, known_names)
    
    rclpy.spin(minimal_subscriber)

    minimal_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
