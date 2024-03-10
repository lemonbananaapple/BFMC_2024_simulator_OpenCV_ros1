#!/usr/bin/env python3

import rospy
import numpy as np
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from utils.msg import IMU
from cv_bridge import CvBridge, CvBridgeError
import cv2
from Lane_detection import Lane  # Replace 'your_module_name' with the actual name of your module
import pickle
import time

distance_values_p = [1.0, 5.0, 7.0, 10.0]
distance_values_n = [-1.0, -5.0, -7.0, -10.0]
steering_values_p = [1.0, 8.0, 15.0, 20.0]
steering_values_n = [-1.0, -8.0, -15.0, -20.0]

# Function to pause and resume video
def pause_video():
    while True:
        key = cv2.waitKey(0)
        if key == ord('p'):
            break

# Initialize the camera calibration parameters
# with open('/home/lucis/PycharmProject/Bosch_car/Lane_detection/calibration_data.pkl', 'rb') as f:
#     objpoints, imgpoints, chessboards = pickle.load(f)
# ret, mtx, dist, dst = camera_calibrate(objpoints, imgpoints, img)
# Your camera calibration function (camera_calibrate) should be called here to get calibration parameters

# Initialize Lane object
lane = Lane(img_height=360, img_width=640)  # Replace with actual calibration parameters


# Hàm khởi tạo một lane mới
# Định nghĩa các biến toàn cục
lane_threshold = 0  # Giả sử ban đầu không có ngưỡng nào

def create_new_lane_if_needed(distance_threshold,distance_threshold_far, distance):
    global lane
    global lane_threshold

    # Lấy thời gian hiện tại
    current_time = cv2.getTickCount()

    # Kiểm tra nếu khoảng cách nhỏ hơn ngưỡng
    if distance < distance_threshold or distance > distance_threshold_far:
        # Khởi tạo một lane mới
        lane.road_line_left = None
        lane.road_line_right = None
        # Cập nhật lại ngưỡng để tránh tạo lane liên tục trong một khoảng thời gian ngắn
        lane_threshold = current_time + 5 * cv2.getTickFrequency()  # Ví dụ: sau 5 giây mới được tạo lane mới


class CameraHandler():
    # ===================================== INIT ==========================================
    def __init__(self):
        """		
        Creates a bridge for converting the image from Gazebo image into OpenCV image.
        """
        self.distance_queue = []  # Initialize an empty list to store data
        self.steering_queue = []
        self.queue_size = 3  # Set the desired queue size
        self.distance_global = 0
        self.steering_global = 0
        self.steering = 0
        self.distance = 0
        self.bridge = CvBridge()
        self.cv_image = np.zeros((640, 480))
        rospy.init_node('CAMnod', anonymous=True)
        self.image_sub = rospy.Subscriber("/image_raw", Image, self.callback)
        self.imu_sub = rospy.Subscriber("/automobile/IMU", IMU, self.imu_callback)
        #self.publisher = rospy.Publisher("/distance", Float32, queue_size=1)
        #self.rate = rospy.Rate(1) # Publish at 1 Hz
        #############################################Dang TUNE
        # if self.distance_global > 0:
        #     if self.distance_global < self.distance_values_p[0]:
        #         return 0
        #     if self.distance_global >= self.distance_values_p[-1]:
        #         return self.steering_values_p[-1]
        #     else:
        #         for i in range(len(self.distance_values_p) - 1):
        #             if self.distance_values_p[i] <= self.distance_global <= self.distance_values_p[i + 1]:
        #                 self.steering = ((self.distance_global - self.distance_values_p[i]) * self.steering_values_p[i + 1] +
        #                                             (self.distance_values_p[i + 1] - self.distance_global) * self.steering_values_p[i]) / \
        #                                            (self.distance_values_p[i + 1] - self.distance_values_p[i])
        #                 return self.steering
        # elif self.distance_global < 0:
        #     if self.distance_global > self.distance_values_n[0]:
        #         return 0
        #     if self.distance_global <= self.distance_values_n[-1]:
        #         return self.steering_values_n[-1]
        #     else:
        #         for i in range(len(self.distance_values_n) - 1):
        #             if self.distance_values_n[i] >= self.distance_global >= self.distance_values_n[i + 1]:
        #                 self.steering = ((self.distance_global - self.distance_values_n[i]) * self.steering_values_n[i + 1] +
        #                                             (self.distance_values_n[i + 1] - self.distance_global) * self.steering_values_n[i]) / \
        #                                            (self.distance_values_n[i + 1] - self.distance_values_n[i])
        #                 return self.steering (cho nay nen publish cho mot node khac)
        # return 0 
        ##########################
        rospy.spin()

    def callback(self, data):
        """
        :param data: sensor_msg array containing the image in the Gazebo format
        :return: nothing but sets [cv_image] to the useful image that can be used in OpenCV (numpy array)
        """
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        frame = self.cv_image
        # Apply lane detection on the frame
        lane.image = frame  # This will trigger the processing pipeline in your Lane class

        # Get the decorated result with lane information
        result, distance = lane.result_decorated
        self.distance_global = distance
        #print('distance =',distance)
        # Publish the calculated distance
        # self.publisher.publish(distance)
        # show the distance
        #rospy.loginfo(f"Distance: {distance:.2f} meters") 
        
        create_new_lane_if_needed(140, 140, distance)

        # # Append the new data to the queue
        # self.distance_queue.append(distance)

        # # If the queue size exceeds the desired size, remove the oldest value
        # if len(self.distance_queue) > self.queue_size:
        #     self.distance_queue.pop(0)  # Remove the first element (oldest data)

        # Display the result
        cv2.imshow('Lane Detection', result)

	######################################################ADDED
        
    # distanceValueP = [1.0, 5.0, 7.0, 10.0]
    # distanceValueN = [-1.0, -5.0, -7.0, -10.0]
    # steeringValueP = [1.0, 8.0, 15.0, 20.0]
    # steeringValueN = [-1.0, -8.0, -15.0, -20.0]

    # for i in range(len(distanceValueP) - 1):
    #     if distanceValueP[i] <= distance <= distanceValueP[i + 1]:
    #             distance_handle_p.steering = ((distance_handle_p.distance - distanceValueP[i]) * steeringValueP[i + 1] +
    #                                 (distanceValueP[i + 1] - distance_handle_p.distance) * steeringValueP[i]) / \
    #                                 (distanceValueP[i + 1] - distanceValueP[i])

    # for i in range(len(distanceValueN) - 1):
    #     if distanceValueN[i] >= distance_handle_p.distance >= distanceValueN[i + 1]:
    #             distance_handle_p.steering = ((distance_handle_p.distance - distanceValueN[i]) * steeringValueN[i + 1] +
    #                                 (distanceValueN[i + 1] - distance_handle_p.distance) * steeringValueN[i]) / \
    #                                 (distanceValueN[i + 1] - distanceValueN[i])
        
	
    def imu_callback(self, msg):
        # Extract the desired fields from the IMU message
        roll = msg.roll
        pitch = msg.pitch
        yaw = msg.yaw
        accel_x = msg.accelx
        accel_y = msg.accely
        accel_z = msg.accelz
        self.steering_global = yaw

        # Print the extracted values
        rospy.loginfo(f"Roll: {roll:.2f} degrees") 
        rospy.loginfo(f"Pitch: {pitch:.2f} degrees") 
        rospy.loginfo(f"Yaw: {yaw:.2f} degrees") 
        rospy.loginfo(f"Linear Acceleration (X): {accel_x:.2f} m/s^2") 
        rospy.loginfo(f"Linear Acceleration (Y): {accel_y:.2f} m/s^2") 
        rospy.loginfo(f"Linear Acceleration (Z): {accel_z:.2f} m/s^2") 


        


if __name__ == '__main__':
    try:
        nod = CameraHandler()
    except rospy.ROSInterruptException:
        pass
