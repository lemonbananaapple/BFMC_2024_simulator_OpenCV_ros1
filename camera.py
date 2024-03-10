#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

def draw_lines(img, vertices):
    pts = np.int32([vertices])
    cv2.polylines(img, pts, True, (0, 0, 255))

def perspective_transforms(src, dst):
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    return M, Minv

def perspective_warp(img, M):
    img_size = (img.shape[1], img.shape[0])
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    return warped

def calc_warp_points(img_height, img_width, xfd, yf, x_center_adj=0):
    imshape = (img_height, img_width)
    xcenter = imshape[1] / 2 + x_center_adj
    xoffset = 0

    src = np.float32([
        (xoffset, imshape[0]),
        (xcenter - xfd, yf),
        (xcenter + xfd, yf),
        (imshape[1] - xoffset, imshape[0])
    ])

    dst = np.float32([
        (xoffset, imshape[1]),
        (xoffset, 0),
        (imshape[0] - xoffset, 0),
        (imshape[0] - xoffset, imshape[1])
    ])

    return src, dst

def value():
    pass

cv2.namedWindow('Trackbar Example')
cv2.createTrackbar('xfd', 'Trackbar Example', 0, 2000, value)
cv2.createTrackbar('yf', 'Trackbar Example', 0, 2000, value)
cv2.createTrackbar('x_left', 'Trackbar Example', 0, 2000, value)
cv2.createTrackbar('x_right', 'Trackbar Example', 5, 2000, value)
cv2.createTrackbar('y_top', 'Trackbar Example', 0, 2000, value)
cv2.createTrackbar('y_bottom', 'Trackbar Example', 5, 2000, value)

class CameraHandler():
    # ===================================== INIT==========================================
    def __init__(self):
        """
        Creates a bridge for converting the image from Gazebo image intro OpenCv image
        """
        self.bridge = CvBridge()
        self.cv_image = np.zeros((640, 480))
        rospy.init_node('CAMnod', anonymous=True)
        self.image_sub = rospy.Subscriber("/image_raw", Image, self.callback)
        rospy.spin()

    def callback(self, data):
        """
        :param data: sensor_msg array containing the image in the Gazsbo format
        :return: nothing but sets [cv_image] to the usefull image that can be use in opencv (numpy array)
        """
        self.cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        frame=self.cv_image
        xfd = cv2.getTrackbarPos('xfd', 'Trackbar Example')
        yf = cv2.getTrackbarPos('yf', 'Trackbar Example')
        x_left_value = cv2.getTrackbarPos('x_left', 'Trackbar Example')
        x_right_value = cv2.getTrackbarPos('x_right', 'Trackbar Example')
        y_top_value = cv2.getTrackbarPos('y_top', 'Trackbar Example')
        y_bottom_value = cv2.getTrackbarPos('y_bottom', 'Trackbar Example')

        # Draw lines on the original frame before extracting ROI
        src, dst = calc_warp_points(frame.shape[0], frame.shape[1], xfd, yf)
        draw_lines(frame, src)

        # Extract ROI
        frame_roi = frame[y_top_value:y_bottom_value, x_left_value:x_right_value]
        cv2.imshow("Region of Interest", frame_roi)

        M, _ = perspective_transforms(src, dst)
        warped = perspective_warp(frame_roi, M)
        warped = cv2.cvtColor(warped, cv2.COLOR_BGRA2BGR)

        cv2.imshow('Birdeye view', warped)
        """cv2.imshow("Frame preview", self.cv_image)"""
        key = cv2.waitKey(1)


if __name__ == '__main__':
    try:
        nod = CameraHandler()
    except rospy.ROSInterruptException:
        pass
