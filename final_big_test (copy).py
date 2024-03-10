class CameraHandler():
    # ===================================== INIT ==========================================
    def __init__(self):
        """		
        Creates a bridge for converting the image from Gazebo image into OpenCV image.
        """
        rospy.init_node('distancehandler', anonymous=True)
        self.bridge = CvBridge()
        self.cv_image = np.zeros((640, 480))
        rospy.init_node('CAMnod', anonymous=True)
        self.image_sub = rospy.Subscriber("/image_raw", Image, self.callback)
        self.publisher = rospy.Publisher("/distance", Float32, queue_size=1)
        self.rate = rospy.Rate(1) # Publish at 1 Hz
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
        #print('distance =',distance)
        # Publish the calculated distance
        self.publisher.publish(distance)
        rospy.loginfo(f"Distance: {distance:.2f} meters")
        create_new_lane_if_needed(140, 140, distance)

        # Display the result
        cv2.imshow('Lane Detection', result)

        # Check for key presses
        key = cv2.waitKey(1)

        # Pause or resume the video on 'p' key press
        if key == ord('p'):
            pause_video()

        # Exit the loop if 'q' key is pressed
        elif key == ord('q'):
            cv2.destroyAllWindows()
            rospy.signal_shutdown("Shutdown")
