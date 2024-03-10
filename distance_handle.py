#!/usr/bin/env python3
import rospy
from std_msgs.msg import Float32

class DistanceInterpolator:
    def __init__(self):
        rospy.init_node('distance_interpolator_node')
        self.distance_sub = rospy.Subscriber("/distance", Float32, self.callback)
	
        # Assuming you have defined distanceValueP, steeringValueP, distanceValueN, and steeringValueN
        # Replace these placeholders with actual values or load them from ROS parameters

        # Example placeholders (replace with actual data)
        distanceValueP = [0.1, 0.5, 1.0]  # Example positive distance values
        steeringValueP = [10.0, 20.0, 30.0]  # Example corresponding steering values
        distanceValueN = [-0.1, -0.5, -1.0]  # Example negative distance values
        steeringValueN = [-10.0, -20.0, -30.0]  # Example corresponding steering values

        self.distance_handle_p = {
            'SIZE': len(distanceValueP),  # Assuming both positive and negative have the same size
            'distance': 0.5,  # Example distance value (replace with actual input)
            'steering': 0.0  # Initialize steering
        }
	
	
        self.interpolate_distance()

    def callback(self):
        if self.distance_handle_p['distance'] > 0:
            # Positive distance interpolation
            for i in range(self.distance_handle_p['SIZE'] - 1):
                if distanceValueP[i] <= self.distance_handle_p['distance'] <= distanceValueP[i + 1]:
                    self.distance_handle_p['steering'] = (
                        (self.distance_handle_p['distance'] - distanceValueP[i]) * steeringValueP[i + 1] +
                        (distanceValueP[i + 1] - self.distance_handle_p['distance']) * steeringValueP[i]
                    ) / (distanceValueP[i + 1] - distanceValueP[i])
                    return self.distance_handle_p['steering']

        elif self.distance_handle_p['distance'] < 0:
            # Negative distance interpolation
            for i in range(self.distance_handle_p['SIZE'] - 1):
                if distanceValueN[i] >= self.distance_handle_p['distance'] >= distanceValueN[i + 1]:
                    self.distance_handle_p['steering'] = (
                        (self.distance_handle_p['distance'] - distanceValueN[i]) * steeringValueN[i + 1] +
                        (distanceValueN[i + 1] - self.distance_handle_p['distance']) * steeringValueN[i]
                    ) / (distanceValueN[i + 1] - distanceValueN[i])
                    return self.distance_handle_p['steering']

        return 0.0





if __name__ == '__main__':
    try:
        DistanceInterpolator()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

