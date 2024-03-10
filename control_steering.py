#!/usr/bin/env python3

import rospy
from std_msgs.msg import Float32
from math import fabs

# Randomly assigned values (replace with actual values)
KP_VALUE = 1.2
KI_VALUE = 0.5
KD_VALUE = 0.8
TS_VALUE = 0.01
INF_VALUE = 100.0
SUP_VALUE = 25.0

class ControlNode:
    def __init__(self):
        rospy.init_node('control_node')
        self.control_p = {
            'error': 0.0,
            'pre_error': 0.0,
            'pre2_error': 0.0,
            'pre_Out': 0.0,
            'Kp': KP_VALUE,  # Replace with actual values
            'Ki': KI_VALUE,
            'Kd': KD_VALUE,
            'Ts': TS_VALUE,
            'Inf_Saturation': INF_VALUE,
            'Sup_Saturation': SUP_VALUE
        }

        # Create a publisher for the control output
        self.control_pub = rospy.Publisher('control_output', Float32, queue_size=10)

        # Subscribe to the reference topic (replace 'reference_topic' with the actual topic name)
        rospy.Subscriber('reference_topic', Float32, self.reference_callback)
        

    def reference_callback(self, msg):
        # Error calculation
        if msg.data > 0:
            self.control_p['error'] = msg.data - fabs(self.control_p['angle1'] - self.control_p['angle0'])
        elif msg.data < 0:
            self.control_p['error'] = msg.data + fabs(self.control_p['angle1'] - self.control_p['angle0'])

        # PID control calculations
        self.control_p['Kp_part'] = self.control_p['Kp'] * (self.control_p['error'] - self.control_p['pre_error'])
        self.control_p['Ki_part'] = 0.5 * self.control_p['Ki'] * self.control_p['Ts'] * (self.control_p['error'] + self.control_p['pre_error'])
        self.control_p['Kd_part'] = (self.control_p['Kd'] / self.control_p['Ts']) * (self.control_p['error'] - 2 * self.control_p['pre_error'] + self.control_p['pre2_error'])

        # Calculate control output
        self.control_p['Out'] = self.control_p['pre_Out'] + self.control_p['Kp_part'] + self.control_p['Ki_part'] + self.control_p['Kd_part']

        # Boundary output
        if self.control_p['Out'] > self.control_p['Inf_Saturation']:
            self.control_p['Out'] = self.control_p['Inf_Saturation']
        elif self.control_p['Out'] < self.control_p['Sup_Saturation']:
            self.control_p['Out'] = self.control_p['Sup_Saturation']

        # Update pre errors
        self.control_p['pre_Out'] = self.control_p['Out']
        self.control_p['pre2_error'] = self.control_p['pre_error']
        self.control_p['pre_error'] = self.control_p['error']

        # Publish control output
        self.control_pub.publish(self.control_p['Out'])
        
    def PID_CONTROL_Init(control_p, set_control_func):
        control_p['set_control'] = set_control_func
        PID_CONTROL_Reset_Parameters(control_p)
        PID_CONTROL_Set_Parameters(control_p)
        control_p['angle1'] = 0.0
        control_p['angle0'] = 0.0
        PID_CONTROL_Set_control(control_p)

    def PID_CONTROL_Create():
        control_p = {
        'error': 0.0,
        'pre_error': 0.0,
        'pre2_error': 0.0,
        'pre_Out': 0.0,
        'Kp': KP_VALUE,  # Replace with actual values
        'Ki': KI_VALUE,
        'Kd': KD_VALUE,
        'Ts': TS_VALUE,
        'Inf_Saturation': INF_VALUE,
        'Sup_Saturation': SUP_VALUE
        }
        PID_CONTROL_Init(control_p, PID_CONTROL_Set_control)
        return control_p

    def PID_CONTROL_Destroy(control_p):
        # No need to do anything special for memory deallocation in Python
        pass


if __name__ == '__main__':
    try:
        ControlNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
