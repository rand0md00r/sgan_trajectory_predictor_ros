#!/usr/bin/env python
# -*- coding: utf-8 -*-
#!/usr/bin/env python

import rospy
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
import math

class CircleMotionNode:
    def __init__(self):
        rospy.init_node('circle_motion_node', anonymous=True)
        self.marker_array_pub = rospy.Publisher('motion_markers', MarkerArray, queue_size=1)

    def publish_marker_array(self):
        rate = rospy.Rate(5)  # 1 Hz
        while not rospy.is_shutdown():
            marker_array = MarkerArray()

            # Define three circle centers
            centers = [(0.0, 0.0), (6.0, 0.0), (0.0, 6.0)]

            for i, (center_x, center_y) in enumerate(centers):
                marker = Marker()
                marker.header.frame_id = "os_sensor"
                marker.header.stamp = rospy.Time.now()
                marker.ns = str(i)
                marker.id = 0
                marker.type = Marker.CUBE
                marker.action = Marker.ADD

                # Calculate position on the circle
                angle = rospy.get_time() * 0.3  # Adjust the factor for speed
                radius = 3.0
                marker.pose.position.x = center_x + radius * math.cos(angle+ i * 2.0 * math.pi / 3.0) #  + i * 2.0 * math.pi / 3.0
                marker.pose.position.y = center_y + radius * math.sin(angle+ i * 2.0 * math.pi / 3.0)
                marker.pose.position.z = 0.5
                marker.pose.orientation.w = 1.0
                marker.scale.x = 0.5
                marker.scale.y = 0.5
                marker.scale.z = 1.0
                marker.color.a = 0.5
                marker.color.r = 1.0 if i == 0 else 0.0
                marker.color.g = 1.0 if i == 1 else 0.0
                marker.color.b = 1.0 if i == 2 else 0.0
                marker_array.markers.append(marker)

            self.marker_array_pub.publish(marker_array)
            rospy.loginfo("Published three fake poses.")
            rate.sleep()

if __name__ == '__main__':
    try:
        node = CircleMotionNode()
        node.publish_marker_array()
    except rospy.ROSInterruptException:
        pass
