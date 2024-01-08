import rospy
import collections
from visualization_msgs.msg import Marker, MarkerArray

class Trajectory:
    def __init__(self):
        self.zero_queue = collections.deque([0]*8, maxlen=8)
        self.obs_traj = []
        self.peds_in_curr_seq = 0
        
        self.publisher = rospy.Publisher('/predic_trajectories', MarkerArray, queue_size=8)
        
    def get_obs_traj(markarray):
        
        pass
    
    def publish_marker_array(self, data):
        
        marker_array = MarkerArray()
        max_ped_num = data.shape[0]
        max_pos_num = data.shape[1]
        print("data:", data.shape)
        # print(data)

        for pos_idx in range(data.shape[0]):
            print("pos_idx:", pos_idx)
            marker = Marker()
            marker.header.frame_id = "map"
            marker.type = marker.SPHERE
            marker.action = marker.ADD
            marker.scale.x = 0.3
            marker.scale.y = 0.3
            marker.scale.z = 0.3
            marker.color.a = 1.0
            marker.color.b = 0.0
            marker.pose.orientation.w = 1.0
            marker.pose.position.z = 0
            for ped_idx in range(data.shape[1]):
                marker.color.r = ped_idx / float(max_ped_num)
                marker.color.g = pos_idx / float(max_pos_num)
                marker.pose.position.x = data[pos_idx, ped_idx, 0]
                marker.pose.position.y = data[pos_idx, ped_idx, 1]
                marker.id = 20 * (ped_idx+1) * data.shape[1] + pos_idx  # 设置唯一的id
                
                marker_array.markers.append(marker)

        self.publisher.publish(marker_array)