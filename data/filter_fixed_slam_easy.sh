rosbag filter fixed_slam_easy.bag filtered_slam_easy.bag "topic != '/tf' or (topic == '/tf' and (len(m.transforms)>0 and (m.transforms[0].header.frame_id!='odom')))"