rosbag filter hallway.bag filtered_hallway.bag "topic != '/tf' or (topic == '/tf' and (len(m.transforms)>0 and (m.transforms[0].header.frame_id!='odom')))"