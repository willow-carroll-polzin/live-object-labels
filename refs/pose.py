## Based on code by: Apache 2.0

###DEPRICATED - Jan 31, 2020

# First import the library
import pyrealsense2 as rs
import math as m
import time

def pose_calculation(robot_x, robot_y, direction):
    RADIUS = 1

    # Declare RealSense pipeline, encapsulating the actual device and sensors
    pipe = rs.pipeline()

    # Build config object and request pose data
    cfg = rs.config()
    cfg.enable_stream(rs.stream.pose)

    # Start streaming with requested config
    pipe.start(cfg)

    try:
        # Wait for the next set of frames from the camera
        frames = pipe.wait_for_frames()

        # Fetch pose frame
        pose = frames.get_pose_frame()
        if pose:
            data = pose.get_pose_data()
            x = data.translation.x
            y = data.translation.y
            z = data.translation.z
            rw = data.rotation.w
            rx = -data.rotation.z
            ry = data.rotation.x
            rz = -data.rotation.y
            xz_distance = m.sqrt(x*x+z*z)
            pitch =  -m.asin(2.0 * (rx*rz - rw*ry)) * 180.0 / m.pi;
            roll  =  m.atan2(2.0 * (rw*rx + ry*rz), rw*rw - rx*rx - ry*ry + rz*rz) * 180.0 / m.pi;
            yaw   =  m.atan2(2.0 * (rw*rz + rx*ry), rw*rw + rx*rx - ry*ry - rz*rz) * 180.0 / m.pi;
            
            x_offset = RADIUS * m.sin(-yaw*m.pi/180)
            z_offset = -x_offset/ m.tan((180-yaw)/2*m.pi/180)
            
            mx = x + x_offset
            mz = z + z_offset
            m_dist = m.sqrt(mx*mx + mz*mz)
            
            #print("Camera has travelled: {0:.3f} metres (Camera X: {0:.3f}, Camera Z: {0:.3f})".format(xz_distance,x,z))
            #print("Yaw is: {0:.3f}".format(yaw))
            #print("X Offset is: {0:.3f} and Y Offset is: {0:.3f}".format(x_offset,z_offset))
            #print("Robot has travelled: {0:.3f} metres (Robot X: {0:.3f}, Robot Z: {0:.3f})".format(m_dist, mx, mz))
            time.sleep(2)
    finally:
        pipe.stop()
#adding a test comment2
#robot_x = 0
#robot_y = 0
#direction = 0

#pose_calculation(robot_x, robot_y, direction)

#print("Robot X: {0:.2f}, Robot Y: {0:.2f}, Direction: {0:.2f}".format(robot_x, robot_y, direction))

