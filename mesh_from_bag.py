import time

import cv2
import numpy as np
import pdb

import fusion
import os
import open3d as o3d

import message_filters
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge

import rosbag
from utils import R_axis_angle
from scipy.spatial.transform import Rotation as R

cam_intr = np.array([[612.569, 0, 319.37], 
                    [0, 612.861, 241.99], 
                    [0, 0, 1]])

def reconstruct(images, depth_images, poses):

    print("Estimating voxel volume bounds...")
    #n_imgs = len(images)
    n_imgs = len(images)

    vol_bnds = np.zeros((3,2))
    depth_threshold = .5
    padding = np.array([[0, 0, 0, 1]])

    for i in range(n_imgs):
        # Read depth image and camera pose
        depth_im = np.array(depth_images[i])
        depth_im /= 1000.  # depth is saved in 16-bit PNG in millimeters
        depth_im[depth_im > depth_threshold] = 0 
        cam_pose = np.array(poses[i])

        cam_pose = np.concatenate((cam_pose, padding), axis=0)
        cam_pose = np.linalg.inv(cam_pose)

        # Compute camera view frustum and extend convex hull
        view_frust_pts = fusion.get_view_frustum(depth_im, cam_intr, cam_pose)
        vol_bnds[:,0] = np.minimum(vol_bnds[:,0], np.amin(view_frust_pts, axis=1))
        vol_bnds[:,1] = np.maximum(vol_bnds[:,1], np.amax(view_frust_pts, axis=1))

    print(vol_bnds)

    # Initialize voxel volume
    print("Initializing voxel volume...")
    tsdf_vol = fusion.TSDFVolume(vol_bnds, voxel_size=0.002)

    # Loop through RGB-D images and fuse them together

    t0_elapse = time.time()
    for i in range(n_imgs):
        print("Fusing frame %d/%d"%(i+1, n_imgs))
        color_image = images[i]
        color_image = color_image[:, :, :]

        depth_im = np.array(depth_images[i])
        depth_im /= 1000.  # depth is saved in 16-bit PNG in millimeters
        depth_im[depth_im > depth_threshold] = 0  

        cam_pose = poses[i]
        cam_pose = np.concatenate((cam_pose, padding), axis=0)
        cam_pose = np.linalg.inv(cam_pose)
        tsdf_vol.integrate(color_image, depth_im, cam_intr, cam_pose, obs_weight=1.)

    fps = n_imgs / (time.time() - t0_elapse)
    print("Average FPS: {:.2f}".format(fps))

    # Get mesh from voxel volume and save to disk (can be viewed with Meshlab)
    print("Saving mesh to mesh.ply...")
    verts, faces, norms, colors = tsdf_vol.get_mesh()
    fusion.meshwrite("mesh.ply", verts, faces, norms, colors)

    # Get point cloud from voxel volume and save to disk (can be viewed with Meshlab)
    print("Saving point cloud to pc.ply...")
    point_cloud = tsdf_vol.get_point_cloud()
    #fusion.pcwrite("pc.ply", point_cloud)

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(point_cloud[:, :3]))
    pcd.colors = o3d.utility.Vector3dVector(point_cloud[:, 3:]/255)
    o3d.visualization.draw_geometries([pcd])

if __name__ == "__main__":

    bag_path = "/home/claude/Documents/research/tagslam/merged.bag"

    bridge = CvBridge()

    all_depth = []
    all_depth_t = []

    all_color = []
    all_color_t = []

    all_pose = []
    all_pose_t = []

    for topic, msg, t in rosbag.Bag(bag_path).read_messages():
        # read pose and find the closest image
        if topic == '/camera/aligned_depth_to_color/image_raw':
            depth_image = bridge.imgmsg_to_cv2(msg, desired_encoding='16UC1')
            all_depth.append(depth_image.astype(np.float))
            all_depth_t.append(t)

        if topic == '/camera/color/image_raw':
            cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='8UC3')
            all_color.append(cv_image)
            all_color_t.append(t)

        if topic == '/tagslam/odom/body_tag_board':
            tx, ty, tz = msg.pose.pose.position.x, \
                            msg.pose.pose.position.y, \
                            msg.pose.pose.position.z

            wx, wy, wz, ww = msg.pose.pose.orientation.x, \
                            msg.pose.pose.orientation.y, \
                            msg.pose.pose.orientation.z, \
                            msg.pose.pose.orientation.w
            #R = np.eye(3)
            #R_axis_angle(R, [wx, wy, wz], angle)

            rr = R.from_quat([wx, wy, wz, ww]).as_matrix()
            pose = np.concatenate((rr, np.array([[tx], [ty], [tz]])), axis=1)

            all_pose.append(pose)
            all_pose_t.append(t)

    all_depth_t = np.array(all_depth_t)
    all_pose_t = np.array(all_pose_t)
    all_color_t = np.array(all_color_t)

    # find the closest pose and reconstruct
    color_match = []
    depth_match = []

    for i in range(len(all_pose_t)):
        target_t = all_pose_t[i]
        diff_color_t = np.abs(all_color_t - target_t)
        diff_depth_t = np.abs(all_depth_t - target_t)

        color_t_idx = np.argmin(diff_color_t)
        depth_t_idx = np.argmin(diff_depth_t)

        color_match.append(all_color[color_t_idx])
        depth_match.append(all_depth[depth_t_idx])


        #print(i, color_t_idx, depth_t_idx)
    reconstruct(color_match, depth_match, all_pose)








