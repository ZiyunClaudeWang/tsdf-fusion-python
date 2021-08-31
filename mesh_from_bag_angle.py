import time

import cv2
import numpy as np
import pdb
import pickle

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

def plot_axis(cv_image, t_cam_table):

    cf = np.array([[0, 0, 0],
                        [0.1, 0, 0],
                        [0, 0.1, 0],
                        [0, 0, 0.1]]).T
    cf = np.concatenate((cf, np.ones([1, 4])), axis=0)
    cf_cam = t_cam_table.dot(cf)

    # project points
    cf_pj = cam_intr.dot(cf_cam[:3, :])

    cf_pj = cf_pj / (1e-9 + cf_pj[2:3, :])

    # draw line
    cf_pj = cf_pj.T
    cv_image = cv2.line(cv_image, (int(cf_pj[0, 0]), int(cf_pj[0, 1])), 
                                        (int(cf_pj[1, 0]), int(cf_pj[1, 1])), 
                                        (255, 0, 0), 3)

    cv_image = cv2.line(cv_image, (int(cf_pj[0, 0]), int(cf_pj[0, 1])), 
                                        (int(cf_pj[2, 0]), int(cf_pj[2, 1])), 
                                        (0, 255, 0), 3)

    cv_image = cv2.line(cv_image, (int(cf_pj[0, 0]), int(cf_pj[0, 1])), 
                                        (int(cf_pj[3, 0]), int(cf_pj[3, 1])), 
                                        (0, 0, 255), 3)
    return cv_image


def reconstruct(images, depth_images, poses, obj_name):

    print("Estimating voxel volume bounds...")
    #n_imgs = len(images)
    n_imgs = len(images)

    vol_bnds = np.zeros((3,2))
    depth_threshold = .6

    for i in range(n_imgs):
        # Read depth image and camera pose
        depth_im = np.array(depth_images[i])
        depth_im /= 1000.  # depth is saved in 16-bit PNG in millimeters
        depth_im[depth_im > depth_threshold] = 0 
        cam_pose = np.array(poses[i])

        cam_pose = np.linalg.inv(cam_pose)

        # Compute camera view frustum and extend convex hull
        view_frust_pts = fusion.get_view_frustum(depth_im, cam_intr, cam_pose)
        vol_bnds[:,0] = np.minimum(vol_bnds[:,0], np.amin(view_frust_pts, axis=1))
        vol_bnds[:,1] = np.maximum(vol_bnds[:,1], np.amax(view_frust_pts, axis=1))

    print(vol_bnds)

    # Initialize voxel volume
    print("Initializing voxel volume...")
    tsdf_vol = fusion.TSDFVolume(vol_bnds, voxel_size=0.0025)

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
        cam_pose = np.linalg.inv(cam_pose)

        tsdf_vol.integrate(color_image, depth_im, cam_intr, cam_pose, obs_weight=1.)

    fps = n_imgs / (time.time() - t0_elapse)
    print("Average FPS: {:.2f}".format(fps))

    # Get mesh from voxel volume and save to disk (can be viewed with Meshlab)
    print("Saving mesh to mesh.ply...")

    # points on the ground should be removed
    verts, faces, norms, colors = tsdf_vol.get_mesh()
    fusion.meshwrite("{}.ply".format(obj_name), verts, faces, norms, colors)

    # Get point cloud from voxel volume and save to disk (can be viewed with Meshlab)
    print("Saving point cloud to pc.ply...")
    point_cloud = tsdf_vol.get_point_cloud()
    fusion.pcwrite("{}_pc.ply".format(obj_name), point_cloud)

    '''
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(point_cloud[:, :3]))
    pcd.colors = o3d.utility.Vector3dVector(point_cloud[:, 3:]/255)
    o3d.visualization.draw_geometries([pcd])
    '''

if __name__ == "__main__":

    '''
    bag_path = "/home/claude/Documents/data/mesh_recons/test/soup_no_tag.bag"
    pose_dict_path = "/home/claude/Documents/data/mesh_recons/test/recons_data_2021-08-18-20-22-37.pkl"
    '''
    bag_path = "/home/claude/Documents/data/mesh_recons/2021-08-30/objects/beer.bag"
    pose_dict_path = "/home/claude/Documents/data/mesh_recons/2021-08-30/match_pose_100.pkl"

    '''
    bag_path = "/home/claude/Documents/data/mesh_recons/2021-08-30/match_angle_to_pose.bag"
    pose_dict_path = "/home/claude/Documents/data/mesh_recons/2021-08-30/match_angle_to_pose.pkl"
    '''


    with open(pose_dict_path, 'rb') as f:
        pose_dict = pickle.load(f)

    bridge = CvBridge()

    all_depth = []
    all_depth_t = []

    all_color = []
    all_color_t = []

    all_angle = []
    all_pose_t = []
    all_pose = []

    for topic, msg, t in rosbag.Bag(bag_path).read_messages():
        # read pose and find the closest image
        if topic == '/camera/aligned_depth_to_color/image_raw':
            depth_image = bridge.imgmsg_to_cv2(msg, desired_encoding='16UC1')
            all_depth.append(depth_image.astype(float))
            all_depth_t.append(t.to_sec())

        if topic == '/camera/color/image_raw':
            cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='8UC3')
            all_color.append(cv_image)
            all_color_t.append(t.to_sec())

        if topic == '/joint_states':
            # find the closes in keys 
            all_angle.append(msg.position[0])
            all_pose_t.append(t.to_sec())

    # pose from another file
    all_depth_t = np.array(all_depth_t)
    all_pose_t = np.array(all_pose_t)
    all_color_t = np.array(all_color_t)

    def angle_diff(angle_a, angle_b):
        diff = angle_b - angle_a
        return (diff + np.pi) % (np.pi * 2) - np.pi

    # find the closest match in 
    keys = np.array(list(pose_dict.keys()))
    for i in range(all_pose_t.shape[0]):
        diff = np.abs(angle_diff(keys, all_angle[i]))
        idx = np.argmin(diff)
        print(diff[idx])
        all_pose.append(pose_dict[keys[idx]])
    all_pose = np.array(all_pose)

    # find the closest pose and reconstruct

    depth_match = []
    pose_match = []
    color_match = []

    for i in range(len(all_depth)):
        depth_match.append(all_depth[i])

        target_t = all_color_t[i]
        diff_pose_t = np.abs(all_pose_t - target_t)
        pose_t_idx = np.argmin(diff_pose_t)

        diff_color_t = np.abs(all_color_t - target_t)
        color_t_idx = np.argmin(diff_color_t)

        pose_match.append(all_pose[pose_t_idx])
        color_match.append(all_color[color_t_idx])

        cv_image = np.array(all_color[color_t_idx])

        # project a list of points based on pose estimate and K
        t_cam_table = all_pose[pose_t_idx]
        cv_image = plot_axis(cv_image, t_cam_table)
        #cv_image = plot_axis(cv_image, t_cam_table_good)

        '''
        cv2.imshow('image', cv_image)
        cv2.waitKey(0)
        '''

    print(len(color_match), len(depth_match), len(pose_match))
    reconstruct(color_match, depth_match, pose_match, obj_name = bag_path.strip(".bag"))







