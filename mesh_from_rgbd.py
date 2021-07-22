"""Fuse 1000 RGB-D images from the 7-scenes dataset into a TSDF voxel volume with 2cm resolution.
"""

import time

import cv2
import numpy as np
import pdb

import fusion
import os
import open3d as o3d


if __name__ == "__main__":

    print("Estimating voxel volume bounds...")
    data_dir = os.path.join("/home/claude/Documents/research/mesh_realsense/build/images")

    # read meta file
    f = open(os.path.join(data_dir, "poses.txt"))
    lines = f.readlines()
    n_imgs = len(lines)

    vol_bnds = np.zeros((3,2))
    cam_intr = np.array([[612.569, 0, 319.37], 
                        [0, 612.861, 241.99], 
                        [0, 0, 1]])
    depth_threshold = .5

    padding = np.array([[0, 0, 0, 1]])

    for i in range(len(lines)):

        l = lines[i].strip('\n')
        fields = l.split(" ")

        # Read depth image and camera pose
        depth_im = cv2.imread(os.path.join(data_dir, fields[0] + ".png"), -1).astype(float)
        depth_im /= 1000.  # depth is saved in 16-bit PNG in millimeters
        depth_im[depth_im > depth_threshold] = 0 
        cam_pose = np.array(fields[1:13]).reshape([3, 4]).astype(np.float)
        cam_pose = np.concatenate((cam_pose, padding), axis=0)
        cam_pose = np.linalg.inv(cam_pose)

        # Compute camera view frustum and extend convex hull
        view_frust_pts = fusion.get_view_frustum(depth_im, cam_intr, cam_pose)
        vol_bnds[:,0] = np.minimum(vol_bnds[:,0], np.amin(view_frust_pts, axis=1))
        vol_bnds[:,1] = np.maximum(vol_bnds[:,1], np.amax(view_frust_pts, axis=1))

    print(vol_bnds)

    # Initialize voxel volume
    print("Initializing voxel volume...")
    tsdf_vol = fusion.TSDFVolume(vol_bnds, voxel_size=0.001)

    # Loop through RGB-D images and fuse them together

    debug = False
    t0_elapse = time.time()
    for i in range(n_imgs):
        print("Fusing frame %d/%d"%(i+1, n_imgs))

        # Read RGB-D image and camera pose
        l = lines[i].strip('\n')
        fields = l.split(" ")

        # Read depth image and camera pose
        color_image = cv2.imread(os.path.join(data_dir, fields[0] + "_color.png"))
        color_image = color_image[:, :, [2, 1, 0]]

        depth_im = cv2.imread(os.path.join(data_dir, fields[0] + ".png"), -1).astype(float)
        depth_im /= 1000.  # depth is saved in 16-bit PNG in millimeters
        depth_im[depth_im > depth_threshold] = 0  
        cam_pose = np.array(fields[1:13]).reshape([3, 4]).astype(np.float)
        cam_pose = np.concatenate((cam_pose, padding), axis=0)

        # test this
        cam_pose = np.linalg.inv(cam_pose)

        # Integrate observation into voxel volume (assume color aligned with depth)
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
    fusion.pcwrite("pc.ply", point_cloud)

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(point_cloud[:, :3]))
    pcd.colors = o3d.utility.Vector3dVector(point_cloud[:, 3:]/255)
    o3d.visualization.draw_geometries([pcd])
