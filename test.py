import time

import cv2
import numpy as np
import pdb
import os
import open3d as o3d

#612.569 612.861 319.37 241.99
cam_intr = np.array([[612.569, 0, 319.37], 
                        [0, 612.861, 241.99], 
                        [0, 0, 1]])


data_dir = os.path.join("/home/claude/Documents/research/mesh_realsense/build/images")

# read meta file
point_cloud = np.loadtxt(os.path.join(data_dir, "points.txt"))
z_dist = np.abs(point_cloud[:, 2])
#print(z_dist.min(), z_dist.max(), z_dist.mean())
point_cloud = point_cloud[z_dist < 1., :]
print(point_cloud[:,2].mean())

xx_norm = point_cloud[:, 0] / point_cloud[:, 2]
yy_norm = point_cloud[:, 1] / point_cloud[:, 2]

xx_pix = (point_cloud[:, 0] / point_cloud[:, 2]) * cam_intr[0, 0] + cam_intr[0, 2]
yy_pix = (point_cloud[:, 1] / point_cloud[:, 2]) * cam_intr[1, 1] + cam_intr[1, 2]

img = np.ones([480, 640, 3])
img[yy_pix.astype(int),xx_pix.astype(int), :] = point_cloud[:, 3:]/255.

#cv2.imshow("project", img)
#cv2.waitKey(0)


#print(point_cloud[:, 3:].max(), point_cloud[:, 3:].mean())

pcd_gt = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(point_cloud[:, :3]))
pcd_gt.colors = o3d.utility.Vector3dVector(point_cloud[:, 3:]/255)
#o3d.visualization.draw_geometries([pcd_gt])

depth_threshold = 1.
f = open(os.path.join(data_dir, "poses.txt"))
lines = f.readlines()
n_imgs = len(lines)


i = 0
# Read RGB-D image and camera pose
l = lines[i].strip('\n')
fields = l.split(" ")

# Read depth image and camera pose
color_image = cv2.imread(os.path.join(data_dir, fields[0] + "_color.png"))

depth_im = cv2.imread(os.path.join(data_dir, fields[0] + ".png"), -1).astype(float)
depth_im /= 1000.  # depth is saved in 16-bit PNG in millimeters
#print(depth_im.min(), depth_im.max(), depth_im[depth_im>0].mean())
#depth_im[depth_im > depth_threshold] = 0

x, y = np.meshgrid(np.arange(color_image.shape[1]),
                    np.arange(color_image.shape[0]))

valid = np.logical_and(depth_im > 0, depth_im <= depth_threshold)

x, y = x[valid], y[valid]

x = x.flatten()
y = y.flatten()

x_norm = (x - cam_intr[0, 2]) / cam_intr[0, 0]
y_norm = (y - cam_intr[1, 2]) / cam_intr[1, 1]
z = depth_im[y, x]

point_cloud = np.stack((x_norm * z, y_norm * z, z), axis=-1)
color = color_image[y, x, :]

pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(point_cloud))
#pcd.colors = o3d.utility.Vector3dVector(color[:, [2, 1, 0]]/255)
o3d.visualization.draw_geometries([pcd, pcd_gt])




