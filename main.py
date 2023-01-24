import os
import argparse
import json
import glob
import cv2
import numpy as np
import open3d as o3d
import trimesh
from PIL import Image
import pyrender
from natsort import natsorted
import matplotlib.pyplot as plt
from visualization import Visualizer

def align_color2depth(o3d_color, o3d_depth):
    color_data = np.asarray(o3d_color)
    depth_data = np.asarray(o3d_depth)
    scale = [np.shape(depth_data)[0]/np.shape(color_data)[0], \
        np.shape(depth_data)[1]/np.shape(color_data)[1]]
    if scale != [1.0, 1.0]:
        color = Image.fromarray(color_data)
        depth = Image.fromarray(depth_data)
        color = color.resize(depth.size)
        return o3d.geometry.Image(np.asarray(color)), scale, np.shape(depth_data)
    return o3d_color, scale, np.shape(depth_data)

def read_cameras(camera_file, scale_x=1, scale_y=1):
    intrinsics = []
    extrinsics = []
    with open(camera_file, 'r') as fp:
        for line in fp:
            cam_info = json.loads(line)
            C= cam_info.get('transform', None) # ARKit pose (+x along long axis of device toward home button, +y upwards, +z away from device)
            assert C!= None
            C= np.asarray(C)
            C= C.reshape(4, 4).transpose()

            C= np.matmul(C, np.diag([1, -1, -1, 1])) # open3d camera pose (flip y and z)
            C= C / C[3][3]
            extrinsics.append(np.linalg.inv(C))

            K = cam_info.get('intrinsics')
            K = np.asarray(K)
            K = K.reshape(3, 3).transpose()
            scale = np.array([scale_x, scale_y, 1.0])
            K_depth = np.matmul(np.diag(scale), K)
            intrinsics.append(K_depth)

    return intrinsics, extrinsics

def render(viz, color_path, depth_path, intrinsic, extrinsic, view_angle=0, update_view=False):
    o3d_intrinsic = o3d.camera.PinholeCameraIntrinsic(shape[1], shape[0], intrinsic[0, 0], intrinsic[1, 1], intrinsic[0,2], intrinsic[1,2])
    color = o3d.io.read_image(color_path)
    depth = o3d.io.read_image(depth_path)
    color, scale, shape = align_color2depth(color, depth)
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color, depth, convert_rgb_to_intensity=False)
    tmp_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, o3d_intrinsic, extrinsic)
    pcd = pyrender.Mesh.from_points(np.asarray(tmp_pcd.points), colors=np.asarray(tmp_pcd.colors))
    pcd_node = pyrender.Node(mesh=pcd)
    viz.add_node(pcd_node)
    if update_view:
        viz.scene_centroid = viz.scene.centroid
        viz.scene_scale = viz.scene.scale
        viz.initialize_camera()
        viz.set_camera_pose_by_angle(view_angle)
    rgb, _ = viz.render()
    viz.remove_node(pcd_node)
    return rgb

def main(args):
    color_images = natsorted(glob.glob(args.color))
    depth_images = natsorted(glob.glob(args.depth))
    tmp_color = o3d.io.read_image(color_images[0])
    tmp_depth = o3d.io.read_image(depth_images[0])
    _, scale, shape = align_color2depth(tmp_color, tmp_depth)
    intrinsics, extrinsics = read_cameras(args.camera, scale[0], scale[1])
    assert len(color_images) == len(depth_images), "ERROR: Different number of color images and depth images"
    assert len(intrinsics) == len(depth_images), "ERROR: Different number of in cameras and depth images"
    viz = Visualizer()
    # initialize camera pose
    start_frame = args.start_frame
    render(viz, color_images[start_frame], depth_images[start_frame], intrinsics[start_frame], extrinsics[start_frame], args.view_angle, update_view=True)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 60
    video = cv2.VideoWriter(args.output, fourcc, fps, (viz.resolution[0], viz.resolution[1]))
    for i in range(len(depth_images)):
        color_path = color_images[i]
        depth_path = depth_images[i]
        intrinsic = intrinsics[i]
        extrinsic = extrinsics[i]
        rgb_frame = render(viz, color_path, depth_path, intrinsic, extrinsic, args.view_angle)
        cv_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        video.write(cv_frame)
    video.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert RGBD images to a point cloud!')
    parser.add_argument('-c', '--color', required=True, \
        help='Input RGB color images directory')
    parser.add_argument('-d', '--depth', required=True, \
        help='Input depth maps directory')
    parser.add_argument('--camera', type=str, required=True,
        help='Camera intrinsic parameters')
    parser.add_argument('--view_angle', type=float, required=False, default=0,
        help='Render view angle')
    parser.add_argument('--start_frame', type=int, required=False, default=0,
        help='Starting frame used to compute the camera pose of the render view')
    parser.add_argument('-o', '--output', required=True, \
        help='Output video path')

    args = parser.parse_args()
    
    main(args)