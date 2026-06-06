import sys
import os
sys.path.append("./")

current_dir = os.path.dirname(os.path.abspath(__file__))
rgen_path = os.path.join(current_dir, 'control_your_robot') 
rgen_path = os.path.abspath(rgen_path) 
sys.path.append(rgen_path)

from control_your_robot.my_robot.franka_sim_curobo_robot import FrankaSimRobot  # TODO

# from control_your_robot.my_robot.convert_obj_to_usd import convert_obj_to_usd

os.environ["INFO_LEVEL"] = "INFO" # DEBUG , INFO, ERROR
import time
import numpy as np
from scipy.spatial.transform import Rotation as R
from math import pi, cos, sin, radians


import os
import open3d as o3d
import numpy as np
from tqdm import tqdm
import trimesh
import copy
import h5py
import argparse
import cv2
import torch
from copy import deepcopy
import random
import imageio
from scipy.spatial.transform import Rotation as R
import torch
import torch.nn.functional as F
import math

from gen_utils.vis_utils import  point_cloud_to_video
from gen_utils.sam_utils import get_objects_pcd_from_sam_mask
from gen_utils.traj_utils import generate_trans_vectors, interpolate_positions_numpy
from gen_utils.pcd_utils import pcd_bbox, pcd_divide, pcd_translate, trans_pcd, apply_rotation
from gen_utils.obj_utils import convert_mesh_to_o3d, find_best_ratio_combination, trans_3d_to_uv, load_ply_minmax, get_point_forward, get_point_extent
from gen_utils.path_utils import load_h5_data
from gen_utils.render_utils import render_point_cloud_sequence, render_point_cloud_sequence_with_depth, render_point_cloud_sequence_multithread
from gen_utils.ground_utils import segment_ground_and_compute_normals, extract_ground_points_with_normals
from gen_utils.exec_utils import load_functions_from_txt
from gen_utils.frame_utils import read_video_to_torch, pad_to_bg, save_video
from gen_utils.cuda_utils import to_tpcd_gpu, concat_tpcd_gpu
from gen_utils.real_utils import transfer_pointcloud_colors_with_mask

class PointCloudProcessor:
    def __init__(self, args):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scene_name = args.scene_name # CHANGE
        self.rgb_path = f"data/{self.scene_name}/{self.scene_name}.jpg"
        self.bg_rgb_path = f"saves/{self.scene_name}/bg/bg.png"
        self.pcd_list = []
        self.num_path_frame = 15
        self.use_linear_interpolation = True
        
        self.read_img()

        self.extrinsic = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        
        self.ground_uv = args.ground_uv
        self.target_uv = args.target_uv
        self.robot_uv = args.robot_uv
        self.robot_rot = args.robot_rot
        
        self.scale_obj_00 = args.scale_obj_00

        self.intrinsic_path = "configs/cam.yml"

        self.ply_path = f"./saves/{self.scene_name}/depth/pcd/data/{self.scene_name}_fx_{self.fx}_fy_{self.fy}.ply"
        self.bg_ply_path = f"saves/{self.scene_name}/bg/depth/pcd/{self.scene_name}/bg_fx_{self.fx}_fy_{self.fy}.ply"
        # self.bg_ply_path = f"saves/{self.scene_name}/combined_pcd.ply"
        self.robo_ply_path = f"saves/{self.scene_name}/robot/pcd.ply"

        self.mask_path = f"saves/{self.scene_name}/keypoints/mask.png"
        self.mask_ground_path = f"saves/{self.scene_name}/keypoints/mask_ground.png"
        
        self.bg_depth_path = f"saves/{self.scene_name}/bg/depth/pcd/{self.scene_name}/bg_depth.npy"

        self.obj_path = f"saves/{self.scene_name}/reconstruction/object_00/mesh/color.obj"
        self.obj_usd_path = f"saves/{self.scene_name}/reconstruction/object_00/mesh/color.usd"

    def read_img(self):
        rgb = cv2.imread(self.rgb_path)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        self.rgb = torch.from_numpy(rgb)
        self.H, self.W, _ = rgb.shape
        
        
        self.fx = args.intrinsic[0] # CHANGE
        self.fy = args.intrinsic[1]
        
        self.cx = self.W // 2
        self.cy = self.H // 2
        self.intrinsic = np.array([
            [self.fx,  0, self.cx],
            [ 0, self.fy, self.cy],
            [ 0,  0,  1]
        ])

    def trans_pcd_np_to_o3d(self, pcd):
        if isinstance(pcd, np.ndarray):
            pcd_o3d = o3d.geometry.PointCloud()
            pcd_o3d.points = o3d.utility.Vector3dVector(pcd[:, :3])
            if pcd.shape[1] >= 6:
                pcd_o3d.colors = o3d.utility.Vector3dVector(pcd[:, 3:6] )
            return pcd_o3d 
        
    def get_pos_3d(self, uv):
        self.W_pcd = 1664
        return self.pcd_bg[uv[1] * self.W_pcd + uv[0]][:3]
        
    def process_point_cloud(self):
        
        fx = self.fx
        fy = self.fy
        cx = self.cx
        cy = self.cy
        intrinsic_matrix = np.array([
            [fx,  0, cx],
            [ 0, fy, cy],
            [ 0,  0,  1]
        ])
        image_size = (self.W, self.H)

        pcd_bg = o3d.io.read_point_cloud(self.bg_ply_path)
        points = np.asarray(pcd_bg.points) 
        colors = np.asarray(pcd_bg.colors) 
        self.pcd_bg = np.hstack([points, colors])

        self.pcd_ground, _ = get_objects_pcd_from_sam_mask(self.pcd_bg, self.mask_ground_path, self.intrinsic, (self.W, self.H), "object")
             
        pcd_ground_o3d = self.trans_pcd_np_to_o3d(self.pcd_ground)
        
        self.pcd_bg_o3d = self.trans_pcd_np_to_o3d(self.pcd_bg)


    def process_mesh_and_pcd(self):

        obj_ply_path = f"saves/{self.scene_name}/reconstruction/object_00/pcd/sample_pcd.ply"

        self.pcd_mesh_obj = o3d.io.read_point_cloud(obj_ply_path)
        
        
        
        cl, _ = self.pcd_mesh_obj.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        obb_pcd_mesh_obj = cl.get_oriented_bounding_box()
        obb_pcd_mesh_obj.color = (0, 0, 1)
        
        extent_mesh = obb_pcd_mesh_obj.extent
        
        # CHANGE
        self.pcd_mesh_obj.scale(self.scale_obj_00, center=obb_pcd_mesh_obj.center) # y axis (height) # clean
        obb_pcd_mesh_obj = self.pcd_mesh_obj.get_oriented_bounding_box(robust=True)
        obb_pcd_mesh_obj.color = (1, 0, 0)


        # save as o3d
        points = np.asarray(self.pcd_mesh_obj.points)  
        colors = np.asarray(self.pcd_mesh_obj.colors) 
        
        center = points.mean(axis=0)   # (3,)
        points_centered = points - center
        
        # # -------- 处理颜色 --------
        # # 定义接近白色的阈值（比如 >0.9 就算白）
        # white_mask = (colors > 0.5).all(axis=1)
        # from sklearn.neighbors import KDTree
        # # 构建 KDTree，只用非白色点作为参考
        # non_white_idx = np.where(~white_mask)[0]
        # white_idx = np.where(white_mask)[0]
        
        # print(white_idx.shape)

        # if len(non_white_idx) > 0 and len(white_idx) > 0:
        #     tree = KDTree(points[non_white_idx])
        #     # 查找每个白色点最近的非白点
        #     dists, nn_indices = tree.query(points[white_idx], k=1)
        #     # 替换颜色
        #     colors[white_idx] = colors[non_white_idx[nn_indices[:, 0]]]

        # 组合结果
        self.pcd_obj_recon = np.hstack([points_centered, colors])


        # ================== Visual Real ==================
        mask_obj = np.load(f"saves/{self.scene_name}/reconstruction/masks_npz/mask_00.npz")['arr_0'].astype(bool)
        rgb_np = self.rgb.cpu().numpy()

        _, _, self.pcd_obj_recon = transfer_pointcloud_colors_with_mask(
            points_centered, colors, rgb_np, mask_obj, keep_L=True
        )




        
        self.pose_path = f"saves/{self.scene_name}/reconstruction/object_00/6D/6D_pose.txt"
        with open(self.pose_path, 'r') as file:
            lines = file.readlines()
        matrix_data = []
        for line in lines:
            numbers = line.strip().split()
            row = [float(num) for num in numbers]
            matrix_data.append(row)
        transform_matrix = np.array(matrix_data)
        
        
        
        self.rotation_matrix = transform_matrix[:3, :3]
        self.translation_vector = transform_matrix[:3, 3]
                
           
        pitch_angle = radians(-90)  
        cos_pitch = cos(pitch_angle)
        sin_pitch = sin(pitch_angle)

        pitch_rotation = np.array([
            [1, 0, 0],
            [0, cos_pitch, -sin_pitch],
            [0, sin_pitch, cos_pitch]
        ])
        
        rotation_matrix = np.dot(self.rotation_matrix, pitch_rotation)
        
        
        self.pcd_obj_recon = apply_rotation(self.pcd_obj_recon, rotation_matrix)

        self.pcd_obj_recon = pcd_translate(self.pcd_obj_recon, self.translation_vector)
        

        
        
        
        pcd_mesh_obj_ground = self.get_pos_3d(self.ground_uv)
        
        self.move_vec = pcd_mesh_obj_ground - self.translation_vector
        

    def process_action(self, actions, index):
    
        # # --- 3. 归一化到 index 帧 ---
        # q0 = actions[index, 3:7].copy()
        # R0_inv = R.from_quat(q0).inv()
        # R_all = R.from_quat(actions[:, 3:7])
        # R_rel = R_all * R0_inv
        # Robo_Actions = actions.copy()
        # Robo_Actions[:, 3:7] = R_rel.as_quat()
        
        # action_copy = actions.copy()
        # actions[3] = -action_copy[5]
        # actions[5] = -action_copy[3]
        
        self.ee_robot_actions = actions.copy()
        self.ee_world_actions = actions.copy()

        def rot(point):
            center = self.robo_trans
            # Same Euler and sequence as your forward transform
            R_mat = R.from_euler('xyz', self.robo_rot, degrees=True).as_matrix()

            point = np.asarray(point)
            p0 = point @ R_mat + center
            return p0
        
        # actions_world t
        for i in range(len(actions)):
            self.ee_world_actions[i, :3] = rot(self.ee_robot_actions[i, :3])
        
        # # actions_world R
        # # R_FIX
        # self.R_fixed = R.from_euler('xyz', self.root_orientation, degrees=True)
        # R0_all = R.from_quat(self.ee_robot_actions[:, 3:7])   # scipy expects [x, y, z, w]
        # R1_all = self.R_fixed * R0_all
        # self.ee_world_actions[:, 3:7] = R1_all.as_quat()      # stays in [x, y, z, w]
        

        # 假设 self.root_orientation = [x, y, z]（单位：度）
        x, y, z = self.root_orientation
        x_rad, y_rad, z_rad = np.radians(x), np.radians(y), np.radians(z)

        # 计算绕 X、Y、Z 轴旋转的四元数（顺序为 [w, x, y, z]）
        qx = [np.cos(x_rad/2), np.sin(x_rad/2), 0, 0]
        qy = [np.cos(y_rad/2), 0, np.sin(y_rad/2), 0]
        qz = [np.cos(z_rad/2), 0, 0, np.sin(z_rad/2)]

        # 四元数乘法（顺序：Z -> Y -> X）
        q_total = [
            qz[0]*qy[0] - qz[1]*qy[1] - qz[2]*qy[2] - qz[3]*qy[3],
            qz[0]*qy[1] + qz[1]*qy[0] + qz[2]*qy[3] - qz[3]*qy[2],
            qz[0]*qy[2] - qz[1]*qy[3] + qz[2]*qy[0] + qz[3]*qy[1],
            qz[0]*qy[3] + qz[1]*qy[2] - qz[2]*qy[1] + qz[3]*qy[0]
        ]
        q_total = [
            q_total[0]*qx[0] - q_total[1]*qx[1] - q_total[2]*qx[2] - q_total[3]*qx[3],
            q_total[0]*qx[1] + q_total[1]*qx[0] + q_total[2]*qx[3] - q_total[3]*qx[2],
            q_total[0]*qx[2] - q_total[1]*qx[3] + q_total[2]*qx[0] + q_total[3]*qx[1],
            q_total[0]*qx[3] + q_total[1]*qx[2] - q_total[2]*qx[1] + q_total[3]*qx[0]
        ]

        # 遍历并更新 self.ee_robot_actions 中的四元数
        for i in range(self.ee_robot_actions.shape[0]):
            q_original = self.ee_robot_actions[i, 3:7]
            w, x, y, z = q_original
            # 四元数乘法（q_total * q_original）
            self.ee_world_actions[i, 3:7] = [
                q_total[0]*w - q_total[1]*x - q_total[2]*y - q_total[3]*z,
                q_total[0]*x + q_total[1]*w + q_total[2]*z - q_total[3]*y,
                q_total[0]*y - q_total[1]*z + q_total[2]*w + q_total[3]*x,
                q_total[0]*z + q_total[1]*y - q_total[2]*x + q_total[3]*w
            ]
        
        q_anchor = [0, 1, 0, 0]   # [w,x,y,z]
        w, x, y, z = q_anchor
        # 四元数乘法 q_total * q_anchor
        self.q_anchor_world = [
            q_total[0]*w - q_total[1]*x - q_total[2]*y - q_total[3]*z,
            q_total[0]*x + q_total[1]*w + q_total[2]*z - q_total[3]*y,
            q_total[0]*y - q_total[1]*z + q_total[2]*w + q_total[3]*x,
            q_total[0]*z + q_total[1]*y - q_total[2]*x + q_total[3]*w
        ]  
        
        
        
        # world delta
        self.ee_world_delta_actions = self.ee_world_actions.copy()
        q0 = self.ee_world_actions[index, 3:7]
        R0_inv = R.from_quat(q0).inv()
        q_m = np.array([-0.5, 0.5, -0.5, 0.5]) #MODIFY
        # q_m = np.array(self.q_anchor_world)
        Rm = R.from_quat(q_m)
        Rm_inv = R.from_quat(q_m).inv()
        R_all = R.from_quat(self.ee_world_actions[:, 3:7])
        R_rel = R_all * Rm_inv
        self.ee_world_delta_actions[:, 3:7] = R_rel.as_quat()
        self.R0_inv = R0_inv
        
        # robot delta
        self.ee_robot_delta_actions = self.ee_robot_actions.copy()
        q0 = self.ee_robot_actions[index, 3:7]
        R0_inv = R.from_quat(q0).inv()
        R_all = R.from_quat(self.ee_robot_actions[:, 3:7])
        R_rel = R_all * R0_inv
        self.ee_robot_delta_actions[:, 3:7] = R_rel.as_quat()
        
        
        
        
        return self.ee_world_delta_actions
    



         
    def process_ply(self,depth_image,rgb_image):
        depth_min = 0.1
        depth_max = 5.0
        valid_mask = (depth_image > depth_min) & (depth_image < depth_max)

        depth_valid = depth_image[valid_mask]
        rgb_valid = rgb_image[valid_mask]

        fx, fy = self.intrinsic[0, 0], self.intrinsic[1, 1]
        cx, cy = self.intrinsic[0, 2], self.intrinsic[1, 2]

        u, v = np.meshgrid(np.arange(depth_image.shape[1]), np.arange(depth_image.shape[0]))
        u_valid = u[valid_mask].flatten()
        v_valid = v[valid_mask].flatten()

        x = (u_valid - cx) * depth_valid / fx
        y = (v_valid - cy) * depth_valid / fy
        z = depth_valid
        points_cam = np.column_stack([x, y, z])
        return points_cam,rgb_valid


    def init_robo(self):
        self.root_position = self.get_pos_3d(self.robot_uv).tolist()
        
        self.root_orientation = self.robot_rot
        
        self.robo_trans = self.root_position
        self.robo_rot = [-x for x in self.root_orientation]
        
        self.robot = FrankaSimRobot(root_position = self.root_position, root_orientation = self.root_orientation, H = self.H, W = self.W, fxy = self.fx)
        
        
        num_episode = 10
        self.robot.condition["task_name"] = self.scene_name + '_' + time.strftime("%Y%m%d_%H%M%S")
        
        self.result_name = self.robot.condition["task_name"]
        
        self.result_path = os.path.join('./results', self.scene_name, f'{self.result_name}')
        os.makedirs(self.result_path, exist_ok=True)
        
        self.robot.env.my_franka.gripper.set_linear_velocity(0.005)
        
        # convert_obj_to_usd(self.obj_path, self.obj_usd_path)


    def gen_frames(self, motion, index):
        
        
        self.motion_process(motion) # TODO
        

        robo_pcds = []
        
        device = o3d.core.Device("CUDA:0")
        device_o3d = o3d.core.Device("CUDA:0")
        device_torch = "cuda" if "CUDA" in str(device_o3d) else "cpu"

        for i in tqdm(range(len(self.depth_robo)), desc="Processing Robot PLY data"):
            # 仍然调用你现有的处理函数；返回 numpy
            points_cam_np, rgb_valid_np = self.process_ply(self.depth_robo[i], self.rgb_robo[i])  # (N,3), (N,3) in BGR

            # —— GPU 上的计算（通道重排 + 归一化）——
            points_cam_t = torch.as_tensor(points_cam_np, dtype=torch.float32, device=device_torch).contiguous()  # (N,3)
            rgb_valid_t  = torch.as_tensor(rgb_valid_np,  dtype=torch.float32, device=device_torch).contiguous()   # (N,3) BGR
            rgb_valid_t  = rgb_valid_t[:, [2, 1, 0]] / 255.0  # BGR->RGB，并归一化到 [0,1]

            # —— 用 DLPack 零拷贝把 torch Tensor 交给 Open3D（保持在 GPU）——
            pos_o3d = o3d.core.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(points_cam_t))
            col_o3d = o3d.core.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(rgb_valid_t))

            pcd_t = o3d.t.geometry.PointCloud(device=device_o3d)
            pcd_t.point["positions"] = pos_o3d  # (N,3) float32, GPU
            pcd_t.point["colors"]    = col_o3d  # (N,3) float32, GPU [0,1]

            robo_pcds.append(pcd_t)
            
        self.robo_pcds = robo_pcds
        
        
        self.pcd_obj_recon_random = self.pcd_obj_recon.copy()

        ############## start of generating one episode ##############
        current_frame = 0
        traj_states = []
        traj_actions = []
        traj_pcds = []
        ############# stage {motion-1} starts #############
        
        ## ( + 右边, + 下降, +后退)
        ## (y, -z, -x)
        
        
        
        mask = np.all(self.grippers[1:] < 0.039, axis=1) 
        grasp_index = np.argmax(mask) + 1 if np.any(mask) else None
        
        self.ee_world_delta_actions = self.process_action(self.qpos, grasp_index)
        
                            
        
        
        start_pc = motion
        
        
        start_pc = start_pc - self.translation_vector
        
        self.pcd_obj_recon_random = pcd_translate(self.pcd_obj_recon_random, start_pc)
        
        # (y, -z, -x)
        
        self.ee_world_delta_quats = self.ee_world_delta_actions[:, 3:7].copy()

        self.num_path_frame = len(self.ee_world_delta_actions)

        self.pcd_obj_recon_o3d = self.trans_pcd_np_to_o3d(self.pcd_obj_recon_random)

        pcd_obj = self.pcd_obj_recon_o3d
        
        
        assert o3d.core.cuda.is_available(), "Open3D must be built with CUDA"
        device = o3d.core.Device("CUDA:0")
        DTypeF = o3d.core.Dtype.Float32


        # ----- convert your inputs to GPU once -----
        pcd_obj_t = to_tpcd_gpu(pcd_obj)  # your initial object pcd
        robo_t_list = robo_pcds  # pre-convert all robot frames
        pcd_bg_t = o3d.t.geometry.PointCloud.from_legacy(self.pcd_bg_o3d, device=device)
        
        self.pcd_bg_t = pcd_bg_t

        traj_pcds = [] 

        
        # H_map = np.array([[0, 1, 0],
        #                 [-1,  0, 0],
        #                 [0,  0,-1]], dtype=np.float64)
        
        H_map = np.array([[0, 1, 0],
                        [0,  0, 1],
                        [1,  0, 0]], dtype=np.float64)
        
        g = grasp_index

        p_grasp = self.ee_world_actions[g, :3].astype(np.float32)
        
        self.ee_world_delta_actions[:, :3] = self.ee_world_delta_actions[:, :3]
        
        R_robot_grasp = R.from_quat(np.asarray(self.ee_robot_actions[:, 3:7][g]))  # scipy: [x, y, z, w]
        R_world_grasp = R.from_quat(np.asarray(self.ee_world_actions[:, 3:7][g]))  # scipy: [x, y, z, w]
        
        
        Matrix_robot_grasp = R_robot_grasp.as_matrix()
        Matrix_world_grasp = R_world_grasp.as_matrix()

        pcd_obj_grasp = pcd_obj_t.clone() 


        def create_highlight_sphere(self, center, radius=0.05, color=[1.0, 0.0, 0.0]):
            mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
            mesh_sphere.translate(center)
            mesh_sphere.paint_uniform_color(color)
            return mesh_sphere.sample_points_poisson_disk(number_of_points=100)

        highlight_sphere = to_tpcd_gpu(create_highlight_sphere(self, motion))


        def rotate_point(point, center, euler_angles, seq='xyz', degrees=True):
            point = np.asarray(point)
            center = np.asarray(center)
            p_centered = point - center
            rot = R.from_euler(seq, euler_angles, degrees=degrees)
            R_mat = rot.as_matrix()
            p_rot = p_centered @ R_mat.T
            p_final = p_rot + center
            return p_final
            
        def rot(point):
            r_point = rotate_point(point,
                            self.robo_trans,
                            self.robo_rot)
            t_point = r_point - self.robo_trans
            t_point[2]+=0.1 
            
            return t_point

        def offset_tran(motion,dz):
            motion[2]+=dz
            return motion   
        
        pcd_legacy = pcd_obj_grasp.to_legacy()
        points_np = np.asarray(pcd_legacy.points)
        obj_center = points_np.mean(axis=0)

        for j in tqdm(range(self.num_path_frame), desc="Generating frames", unit="frame"):
            if j < g:
                R_step  = np.eye(3, dtype=np.float32)
                t_total = np.zeros(3, dtype=np.float32)
                pcd_frame = pcd_obj_grasp.clone()
            else:
                pcd_frame = pcd_obj_grasp.clone()
                
                q_ee_step = self.ee_world_delta_quats[j]
                
                Matrix_delta = R.from_quat(self.ee_world_delta_quats[j]).as_matrix()
                                
                Matrix_grasp = R_world_grasp.as_matrix()
                
                # Matrix_anchor = np.array([
                #     [0, 0, 1],
                #     [-1, 0, 0],
                #     [0, -1, 0]
                # ])
                
                Matrix_anchor = R.from_quat(self.q_anchor_world).as_matrix()
                
                q0 = self.ee_world_actions[g, 3:7]
                R0 = R.from_quat(q0)
                R0_inv = R.from_quat(q0).inv()
                q_m = np.array([-0.5, 0.5, -0.5, 0.5]) #MODIFY
                Rm = R.from_quat(q_m)
                Rm_inv = R.from_quat(q_m).inv()
                
                Matrix_m_step = ( R0 * Rm_inv ).as_matrix()
                
                Matrix_obj_step = Matrix_anchor @ Matrix_delta @ Matrix_anchor.T 
                
                q_grasp = R.from_matrix(Matrix_grasp).as_quat()

                q_obj_step = R.from_matrix(Matrix_obj_step).as_quat()
                # MODIFY
                q_obj_step_clone = q_obj_step.copy()
                q_obj_step[0] = -q_obj_step_clone[2]
                q_obj_step[2] = -q_obj_step_clone[0]
                
                
                
                Matrix_obj_step = R.from_quat(q_obj_step).as_matrix()
                
                q_modify = R.from_matrix((Matrix_anchor @ Matrix_m_step @ Matrix_anchor.T).T).as_quat()
                q_modify_clone = q_modify.copy()
                q_modify[0] = -q_modify_clone[2]
                q_modify[2] = -q_modify_clone[0]
                Matrix_modify = R.from_quat(q_modify).as_matrix()
                
                Matrix_obj_step = Matrix_obj_step @ Matrix_modify
                

                t_j = self.ee_world_delta_actions[j, :3].astype(np.float32)   
                t_total = t_j - (Matrix_obj_step @ (p_grasp.astype(np.float32)))
                

                
                # t_total = t_j + ((R_obj_step - np.eye(3)) @ (obj_center.astype(np.float32) - p_grasp.astype(np.float32)))
                
                T = np.eye(4, dtype=np.float32)
                T[:3, :3] = Matrix_obj_step
                T[:3, 3]  = t_total
                pcd_frame.transform(o3d.core.Tensor(T, dtype=DTypeF, device=device))

            combined_t = concat_tpcd_gpu(
                concat_tpcd_gpu(pcd_frame, robo_t_list[j]),
                pcd_bg_t
            )
            
            traj_pcds.append(combined_t)
            current_frame += 1


        generated_episode = {
            "state": traj_states,
            "action": traj_actions,
            "point_cloud": traj_pcds
        }

        point_clouds_video = generated_episode["point_cloud"]
        
        
        self.random_results_dir = self.result_path + '/random'
        os.makedirs(self.random_results_dir, exist_ok=True)
        render_path = os.path.join(self.random_results_dir, f"video_{index}.mp4")

        render_point_cloud_sequence(
            pcd_list=point_clouds_video,  # each item is (obj, robo) and will be merged
            intrinsic=self.intrinsic,
            extrinsic=self.extrinsic,
            img_width=self.W,
            img_height=self.H,
            output_video_path=render_path,
            fps=30
        )
        
        return traj_pcds
    
    def get_grasp(self):
        
        self.pose_path = f"saves/{self.scene_name}/reconstruction/object_00/grasp/best_pose.txt"
        with open(self.pose_path, 'r') as file:
            lines = file.readlines()
        matrix_data = []
        for line in lines:
            numbers = line.strip().split()
            row = [float(num) for num in numbers]
            matrix_data.append(row)
        grasp_matrix = np.array(matrix_data)
        rotation_matrix = grasp_matrix[:3, :3]
        
        rot = R.from_matrix(rotation_matrix)
        grasp_quaternion = rot.as_quat()
        grasp_diff = grasp_matrix[:3, 3]
        
        return grasp_diff, grasp_quaternion

    def motion_process(self, motion):
                
        def rotate_point(point, center, euler_angles, seq='xyz', degrees=True):
            point = np.asarray(point)
            center = np.asarray(center)
            p_centered = point - center
            rot = R.from_euler(seq, euler_angles, degrees=degrees)
            R_mat = rot.as_matrix()
            p_rot = p_centered @ R_mat.T
            p_final = p_rot + center
            return p_final
        
        def rot(point):
            r_point = rotate_point(point,
                            self.robo_trans,
                            self.robo_rot)
            t_point = r_point - self.robo_trans
            t_point[2]+=0.1 
            
            return t_point

        def offset_tran(motion,d):
            motion[0]+=d[0]
            motion[1]+=d[1]
            motion[2]+=d[2]
            return motion
        
        
        ### Motion Planning ###
        
        self.robot.reset()
        
        grasp_width = 0.02
        
        orientation = np.array([0, -1, 0, 0])
        r = R.from_quat(orientation)
        delta_rot = R.from_euler('x', 60, degrees=True)
        r_new = delta_rot * r  
        orientation = r_new.as_quat()

        
        target_pos = self.get_pos_3d(self.target_uv) - self.move_vec

        
        self.robot.add_grasp_obj(motion, self.rotation_matrix, self.scale_obj_00, self.obj_usd_path)
        self.robot.open()
        
        self.robot.collect_motion_planning_motion(offset_tran(rot(motion),[0.05, 0.09, 0.245]), orientation, grasp_width)
        
        self.robot.collect_motion_planning_motion(offset_tran(rot(motion),[0.07, 0.09, 0.08]), orientation, grasp_width)
        
        self.robot.close_gripper()
        
        self.robot.collect_motion_planning_motion(offset_tran(rot(motion),[0, 0.1, rot(target_pos)[2] - rot(motion)[2] + 0.1]), orientation, grasp_width)
        
        orientation = np.array([0, -1, 0, 0])
        r = R.from_quat(orientation)
        delta_rot = R.from_euler('x', 60, degrees=True)
        r_new = delta_rot * r  
        orientation = r_new.as_quat()
        r = R.from_quat(orientation)
        delta_rot = R.from_euler('y', -45, degrees=True)
        r_new = delta_rot * r  
        orientation = r_new.as_quat()
        
        self.robot.collect_motion_planning_motion(offset_tran(rot(target_pos),[0.15, 0.25, 0.12]), orientation, grasp_width )
        

        # self.robot.open()
        
        self.rgb_robo, self.depth_robo, self.qpos, self.grippers, self.joints = self.robot.finish_robo(self.result_path)


    def process_frames(self):
        
        
        motion_list = []
        
        for i in range(10):
            
            random_motion = random.choice(self.pcd_ground)[:3] - self.move_vec
            
            random_motion = self.get_pos_3d(self.ground_uv) - self.move_vec# CHANGE
            
                        
            motion_list.append(random_motion)
            

        # np.save(f"{self.result_path}/key_poses_0.npy", np.array(motion_list))

        gen_pcds = []
        
        for i, motion in enumerate(motion_list):
            time_start = time.time()
            gen_pcd = self.gen_frames(motion, i)
            gen_pcds.append(gen_pcd)
            
            elasped_time = time.time() - time_start
            print(f"Genrating Data_{i} Time: ", elasped_time)
        

def parse_args():
    parser = argparse.ArgumentParser(description="Build scene with mask, keypoints, and background inpainting.")
    parser.add_argument('--scene_name', type=str, help='Scene name, e.g., "table"')
    parser.add_argument('--prompts', type=str, nargs='+', help='Object prompts')
    parser.add_argument('--intrinsic', type=float, nargs=2, default=[1000, 1000], help='Camera intrinsic parameters [fx, fy]')
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()
    
    args.scene_name = 'flower'
    args.prompts = ['flower', 'flower']
    args.intrinsic = [1000, 1000]
    
    args.robot_rot = [120,0, -90] # 旋转（设定）

    args.robot_uv = [900, 757] # 机械臂位置 (W, H)
    args.ground_uv = [964, 1241] # 被操作物体的位置
    
    # args.target_uv = [1490, 203] # 放柜子
    args.target_uv = [260, 600] # 浇花
    
    
    args.scale_obj_00 = 0.4 #物体大小
    
    processor = PointCloudProcessor(args = args)

    processor.read_img()

    processor.process_point_cloud()

    processor.process_mesh_and_pcd()
    
    processor.init_robo() # TODO
    
    processor.process_frames()

    
