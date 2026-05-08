import json
import os
import os.path
import numpy as np
import torch
from tqdm import tqdm
import smplx
import trimesh
from scipy.spatial.transform import Rotation
from copy import copy
from human_body_prior.body_model.body_model import BodyModel
import sys
import pickle
import random
import argparse
sys.path.append('.')
sys.path.append('..')
from render.mesh_viz import visualize_body_obj

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

to_cpu = lambda tensor: tensor.detach().cpu().numpy()

# dataset = sys.argv[1].upper()

results_folder = "optimization_vis/demo_vis"
OBJMOTION_PATH = 'omomo/sequences_canonical'
OBJECT_PATH = 'omomo/objects'
MODEL_PATH = 'models'

######################################## smplx 16 ########################################
dmpl_fname = None
num_dmpls = None
num_expressions = None
num_betas = 16
SMPLX_PATH = MODEL_PATH + '/smplx'
# SMPLX_PATH = "/home/dixuan/sony/hoifhli_release/data/smpl_all_models/smplx"
surface_model_male_fname = os.path.join(SMPLX_PATH, "SMPLX_MALE.npz")
surface_model_female_fname = os.path.join(SMPLX_PATH, "SMPLX_FEMALE.npz")
surface_model_neutral_fname = os.path.join(SMPLX_PATH, "SMPLX_NEUTRAL.npz")

smplx16_model_male = BodyModel(bm_fname=surface_model_male_fname,
                               num_betas=num_betas,
                               num_expressions=num_expressions,
                               num_dmpls=num_dmpls,
                               dmpl_fname=dmpl_fname)
smplx16_model_female = BodyModel(bm_fname=surface_model_female_fname,
                                 num_betas=num_betas,
                                 num_expressions=num_expressions,
                                 num_dmpls=num_dmpls,
                                 dmpl_fname=dmpl_fname)
smplx16_model_neutral = BodyModel(bm_fname=surface_model_neutral_fname,
                                  num_betas=num_betas,
                                  num_expressions=num_expressions,
                                  num_dmpls=num_dmpls,
                                  dmpl_fname=dmpl_fname)
smplx16 = {'male': smplx16_model_male.to(device), 'female': smplx16_model_female.to(device), 'neutral': smplx16_model_neutral.to(device)}
########################################################################################

os.makedirs(results_folder, exist_ok=True)

mesh_folder = os.path.join(results_folder, "mesh")
video_folder = os.path.join(results_folder, "video")
loss_folder = os.path.join(results_folder, "loss")
os.makedirs(mesh_folder, exist_ok=True)
os.makedirs(video_folder, exist_ok=True)



######################################## Visualize SMPL ########################################
def visualize_smpl(name, MOTION_PATH, model_type, num_betas, num_pca_comps=None):
    """
    BEHAVE for SMPLH 10
    NEURALDOME or IMHD for SMPLH 16
    vertices: (N, 6890, 3)
    Chairs for SMPLX 10
    InterCap for SMPLX 12
    OMOMO for SMPLX 16
    vertices: (N, 10475, 3)
    """
    # print(f"Loading from {os.path.join(MOTION_PATH, name, 'human.npz')}")
    with np.load(os.path.join(MOTION_PATH, name, 'human.npz'), allow_pickle=True) as f:
        poses, betas, trans, gender = f['poses'], f['betas'], f['trans'], str(f['gender'])

    frame_times = poses.shape[0]

    smpl_model = smplx16[gender]
    smplx_output = smpl_model(pose_body=torch.from_numpy(poses[:, 3:66]).float().to(device),
                              pose_hand=torch.from_numpy(poses[:, 66:156]).float().to(device),
                              betas=torch.from_numpy(betas[None, :]).repeat(frame_times, 1).float().to(device),
                              root_orient=torch.from_numpy(poses[:, :3]).float().to(device),
                              trans=torch.from_numpy(trans).float().to(device))

    verts = to_cpu(smplx_output.v)
    faces = to_cpu(smpl_model.f)

    return verts, faces

def vis_seq(seq):
    verts, faces = visualize_smpl(seq, os.path.join(update_path), 'smplx', 16)
    with np.load(os.path.join(OBJMOTION_PATH, seq, 'object.npz'), allow_pickle=True) as f:
        obj_angles, obj_trans, obj_name = f['angles'], f['trans'], str(f['name'])

    mesh_obj = trimesh.load(os.path.join(OBJECT_PATH, f"{obj_name}/{obj_name}.obj"), force='mesh')
    obj_verts, obj_faces = mesh_obj.vertices, mesh_obj.faces

    angle_matrix = Rotation.from_rotvec(obj_angles).as_matrix()
    obj_verts = (obj_verts)[None, ...]
    obj_verts = np.matmul(obj_verts, np.transpose(angle_matrix, (0, 2, 1))) + obj_trans[:, None, :]

    mesh_path = os.path.join(mesh_folder, f"{seq}_upd")
    os.makedirs(mesh_path, exist_ok=True)
    # for idx in range(verts.shape[0]):
    #     if idx % 10 == 0:
    #         obj_mesh = trimesh.Trimesh(vertices=obj_verts[idx], faces=obj_faces)
    #         human_mesh = trimesh.Trimesh(vertices=verts[idx], faces=faces)
    #         obj_mesh.export(os.path.join(mesh_path, f"obj_{idx:03d}.ply"))
    #         human_mesh.export(os.path.join(mesh_path, f"human_{idx:03d}.ply"))

    rend_video_path = os.path.join(video_folder, '{}.mp4'.format(seq))
    visualize_body_obj(verts, faces, obj_verts, obj_faces, save_path=rend_video_path, show_frame=True,
                       multi_angle=True, h=512, w=512)


update_path = "updated_results"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq_name', type=str, default=None, help='Sequence name to visualize. Visualizes all sequences when omitted.')
    return parser.parse_args()


def main():
    args = parse_args()
    if args.seq_name is not None:
        seq_path = os.path.join(update_path, args.seq_name)
        if not os.path.isdir(seq_path):
            raise FileNotFoundError(f"Sequence not found in {update_path}: {args.seq_name}")
        seq_names = [args.seq_name]
    else:
        seq_names = sorted(os.listdir(update_path))

    print(len(seq_names))
    for seq in tqdm(seq_names):
        vis_seq(seq)


if __name__ == '__main__':
    main()
