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

sys.path.append('.')
sys.path.append('..')
from render.mesh_viz import visualize_body_obj
from eval.metrics.metrics import ObjectContactMetrics
from eval.metrics import contactutils

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
to_cpu = lambda tensor: tensor.detach().cpu().numpy()

# dataset = sys.argv[1].upper()

MOTION_PATH = 'omomo_hand_correct/sequences_canonical'
OBJMOTION_PATH = 'omomo/sequences_canonical'
OBJECT_PATH = 'omomo/objects'
MODEL_PATH = 'models'

data_name = os.listdir(MOTION_PATH)
######################################## smplh 10 ########################################
smplh_model_male = smplx.create(MODEL_PATH, model_type='smplh',
                                gender="male",
                                use_pca=False,
                                ext='pkl')

smplh_model_female = smplx.create(MODEL_PATH, model_type='smplh',
                                  gender="female",
                                  use_pca=False,
                                  ext='pkl')

smplh10 = {'male': smplh_model_male, 'female': smplh_model_female}
######################################## smplx 10 ########################################
smplx_model_male = smplx.create(MODEL_PATH, model_type='smplx',
                                gender='male',
                                use_pca=False,
                                ext='pkl')

smplx_model_female = smplx.create(MODEL_PATH, model_type='smplx',
                                  gender="female",
                                  use_pca=False,
                                  ext='pkl')

smplx_model_neutral = smplx.create(MODEL_PATH, model_type='smplx',
                                   gender="neutral",
                                   use_pca=False,
                                   ext='pkl')

smplx10 = {'male': smplx_model_male, 'female': smplx_model_female, 'neutral': smplx_model_neutral}
######################################## smplx 12 ########################################
smplx12_model_male = smplx.create(MODEL_PATH, model_type='smplx',
                                  gender="male",
                                  num_pca_comps=12,
                                  ext='pkl')

smplx12_model_female = smplx.create(MODEL_PATH, model_type='smplx',
                                    gender="female",
                                    num_pca_comps=12,
                                    ext='pkl')

smplx12_model_neutral = smplx.create(MODEL_PATH, model_type='smplx',
                                     gender="neutral",
                                     num_pca_comps=12,
                                     ext='pkl')

smplx12 = {'male': smplx12_model_male, 'female': smplx12_model_female, 'neutral': smplx12_model_neutral}
######################################## smplh 16 ########################################
SMPLH_PATH = MODEL_PATH + '/smplh'
surface_model_male_fname = os.path.join(SMPLH_PATH, 'male', "model.npz")
surface_model_female_fname = os.path.join(SMPLH_PATH, "female", "model.npz")
surface_model_neutral_fname = os.path.join(SMPLH_PATH, "neutral", "model.npz")
dmpl_fname = None
num_dmpls = None
num_expressions = None
num_betas = 16

smplh16_model_male = BodyModel(bm_fname=surface_model_male_fname,
                               num_betas=num_betas,
                               num_expressions=num_expressions,
                               num_dmpls=num_dmpls,
                               dmpl_fname=dmpl_fname)
smplh16_model_female = BodyModel(bm_fname=surface_model_female_fname,
                                 num_betas=num_betas,
                                 num_expressions=num_expressions,
                                 num_dmpls=num_dmpls,
                                 dmpl_fname=dmpl_fname)
smplh16_model_neutral = BodyModel(bm_fname=surface_model_neutral_fname,
                                  num_betas=num_betas,
                                  num_expressions=num_expressions,
                                  num_dmpls=num_dmpls,
                                  dmpl_fname=dmpl_fname)
smplh16 = {'male': smplh16_model_male, 'female': smplh16_model_female, 'neutral': smplh16_model_neutral}
######################################## smplx 16 ########################################
SMPLX_PATH = MODEL_PATH + '/smplx'
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
smplx16 = {'male': smplx16_model_male, 'female': smplx16_model_female, 'neutral': smplx16_model_neutral}
########################################################################################
results_folder = "/home/dixuan/sony/InterAct/omomo_hand_correct/results"
os.makedirs(results_folder, exist_ok=True)

mesh_folder = os.path.join(results_folder, "mesh")
video_folder = os.path.join(results_folder, "video")
os.makedirs(mesh_folder, exist_ok=True)
os.makedirs(video_folder, exist_ok=True)


def evaluation(human_verts, human_face, obj_verts, obj_face, contact_right, contact_left):
    device = 'cuda'
    pene_dists = []
    contact_v_num = []
    right_hand_verts = human_verts[:, rhand_idx]
    left_hand_verts = human_verts[:, lhand_idx]

    frame_num = human_verts.shape[0]

    obj_triangles = obj_verts[:, obj_faces]

    for fid in tqdm(range(frame_num)):
        if fid % 4 != 0:
            continue
        if contact_right[fid] == 1:
            obj_mesh = trimesh.Trimesh(vertices=obj_verts[fid], faces=obj_face)
            trimesh.repair.fix_normals(obj_mesh)
            _, _dist_to_closets_point_on_obj, _, = trimesh.proximity.closest_point(obj_mesh, right_hand_verts[fid])

            exterior = contactutils.batch_mesh_contains_points(torch.tensor(right_hand_verts[None, fid, :]).float().to(device),
                                                               torch.tensor(obj_triangles[None, fid, :, :]).float().to(device),
                                                               torch.Tensor(
                                                                   [0.4395064455, 0.617598629942, 0.652231566745]).to(
                                                                   device))
            penetr_mask = (~exterior.squeeze(dim=0)).detach().cpu().numpy()

            contact_v_num.append(np.sum(_dist_to_closets_point_on_obj < 5 * 1e-3))
            dist_pene = _dist_to_closets_point_on_obj[penetr_mask]
            pene_dists.append(dist_pene)

        if contact_left[fid] == 1:
            obj_mesh = trimesh.Trimesh(vertices=obj_verts[fid], faces=obj_face)
            trimesh.repair.fix_normals(obj_mesh)
            _, _dist_to_closets_point_on_obj, _, = trimesh.proximity.closest_point(obj_mesh, left_hand_verts[fid])

            exterior = contactutils.batch_mesh_contains_points(torch.tensor(left_hand_verts[None, fid, :]).float().to(device),
                                                               torch.tensor(obj_triangles[None, fid, :, :]).float().to(device),
                                                               torch.Tensor(
                                                                   [0.4395064455, 0.617598629942, 0.652231566745]).to(
                                                                   device))
            penetr_mask = (~exterior.squeeze(dim=0)).detach().cpu().numpy()

            contact_v_num.append(np.sum(_dist_to_closets_point_on_obj < 5 * 1e-3))
            dist_pene = _dist_to_closets_point_on_obj[penetr_mask]
            pene_dists.append(dist_pene)

    pene_dists = np.concatenate(pene_dists)
    pene_mean = np.mean(pene_dists)
    pene_max = np.max(pene_dists)
    contact_v_num = np.array(contact_v_num)
    contact_mean = np.mean(contact_v_num)
    contact_succ = np.sum(contact_v_num >= 50) / (np.sum(contact_v_num >= 0) + 0.0)
    return pene_mean, pene_max, contact_mean, contact_succ


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
    with np.load(os.path.join(MOTION_PATH, name, 'human.npz'), allow_pickle=True) as f:
        poses, betas, trans, gender = f['poses'], f['betas'], f['trans'], str(f['gender'])

    frame_times = poses.shape[0]

    smpl_model = smplx16[gender]
    smplx_output = smpl_model(pose_body=torch.from_numpy(poses[:, 3:66]).float(),
                              pose_hand=torch.from_numpy(poses[:, 66:156]).float(),
                              betas=torch.from_numpy(betas[None, :]).repeat(frame_times, 1).float(),
                              root_orient=torch.from_numpy(poses[:, :3]).float(),
                              trans=torch.from_numpy(trans).float())
    verts = to_cpu(smplx_output.v)
    faces = smpl_model.f

    return verts, faces


######################################## Visualize GRAB ########################################
def visualize_grab(name, MOTION_PATH):
    """
    vertices: (N, 10475, 3)
    """
    with np.load(os.path.join(MOTION_PATH, name, 'human.npz'), allow_pickle=True) as f:
        poses, vtemp, trans, gender = f['poses'], f['vtemp'], f['trans'], str(f['gender'])
    n_comps = 24
    T = len(poses)

    smpl_model = smplx.create(
        model_path=MODEL_PATH,
        model_type='smplx',
        gender=gender,
        num_pca_comps=n_comps,
        v_template=vtemp,
        batch_size=T)

    smplx_output = smpl_model(body_pose=torch.from_numpy(poses[:, 3:66]).float(),
                              global_orient=torch.from_numpy(poses[:, :3]).float(),
                              left_hand_pose=torch.from_numpy(poses[:, 66:90]).float(),
                              right_hand_pose=torch.from_numpy(poses[:, 90:114]).float(),
                              transl=torch.from_numpy(trans).float(), )
    verts = to_cpu(smplx_output.vertices)
    faces = smpl_model.faces

    return verts, faces

with open("touch.json", "r", encoding="utf-8") as f:
    touch_data = json.load(f)

rhand_idx = np.load('./assets/smplx_hand_index/rhand_smplx_ids.npy')
lhand_idx_path = './assets/smplx_hand_index/lhand_smplx_ids.npy'
lhand_idx = np.load(lhand_idx_path)

obj_names = os.listdir("/home/dixuan/sony/InterAct/omomo/objects")

quan_results = {}
for objn in obj_names:
    quan_results[objn] = {"pene_mean":[], "pene_max":[], "c_mean":[], "c_succ":[]}
# visualize surface motion of smpl model
for k, name in tqdm(enumerate(data_name)):
    if not os.path.exists(os.path.join(MOTION_PATH, name, 'human.npz')):
         continue
    # valid = False
    # for objn in obj_names:
    #     if objn in name:
    #         valid = True
    # if not valid:
    #     continue
    touch_right = np.array(touch_data[name]['right']).reshape(-1)
    touch_left = np.array(touch_data[name]['left']).reshape(-1)

    verts, faces = visualize_smpl(name, MOTION_PATH, 'smplx', 16)

    with np.load(os.path.join(OBJMOTION_PATH, name, 'object.npz'), allow_pickle=True) as f:
        obj_angles, obj_trans, obj_name = f['angles'], f['trans'], str(f['name'])

    mesh_obj = trimesh.load(os.path.join(OBJECT_PATH, f"{obj_name}/{obj_name}.obj"), force='mesh')
    obj_verts, obj_faces = mesh_obj.vertices, mesh_obj.faces

    angle_matrix = Rotation.from_rotvec(obj_angles).as_matrix()
    obj_verts = (obj_verts)[None, ...]
    obj_verts = np.matmul(obj_verts, np.transpose(angle_matrix, (0, 2, 1))) + obj_trans[:, None, :]

    pene_mean, pene_max, contact_mean, contact_succ = evaluation(verts, faces.detach().cpu().numpy(),
                                                                 np.asarray(obj_verts), np.asarray(obj_faces),
                                                                 touch_right, touch_left)

    quan_results[obj_name]["pene_mean"].append(pene_mean)
    quan_results[obj_name]["pene_max"].append(pene_max)
    quan_results[obj_name]["c_mean"].append(contact_mean)
    quan_results[obj_name]["c_succ"].append(contact_succ)
    # mesh_path = os.path.join(mesh_folder, f"{name}")
    # os.makedirs(mesh_path, exist_ok=True)
    # for idx in range(verts.shape[0]):
    #     if idx % 5 == 0:
    #         obj_mesh = trimesh.Trimesh(vertices=obj_verts[idx], faces=obj_faces)
    #         human_mesh = trimesh.Trimesh(vertices=verts[idx], faces=faces)
    #         obj_mesh.export(os.path.join(mesh_path, f"obj_{idx:03d}.ply"))
    #         human_mesh.export(os.path.join(mesh_path, f"human_{idx:03d}.ply"))
    # rend_video_path = os.path.join(results_folder, '{}_{}.mp4'.format("Motion", name))
    # visualize_body_obj(verts, faces, obj_verts, obj_faces, save_path=rend_video_path, show_frame=True,
    #                    multi_angle=True, h=512, w=512)

for objn in obj_names:
    for k in quan_results[objn].keys():
        quan_results[objn][k] = float(np.mean(np.array(quan_results[objn][k])))

with open("./results.json",  "w", encoding="utf-8") as f:
    json.dump(quan_results, f)