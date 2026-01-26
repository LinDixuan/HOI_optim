import trimesh
import os
import pickle
import numpy as np
import torch
from bps_torch.bps import bps_torch
import joblib
import pytorch3d.transforms as transforms
from manip.data.omomo_dataset import OMOMODataset
from manip.data.cano_traj_dataset import (
    CanoObjectTrajDataset,
    get_smpl_parents,
    quat_ik_torch,
)
from manip.utils.trainer_utils import (
    canonicalize_first_human_and_waypoints,
    cycle,
    find_contact_frames,
    load_palm_vertex_ids,
    run_smplx_model,
    smooth_res,
)

def gen_motion(dataset, data_dict):
    motion = data_dict['motion'].unsqueeze(0)
    contact_labels = data_dict['contact_labels']
    left_contact_labels = contact_labels[:, 0]
    right_contact_labels = contact_labels[:, 1]
    num_seq = motion.shape[0]
    num_joints = 24
    normalized_global_jpos = motion[:, :, 0: num_joints * 3].reshape(num_seq, -1, num_joints, 3)
    global_jpos = dataset.de_normalize_jpos_min_max(
        normalized_global_jpos.reshape(-1, num_joints, 3)
    )
    global_jpos = global_jpos.reshape(num_seq, -1, num_joints, 3)

    global_root_jpos = global_jpos[:, :, 0, :].clone()  # N X T X 3

    global_rot_6d = motion[:, :, 24 * 3: 24 * 3 + 22 * 6].reshape(num_seq, -1, 22, 6)
    global_rot_mat = transforms.rotation_6d_to_matrix(
        global_rot_6d
    )  # N X T X 22 X 3 X 3

    trans2joint = torch.tensor(data_dict["trans2joint"][None,:])  # BS X  3
    seq_len = data_dict["seq_len"]

    if motion.shape[0] != trans2joint.shape[0]:
        trans2joint = trans2joint.repeat(num_seq, 1, 1)  # N X 24 X 3
        seq_len = seq_len.repeat(num_seq)  # N

    for idx in range(num_seq):
        curr_global_rot_mat = global_rot_mat[idx]  # T X 22 X 3 X 3
        curr_local_rot_mat = quat_ik_torch(curr_global_rot_mat)  # T X 22 X 3 X 3
        curr_local_rot_aa_rep = transforms.matrix_to_axis_angle(
            curr_local_rot_mat
        )  # T X 22 X 3
        # 20: left_wrist, 21: right_wrist

        curr_global_root_jpos = global_root_jpos[idx]  # T X 3

        curr_trans2joint = trans2joint[idx: idx + 1].clone()  # 1 X 3

        root_trans = curr_global_root_jpos + curr_trans2joint.to(
            curr_global_root_jpos.device
        )  # T X 3

        # Generate global joint position
        bs = 1
        betas = data_dict["betas"]
        gender = data_dict["gender"]

        curr_gt_obj_rot_mat = data_dict["obj_rot_mat"]  # T X 3 X 3
        curr_gt_obj_com_pos = data_dict["obj_com_pos"]  # T X 3

        curr_seq_name = data_dict["seq_name"]
        object_name = data_dict["obj_name"]

        mesh_jnts, mesh_verts, mesh_faces = run_smplx_model(
            root_trans[None].cuda(),
            curr_local_rot_aa_rep[None].cuda(),
            betas.cuda(),
            [gender],
            dataset.bm_dict,
            return_joints24=True,
        )

        obj_rest_verts, obj_mesh_faces = dataset.load_rest_pose_object_geometry(
            object_name
        )
        obj_rest_verts = torch.from_numpy(obj_rest_verts)

        gt_obj_mesh_verts = dataset.load_object_geometry_w_rest_geo(
            curr_gt_obj_rot_mat, curr_gt_obj_com_pos, obj_rest_verts.float()
        )

        human_jnts_list = mesh_jnts[0]
        # 20: left_wrist 21: right_writs
        human_verts_list = mesh_verts[0]
        trans_list = root_trans

        human_root_pos_list = root_trans
        human_jnts_pos_list = mesh_jnts[0]

    T = human_verts_list.shape[0]
    h_faces = mesh_faces.detach().cpu().numpy()
    o_faces = obj_mesh_faces
    save_path = '/home/dixuan/sony/hoifhli_release/temp_files/mesh_test'
    for idx in range(1):
        human_verts = human_verts_list[idx].detach().cpu().numpy()
        object_verts = gt_obj_mesh_verts[idx]
        mesh_human = trimesh.Trimesh(vertices=human_verts, faces=h_faces)
        mesh_human.export(os.path.join(save_path, f'human_b_{idx:03d}.ply'))
        mesh_object = trimesh.Trimesh(vertices=object_verts, faces=o_faces)
        mesh_object.export(os.path.join(save_path, f'obj_{idx:03d}.ply'))

def look_omomo():
    train_dataset = CanoObjectTrajDataset(
        train=False,
        data_root_folder="./data/processed_data",
        window=120,
        use_object_splits=False,
        input_language_condition=True,
        use_first_frame_bps=False,
        use_random_frame_bps=True,
        use_object_keypoints=True,
        load_ds=True,
    )
    a = train_dataset[0]
    gen_motion(train_dataset, a)
    file_p = "/home/dixuan/sony/InterAct/omomo_correct/sequences_canonical/sub10_clothesstand_000"
    human = np.load(os.path.join(file_p, "human.npz"))
    # poses(132, 156)
    # betas(16, )
    # trans(132, 3)

    obj = np.load(os.path.join(file_p, "object.npz"))
    # angles(132, 3)
    # trans(132, 3)
    # name()


def look_fi():
    with open("/home/dixuan/sony/hoifhli_release/grasp_generation/objects/cache/largebox.pkl", "rb") as f:
        file = pickle.load(f)
    print(file)

def calc_bps():
    save_path = "/home/dixuan/sony/hoifhli_release/temp_files/bps_test"
    obj_bps = torch.load("/home/dixuan/sony/hoifhli_release/bps.pt", map_location="cpu")['obj']
    encoder = bps_torch(
        bps_type="custom",  # 有的版本叫 "custom"
        custom_basis=obj_bps[0],  # 也可能叫 basis_points=bps / bps=bps
        device='cpu',
    )
    obj_mesh = trimesh.load("/home/dixuan/sony/hoifhli_release/grasp_generation/objects/floorlamp.obj")
    verts = torch.Tensor(obj_mesh.vertices)
    res = encoder.encode(verts, feature_type=['dists','deltas'])
    deltas = res['deltas']
    points = obj_bps[0] + deltas[0].cpu()

    pc = trimesh.points.PointCloud(points.detach().cpu().numpy())
    pc.export(os.path.join(save_path, "floorlamp_encode.ply"))

def vis_bps():

    save_path = "/home/dixuan/sony/hoifhli_release/temp_files/bps_test"
    obj_bps = torch.load("/home/dixuan/sony/hoifhli_release/bps.pt", map_location="cpu")['obj']
    with open("/home/dixuan/sony/hoifhli_release/all_object_data_dict_for_eval.pkl", "rb") as f:
        file = pickle.load(f)
    input_bps = file['all_object_data_dict']['floorlamp']['input_obj_bps']

    points = obj_bps[0] + input_bps[0, 0]
    obj_mesh = trimesh.load("/home/dixuan/sony/hoifhli_release/grasp_generation/objects/floorlamp.obj")
    obj_mesh.export(os.path.join(save_path, "floorlamp.ply"))
    pc = trimesh.points.PointCloud(points.detach().cpu().numpy())
    pc.export(os.path.join(save_path, "floorlamp_pc.ply"))

def trans_obj_ply():
    for dirpath, dirnames, filenames in os.walk("/home/dixuan/sony/hoifhli_release/grasp_generation/objects"):
        for name in filenames:
            full_path = os.path.join(dirpath, name)
            if ".obj" in full_path:
                mesh = trimesh.load(full_path)
                mesh.export(full_path.replace(".obj", ".ply"))

def trans_ply_obj():
    for dirpath, dirnames, filenames in os.walk("/home/dixuan/sony/hoifhli_release/data/processed_data/rest_object_geo"):
        for name in filenames:
            full_path = os.path.join(dirpath, name)
            if ".ply" in full_path:
                f_n = os.path.basename(full_path)
                mesh = trimesh.load(full_path)
                mesh.export(os.path.join("/home/dixuan/sony/hoifhli_release/grasp_generation/objects", f_n.replace(".ply", ".obj")))

if __name__ == "__main__":
    look_omomo()