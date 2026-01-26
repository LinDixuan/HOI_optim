import trimesh
import os
import pickle
import numpy as np
import torch
from bps_torch.bps import bps_torch
import joblib
import pytorch3d.transforms as transforms
from argument_parser import parse_opt
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
    decide_no_force_closure_from_objects,
    finger_smooth_transition,
    fix_feet,
    generate_T_pose,
    interaction_to_navigation_smooth_transition,
    load_planned_path_as_waypoints,
    mirror_rot_6d,
    navigation_to_interaction_smooth_transition,
    smooth_res,
    smplx_ik,
)

from sample import (
    run_grasp_generation,
    call_grasp_model_long_seq,
    build_finger_opt,
    build_finger_trainer,
    build_coarse_interaction_trainer,
    build_fine_interaction_trainer,
    save_interaction_motion_meshes,
    render_motion_clip)

def find_hoi_contact(data_dict, left_wrist_pos, right_wrist_pos, left_wrist_rot_mat, right_wrist_rot_mat):
    device = left_wrist_pos.device
    obj_rot_mat = data_dict["obj_rot_mat"].to(device)  # T X 3 X 3
    obj_com_pos = data_dict["obj_com_pos"].to(device)  # T X 3
    left_wrist_rot_mat = left_wrist_rot_mat.to(device)
    right_wrist_rot_mat = right_wrist_rot_mat.to(device)

    left_wrist_pos_in_obj = (
        (obj_rot_mat.transpose(1, 2) @ (left_wrist_pos - obj_com_pos).unsqueeze(-1))[
            ..., 0
        ]
    ).cpu()  # T X 3
    left_wrist_rot_mat_in_obj = (
            obj_rot_mat.transpose(1, 2) @ left_wrist_rot_mat
    ).cpu()  # T X 3 X 3
    right_wrist_pos_in_obj = (
        (obj_rot_mat.transpose(1, 2) @ (right_wrist_pos - obj_com_pos).unsqueeze(-1))[
            ..., 0
        ]
    ).cpu()  # T X 3
    right_wrist_rot_mat_in_obj = (
            obj_rot_mat.transpose(1, 2) @ right_wrist_rot_mat
    ).cpu()  # T X 3 X 3

    contact_labels = data_dict['contact_labels']
    left_contact_labels = contact_labels[:, 0]
    right_contact_labels = contact_labels[:, 1]
    left_contact = len(left_contact_labels[left_contact_labels > 0.95]) > 15
    right_contact = len(right_contact_labels[right_contact_labels > 0.95]) > 15

    contact_begin_frame, contact_end_frame = 121, -1

    if left_contact:
        left_begin_frame = torch.where(left_contact_labels > 0.95)[0][0].item()
        left_end_frame = torch.where(left_contact_labels > 0.95)[0][-1].item()
        contact_begin_frame = min(left_begin_frame, contact_begin_frame)
        contact_end_frame = max(left_end_frame, contact_end_frame)
    else:
        left_begin_frame, left_end_frame = -1, -1

    if right_contact:
        right_begin_frame = torch.where(right_contact_labels > 0.95)[0][0].item()
        right_end_frame = torch.where(right_contact_labels > 0.95)[0][-1].item()
        contact_begin_frame = min(right_begin_frame, contact_begin_frame)
        contact_end_frame = max(right_end_frame, contact_end_frame)
    else:
        right_begin_frame, right_end_frame = -1, -1

    if left_contact:
        left_wrist_pos_in_obj_mean = left_wrist_pos_in_obj[
            left_contact_labels > 0.95
            ].mean(dim=0)
        left_wrist_rot_mat_in_obj_mean = transforms.quaternion_to_matrix(
            transforms.matrix_to_quaternion(
                left_wrist_rot_mat_in_obj[left_contact_labels > 0.95].mean(dim=0)
            )
        )
    else:
        left_wrist_pos_in_obj_mean, left_wrist_rot_mat_in_obj_mean = None, None
    if right_contact:
        right_wrist_pos_in_obj_mean = right_wrist_pos_in_obj[
            right_contact_labels > 0.95
            ].mean(dim=0)
        right_wrist_rot_mat_in_obj_mean = transforms.quaternion_to_matrix(
            transforms.matrix_to_quaternion(
                right_wrist_rot_mat_in_obj[right_contact_labels > 0.95].mean(dim=0)
            )
        )
    else:
        right_wrist_pos_in_obj_mean, right_wrist_rot_mat_in_obj_mean = None, None

    left_wrist = {
        "wrist_pos": left_wrist_pos_in_obj_mean,
        "wrist_rot": left_wrist_rot_mat_in_obj_mean,
    }
    right_wrist = {
        "wrist_pos": right_wrist_pos_in_obj_mean,
        "wrist_rot": right_wrist_rot_mat_in_obj_mean,
    }

    return (
        left_contact,
        right_contact,
        contact_begin_frame,
        contact_end_frame,
        left_begin_frame,
        left_end_frame,
        right_begin_frame,
        right_end_frame,
        left_wrist,
        right_wrist,
    )

def gen_motion(dataset, data_dict):
    motion = data_dict['motion'].unsqueeze(0)
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

    T = motion.shape[1]
    normalized_obj_com_pos = dataset.normalize_obj_pos_min_max(
        curr_gt_obj_com_pos
    )

    all_res = torch.cat([normalized_obj_com_pos.reshape(T, 3),
                         curr_gt_obj_rot_mat.reshape(T, 9),
                         motion.reshape(T, -1),], dim=-1).unsqueeze(0)

    left_wrist_position = human_jnts_list[:, 20, :]
    left_wrist_rot = curr_local_rot_mat[:, 20]
    right_wrist_position = human_jnts_list[:, 21, :]
    right_wrist_rot = curr_local_rot_mat[:, 21]

    return all_res, left_wrist_position, right_wrist_position, left_wrist_rot, right_wrist_rot


def main(opt=None, device='cpu'):
    all_object_data_dict_data = pickle.load(open("all_object_data_dict_for_eval.pkl", "rb"))
    all_object_data_dict = all_object_data_dict_data["all_object_data_dict"]
    ref_data_dict = all_object_data_dict_data["ref_data_dict"]
    rest_hand_pose = pickle.load(open("rest_hand_pose.pkl", "rb"))
    rest_left_hand_pose = rest_hand_pose["left_hand_pose"].cuda()  # 45
    rest_right_hand_pose = rest_hand_pose["right_hand_pose"].cuda()  # 45
    rest_left_hand_local_rot_6d = transforms.matrix_to_rotation_6d(
        transforms.axis_angle_to_matrix(rest_left_hand_pose.reshape(-1, 3) * 0.8)
    ).reshape(1, 15 * 6)
    rest_right_hand_local_rot_6d = transforms.matrix_to_rotation_6d(
        transforms.axis_angle_to_matrix(rest_right_hand_pose.reshape(-1, 3) * 0.8)
    ).reshape(1, 15 * 6)

    finger_opt = build_finger_opt(opt)
    finger_trainer = build_finger_trainer(finger_opt, device)
    fine_interaction_trainer = build_fine_interaction_trainer(
        opt,
        device=device,
        milestone="10",
    )
    interaction_trainer = build_coarse_interaction_trainer(
        opt,
        device=device,
        milestone="10",
    )

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

    for idx in range(len(train_dataset)):
        data_dict = train_dataset[idx]
        obj_name = data_dict["obj_name"]

        no_fc = decide_no_force_closure_from_objects(obj_name)
        (all_res, left_wrist_pos, right_wrist_pos,
         left_wrist_rot_mat, right_wrist_rot_mat) = gen_motion(train_dataset, data_dict)

        all_res = all_res.to(device)
        (
            left_contact,
            right_contact,
            contact_begin_frame,
            contact_end_frame,
            left_begin_frame,
            left_end_frame,
            right_begin_frame,
            right_end_frame,
            left_wrist,
            right_wrist,
        ) = find_hoi_contact(data_dict, left_wrist_pos, right_wrist_pos, left_wrist_rot_mat, right_wrist_rot_mat)

        (
            right_wrist_pos_in_obj,
            right_wrist_rot_mat_in_obj,
            right_finger_local_rot_6d,
            right_wrist_pose,
            left_wrist_pos_in_obj,
            left_wrist_rot_mat_in_obj,
            left_finger_local_rot_6d,
            left_wrist_pose,
        ) = run_grasp_generation(
            curr_object_name=obj_name,
            left_contact=left_contact,
            right_contact=right_contact,
            left_wrist_init_pose=left_wrist,
            right_wrist_init_pose=right_wrist,
            no_force_closure=no_fc,
        )

        # all_res_list: BS X T X (12+24*3+22*6)
        if left_finger_local_rot_6d is not None:
            left_finger_local_rot_6d = left_finger_local_rot_6d.to(device)
        if right_finger_local_rot_6d is not None:
            right_finger_local_rot_6d = right_finger_local_rot_6d.to(device)

        finger_all_res_list = call_grasp_model_long_seq(
            all_res,
            ref_obj_rot_mat=data_dict["reference_obj_rot_mat"].to(device),
            left_contact=left_contact,
            right_contact=right_contact,
            left_wrist_pos=left_wrist_pos.to(device),
            right_wrist_pos=right_wrist_pos.to(device),
            left_wrist_rot_mat=left_wrist_rot_mat.to(device),
            right_wrist_rot_mat=right_wrist_rot_mat.to(device),
            rest_left_hand_local_rot_6d=rest_left_hand_local_rot_6d.to(device),
            rest_right_hand_local_rot_6d=rest_right_hand_local_rot_6d.to(device),
            finger_trainer=finger_trainer,
            interaction_trainer=fine_interaction_trainer,
            object_name=obj_name,
            left_begin_frame=left_begin_frame,
            left_end_frame=left_end_frame,
            right_begin_frame=right_begin_frame,
            right_end_frame=right_end_frame,
            left_finger_local_rot_6d=left_finger_local_rot_6d,
            right_finger_local_rot_6d=right_finger_local_rot_6d,
        )

        vis_tag = f"{idx:05d}_{obj_name}"
        interaction_trainer.add_waypoints_xy = False
        dest_mesh_vis_folder = f"./omo_result/{idx:05d}_{obj_name}"
        os.makedirs(dest_mesh_vis_folder, exist_ok=True)
        (
            dest_mesh_vis_folder,
            params_path,
        ) = save_interaction_motion_meshes(
            interaction_trainer=interaction_trainer,
            all_res_list=all_res,
            ref_obj_rot_mat=data_dict["reference_obj_rot_mat"],
            ref_data_dict=ref_data_dict,
            step="10",
            planned_waypoints_pos=torch.zeros(1),
            curr_object_name=obj_name,
            vis_tag=vis_tag,
            dest_mesh_vis_folder=dest_mesh_vis_folder,
            finger_all_res_list=finger_all_res_list,
        )

        video_path = render_motion_clip(
            mesh_save_folders=[dest_mesh_vis_folder],
            initial_end_obj_mesh_paths=[""],
            p_idx=0,
            video_paths=[dest_mesh_vis_folder],
            use_guidance_str="no",
            interaction_checkpoint_epoch=10,
            video_save_dir_name=dest_mesh_vis_folder,
        )


if __name__ == "__main__":
    opt = parse_opt()
    device = torch.device(f"cuda:{opt.device}" if torch.cuda.is_available() else "cpu")
    main(opt, device)