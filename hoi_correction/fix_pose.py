import torch
import numpy as np
from pytorch3d.transforms import axis_angle_to_matrix,rotation_6d_to_matrix,matrix_to_axis_angle
import trimesh
import os
LEFT_COLLAR = 13
RIGHT_COLLAR = 14
LEFT_SHOULDER = 16
RIGHT_SHOULDER = 17
LEFT_ELBOW = 18
RIGHT_ELBOW = 19
LEFT_WRIST = 20
RIGHT_WRIST = 21


JOINT_TO_POSE_MAPPING = {
    LEFT_COLLAR: 39,    # pose indices 39:42 (joint 13)
    RIGHT_COLLAR: 42,   # pose indices 42:45 (joint 14)
    LEFT_SHOULDER: 48,  # pose indices 48:51 (joint 16)
    RIGHT_SHOULDER: 51, # pose indices 51:54 (joint 17)
    LEFT_ELBOW: 54,     # pose indices 54:57 (joint 18)
    RIGHT_ELBOW: 57,    # pose indices 57:60 (joint 19)
    LEFT_WRIST: 60,     # pose indices 60:63 (joint 20)
    RIGHT_WRIST: 63     # pose indices 63:66 (joint 21)
}

def axis_angle_to_quat(aa):
    # aa: (..., 3) axis-angle, angle = ||aa||
    angle = torch.linalg.norm(aa, dim=-1, keepdims=True).clamp_min(1e-8)
    axis  = aa / angle
    half  = 0.5 * angle
    sin_h = torch.sin(half)
    cos_h = torch.cos(half)
    # quaternion layout: (x, y, z, w)
    return torch.cat([axis * sin_h, cos_h], dim=-1)

def quat_conjugate(q):
    return torch.cat([-q[..., :3], q[..., 3:]], dim=-1)

def quat_mul(a, b):
    ax, ay, az, aw = a.unbind(-1)
    bx, by, bz, bw = b.unbind(-1)
    x = aw*bx + ax*bw + ay*bz - az*by
    y = aw*by - ax*bz + ay*bw + az*bx
    z = aw*bz + ax*by - ay*bx + az*bw
    w = aw*bw - ax*bx - ay*by - az*bz
    return torch.stack([x,y,z,w], dim=-1)

def quat_normalize(q, eps=1e-8):
    return q / (q.norm(dim=-1, keepdim=True) + eps)

def quat_to_angle_axis(q, eps=1e-8):
    q = quat_normalize(q)
    w = q[..., 3].clamp(-1.0, 1.0)
    angle = 2.0 * torch.acos(w)                      # [0, pi]
    sin_half = torch.sqrt((1.0 - w*w).clamp_min(0))  # = ||xyz||
    axis = torch.zeros_like(q[..., :3])
    mask = sin_half > 1e-5
    axis[mask] = q[..., :3][mask] / sin_half[mask].unsqueeze(-1)
    axis[~mask] = torch.tensor([0.,0.,1.], device=q.device, dtype=q.dtype)  # default
    return angle, axis

def quat_from_axis_angle(axis, angle, eps=1e-8):
    if not isinstance(axis, torch.Tensor): axis = torch.tensor(axis, dtype=torch.float32)
    if not isinstance(angle, torch.Tensor): angle = torch.tensor(angle, dtype=torch.float32)
    # normalize axis defensively
    axis = axis / (axis.norm(dim=-1, keepdim=True) + eps)
    half = 0.5 * angle
    s = torch.sin(half)[..., None]
    c = torch.cos(half)[..., None]
    return torch.cat([axis * s, c], dim=-1)
    
def pose_delta_axis_angle(poses):
    # poses: (T, N, 3)
    T, N, _ = poses.shape
    q = axis_angle_to_quat(poses)                # (T, N, 4)
    q_prev = q[:-1]                               # (T-1, N, 4)
    q_curr = q[1:]
    q_rel  = quat_mul(quat_conjugate(q_prev), q_curr)
    q_rel  = quat_normalize(q_rel)
    diff_angle, diff_axis = quat_to_angle_axis(q_rel)  # (T-1, N), (T-1, N, 3)
    return diff_angle, diff_axis

def reverse_rotate(pose_t1_aa, diff_angle, diff_axis):
    """
    Rotate pose at t+1 back to pose at t. Returns axis-angle vector (...,3).
    """
    q_t1   = quat_normalize(axis_angle_to_quat(pose_t1_aa))
    q_corr = quat_from_axis_angle(diff_axis, -diff_angle)   # inverse of delta
    q_t    = quat_normalize(quat_mul(q_t1, q_corr))
    ang, ax = quat_to_angle_axis(q_t)                       # tensors
    return ax * ang.unsqueeze(-1)                           # axis-angle vector

def forward_rotate(pose_t_aa, diff_angle, diff_axis):
    """
    Rotate pose at t forward to pose at t+1. Returns axis-angle vector (...,3).
    """
    q_t    = quat_normalize(axis_angle_to_quat(pose_t_aa))
    q_corr = quat_from_axis_angle(diff_axis,  diff_angle)   # apply delta
    q_t1   = quat_normalize(quat_mul(q_t, q_corr))
    ang, ax = quat_to_angle_axis(q_t1)
    return ax * ang.unsqueeze(-1)

def fix_joint_poses_simple(poses, joint_idx, angle_thresh=0.2, max_passes=20):
    """
    poses: (T, 156) or (T, 52, 3) axis-angle.  Assumes segment 0 is good.
    For any boundary (t -> t+1) with diff_angle > angle_thresh for this joint,
    take the segment after that boundary and reverse-rotate all its frames by that boundary's delta.
    After fixing one segment, recompute diffs and continue, up to max_passes.
    Returns: poses_fixed (same shape), fixed_boundaries (list of t indices used)
    """
    # reshape to (T, 52, 3)
    if poses.ndim == 2 and poses.shape[1] == 156:
        if isinstance(poses, torch.Tensor):
            poses_reshaped = poses.reshape(poses.shape[0], 52, 3).clone()
        else:
            poses_reshaped = poses.reshape(poses.shape[0], 52, 3).copy()
    elif poses.ndim == 3 and poses.shape[1:] == (52, 3):
        if isinstance(poses, torch.Tensor):
            poses_reshaped = poses.clone()
        else:
            poses_reshaped = poses.copy()
    else:
        raise ValueError("poses must be (T,156) or (T,52,3) axis-angle")

    torch_device = None
    np_input = not isinstance(poses_reshaped, torch.Tensor)
    if np_input:
        poses_t = torch.tensor(poses_reshaped, dtype=torch.float32)
    else:
        poses_t = poses_reshaped.float()
        torch_device = poses_t.device

    T = poses_t.shape[0]
    fixed_boundaries = []

    for _ in range(max_passes):
        diff_angle, diff_axis = pose_delta_axis_angle(poses_t)   # (T-1, N), (T-1, N, 3)
        # flips for this joint
        da = diff_angle[:, joint_idx]       # (T-1,)
        ax = diff_axis[:, joint_idx, :]     # (T-1,3)

        # find boundaries with big jumps
        flip_idxs = torch.nonzero(da > angle_thresh, as_tuple=False).flatten().tolist()
        if not flip_idxs:
            break

        # Always take the earliest boundary first, fix the segment after it.
        b = flip_idxs[0]                     # boundary between b (good) and b+1.. (bad)
        angle_b = da[b]
        axis_b  = ax[b]

        # reverse-rotate ALL frames from b+1 to the next flip (or to end)
        next_b = next((k for k in flip_idxs[1:] if k > b), None)
        seg_start = b + 1
        seg_end = (next_b if next_b is not None else (T-1))  # inclusive end boundary for rotations
        # we rotate frames seg_start..(T-1) for the target joint; per your spec, whole segment to the end of that segment
        target_slice = slice(seg_start, seg_end + 1)

        # apply reverse rotation to that joint across the segment
        poses_t[target_slice, joint_idx, :] = reverse_rotate(
            poses_t[target_slice, joint_idx, :], angle_b, axis_b
        )
        fixed_boundaries.append(int(b))
        # loop continues: recompute diffs and handle next earliest boundary (after update)

    poses_fixed = poses_t
    if np_input:
        poses_fixed = poses_fixed.cpu().numpy()
        # reshape back to original
        if poses.ndim == 2:
            poses_fixed = poses_fixed.reshape(T, 156)
    else:
        if poses.ndim == 2:
            poses_fixed = poses_fixed.reshape(T, 156)

    return poses_fixed, fixed_boundaries

def fix_joint_poses_hard(poses, joint_idx, start_idx, angle_thresh=0.2, max_passes=20):
    """
    Anchor-based version of fix_joint_poses_simple.

    Args:
        poses:
            (T, 156) or (T, 52, 3), numpy array or torch.Tensor.
        joint_idx:
            Joint index in [0, 51], e.g. wrist=20, elbow=18.
        start_idx:
            Anchor frame index. This frame is assumed to be correct.
            We repair forward from this frame, and backward from this frame.
        angle_thresh:
            Threshold for detecting a large jump, in radians.
        max_passes:
            Max repair passes for each direction.

    Returns:
        poses_fixed:
            Same type / shape convention as input.
        fixed_info:
            dict with:
              - "forward_boundaries": original boundary indices fixed on forward side
              - "backward_boundaries": original boundary indices fixed on backward side
    """
    if poses.ndim not in (2, 3):
        raise ValueError("poses must be (T,156) or (T,52,3)")
    if poses.shape[0] <= 0:
        raise ValueError("poses must have at least one frame")

    T = poses.shape[0]
    if not (0 <= start_idx < T):
        raise ValueError(f"start_idx must be in [0, {T-1}], got {start_idx}")

    # Make a writable copy
    if isinstance(poses, torch.Tensor):
        poses_fixed = poses.clone()
    else:
        poses_fixed = poses.copy()

    # -------------------------
    # 1) Forward repair
    # Assume poses[start_idx] is correct, and repair start_idx..end
    # -------------------------
    forward_chunk = poses_fixed[start_idx:]
    forward_fixed, forward_boundaries_local = fix_joint_poses_simple(
        forward_chunk,
        joint_idx=joint_idx,
        angle_thresh=angle_thresh,
        max_passes=max_passes,
    )
    poses_fixed[start_idx:] = forward_fixed

    # Local boundary b in forward_chunk maps to original boundary start_idx + b
    forward_boundaries = [start_idx + b for b in forward_boundaries_local]

    # -------------------------
    # 2) Backward repair
    # Reverse 0..start_idx so anchor becomes frame 0 in reversed sequence,
    # then reuse fix_joint_poses_simple.
    # -------------------------
    if isinstance(poses_fixed, torch.Tensor):
        backward_chunk_rev = torch.flip(poses_fixed[:start_idx + 1], dims=[0])
    else:
        backward_chunk_rev = poses_fixed[:start_idx + 1][::-1].copy()

    backward_fixed_rev, backward_boundaries_rev_local = fix_joint_poses_simple(
        backward_chunk_rev,
        joint_idx=joint_idx,
        angle_thresh=angle_thresh,
        max_passes=max_passes,
    )

    if isinstance(poses_fixed, torch.Tensor):
        backward_fixed = torch.flip(backward_fixed_rev, dims=[0])
    else:
        backward_fixed = backward_fixed_rev[::-1].copy()

    poses_fixed[:start_idx + 1] = backward_fixed

    # Mapping reversed local boundary r back to original:
    # reversed boundary r is between rev[r] and rev[r+1]
    # original indices are:
    #   rev[r]   -> orig[start_idx - r]
    #   rev[r+1] -> orig[start_idx - r - 1]
    # so the original boundary index is start_idx - r - 1
    backward_boundaries = []
    for r in backward_boundaries_rev_local:
        b_orig = start_idx - r - 1
        if 0 <= b_orig < T - 1:
            backward_boundaries.append(b_orig)

    return poses_fixed, backward_boundaries + forward_boundaries

def fix_palm_by_segment_sampling_with_contact_list(
    poses,
    betas,
    trans,
    joints,
    object_verts,
    cano_axis,
    contact_frame_list,
    smpl_model=None,
    is_left_hand=True,
    cosine_thres=0.3,
    pre_smooth=10,
    post_smooth=10,
    random_seed=42,  # kept for API compatibility; deterministic search is used below
):
    """
    Improved palm-fixing version:
    1. Split contact_frame_list into continuous segments.
    2. For each segment, search a shared twist delta using a robust score.
    3. Apply ramp smoothing before/after the segment.
    4. Run a per-frame rescue pass for remaining bad frames inside the segment.

    Returns:
        poses_fixed, fixed_frames
    """
    if is_left_hand:
        IDX_WRIST = 20
        IDX_INDEX = 25
        IDX_PINKY = 31
        IDX_ELBOW = LEFT_ELBOW
    else:
        IDX_WRIST = 21
        IDX_INDEX = 40
        IDX_PINKY = 46
        IDX_ELBOW = RIGHT_ELBOW

    T = poses.shape[0]
    poses_fixed = torch.tensor(poses.copy()).reshape(T, -1, 3).float()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if isinstance(joints, np.ndarray):
        joints_t = torch.from_numpy(joints).float()
    else:
        joints_t = joints.detach().clone().float()

    if isinstance(object_verts, np.ndarray):
        object_verts_t = torch.from_numpy(object_verts).float()
    else:
        object_verts_t = object_verts.detach().clone().float()

    def split_into_segments(frame_list):
        if not frame_list:
            return []
        frames = sorted(set(int(x) for x in frame_list))
        segments = []
        start = frames[0]
        prev = frames[0]
        for f in frames[1:]:
            if f == prev + 1:
                prev = f
            else:
                if prev - start > 5:
                    segments.append(list(range(start, prev + 1)))
                start = f
                prev = f
        if prev - start > 5:
            segments.append(list(range(start, prev + 1)))
        subsets = []
        for s in segments:
            anchor = 0
            while len(s) - anchor > 15:
                subsets.append(s[anchor:anchor + 10])
                anchor = anchor + 10
        return subsets

    def compute_twist_angle(pose_wrist):
        rotvec = pose_wrist
        angle = np.linalg.norm(rotvec)
        if angle < 1e-6:
            return 0.0
        axis = rotvec / angle
        twist_cos = np.dot(axis, cano_axis)
        twist_angle = angle * twist_cos
        return np.rad2deg(twist_angle)

    def compute_twist_angle_batch(pose_wrist, cano_axis, eps=1e-6):
        """
        pose_wrist: (..., 3) axis-angle rotvecs
        cano_axis:  (3,) canonical bone axis

        return:
            twist_angle_deg: (...) twist angle in degrees
        """
        cano_axis = cano_axis / (torch.norm(cano_axis) + eps)

        angle = torch.norm(pose_wrist, dim=-1)  # (...)
        axis = pose_wrist / (angle.unsqueeze(-1) + eps)  # (..., 3)

        twist_cos = torch.sum(axis * cano_axis, dim=-1)  # (...)
        twist_angle = angle * twist_cos  # (...)
        twist_angle_deg = torch.rad2deg(twist_angle)

        # keep zero-rotvec exactly zero
        twist_angle_deg = torch.where(
            angle < eps,
            torch.zeros_like(twist_angle_deg),
            twist_angle_deg
        )
        return twist_angle_deg

    def save_arrow_ply(position, direction, shape=[0.01, 0.02, 0.1], out_path="arrow.ply"):
        """
        生成一个从 position 指向 direction 的箭头，并保存为 .ply

        Args:
            position: (3,) 起点
            direction: (3,) 方向向量
            shape: (3,) = [shaft_radius, head_radius, head_length_ratio]
            out_path: 输出 ply 路径
        """
        if isinstance(position, torch.Tensor):
            position = position.detach().cpu().numpy()
        if isinstance(direction, torch.Tensor):
            direction = direction.detach().cpu().numpy()
        position = np.asarray(position, dtype=float).reshape(3)
        direction = np.asarray(direction, dtype=float).reshape(3)
        shape = np.asarray(shape, dtype=float).reshape(3)

        shaft_radius, head_radius, head_length_ratio = shape

        length = np.linalg.norm(direction)
        if length < 1e-8:
            raise ValueError("direction 不能是零向量")

        head_length_ratio = np.clip(head_length_ratio, 1e-4, 0.95)
        head_length = length * head_length_ratio
        shaft_length = length - head_length

        # 箭杆：默认沿 z 轴，从 z=0 到 z=shaft_length
        shaft = trimesh.creation.cylinder(
            radius=shaft_radius,
            height=shaft_length,
            sections=32
        )
        shaft.apply_translation([0, 0, shaft_length / 2.0])

        # 箭头：默认沿 z 轴，底面在 z=shaft_length，尖端在 z=length
        head = trimesh.creation.cone(
            radius=head_radius,
            height=head_length,
            sections=32
        )
        head.apply_translation([0, 0, shaft_length])

        arrow = trimesh.util.concatenate([shaft, head])

        # 把默认 z 轴方向旋转到 direction
        z_axis = np.array([0.0, 0.0, 1.0])
        target = direction / length

        T = trimesh.geometry.align_vectors(z_axis, target)
        arrow.apply_transform(T)

        # 平移到 position
        arrow.apply_translation(position)

        # 导出 ply
        arrow.export(out_path)

    def show_res(human_joints, human_verts, obj_verts):
        save_path = "/move/u/dixuan/pj/InterAct/save_fix/normal_test"
        wrist = human_joints[:, IDX_WRIST, :]  # (T,3)
        idx = human_joints[:, IDX_INDEX, :]  # (T,3)
        pinky = human_joints[:, IDX_PINKY, :]  # (T,3)
        v1 = idx - wrist  # (T,3)
        v2 = pinky - wrist  # (T,3)
        normals = torch.cross(v1, v2, dim=1)  # (T,3)
        if is_left_hand:
            normals = -normals
        normals = normals / (torch.norm(normals, dim=-1, keepdim=True) + 1e-8)
        centroid = (wrist + idx + pinky) / 3.0  # (T,3)

        object_rel = obj_verts - centroid.unsqueeze(1)  # (T, N, 3)

        object_dist = torch.norm(object_rel, dim=-1)
        values, indices = torch.topk(object_dist, k=50, dim=1, largest=False)
        object_normals = object_rel / (object_dist.unsqueeze(-1) + 1e-8)

        close_normals = torch.gather(object_normals, dim=1, index=indices.unsqueeze(-1).expand(-1, -1, 3))
        close_points = torch.gather(obj_verts, dim=1, index=indices.unsqueeze(-1).expand(-1, -1, 3))
        close_normals_mean = torch.mean(close_normals, dim=1)
        close_points_mean = torch.mean(close_points, dim=1)
        OBJ_MESH = trimesh.load('/move/u/dixuan/pj/InterAct/omomo/objects/vacuum/vacuum.obj')
        cosine = torch.sum(normals.unsqueeze(1) * close_normals, dim=-1)
        for i in range(0, wrist.shape[0], 10):
            human_mesh = trimesh.Trimesh(vertices=human_verts[i], faces=smpl_model.f.detach().cpu().numpy())
            human_mesh.export(os.path.join(save_path, f"human_{i:03d}.ply"))
            save_arrow_ply(centroid[i], normals[i], out_path=os.path.join(save_path, f"normal_{i:03d}.ply"))
            save_arrow_ply(close_points_mean[i], close_normals_mean[i], out_path=os.path.join(save_path, f"normal_obj_{i:03d}.ply"))
            obj_point_cloud = trimesh.PointCloud(close_points[i].detach().cpu().numpy())
            obj_point_cloud.export(os.path.join(save_path, f"point_cloud_{i:03d}.ply"))
            obj_mesh = trimesh.Trimesh(vertices=obj_verts[i].detach().cpu().numpy(), faces=OBJ_MESH.faces)
            obj_mesh.export(os.path.join(save_path, f"obj_{i:03d}.ply"))

    def calc_segments_consine(human_joints, obj_verts, segment=None):
        if segment is None:
            wrist = human_joints[:, IDX_WRIST, :]  # (T,3)
            idx = human_joints[:, IDX_INDEX, :]  # (T,3)
            pinky = human_joints[:, IDX_PINKY, :]  # (T,3)
        else:
            wrist = human_joints[segment, IDX_WRIST, :]  # (T,3)
            idx = human_joints[segment, IDX_INDEX, :]  # (T,3)
            pinky = human_joints[segment, IDX_PINKY, :]  # (T,3)

        v1 = idx - wrist  # (T,3)
        v2 = pinky - wrist  # (T,3)
        normals = torch.cross(v1, v2, dim=1)  # (T,3)
        if is_left_hand:
            normals = -normals
        normals = normals / (torch.norm(normals, dim=-1, keepdim=True) + 1e-8)
        centroid = (wrist + idx + pinky) / 3.0  # (T,3)

        if segment is None:
            object_rel = obj_verts - centroid.unsqueeze(1)  # (T, N, 3)
        else:
            object_rel = obj_verts[segment] - centroid.unsqueeze(1)  # (T, N, 3)

        object_dist = torch.norm(object_rel, dim=-1)
        values, indices = torch.topk(object_dist, k=10, dim=1, largest=False)
        object_normals = object_rel / (object_dist.unsqueeze(-1) + 1e-8)

        close_normals = torch.gather(object_normals, dim=1, index=indices.unsqueeze(-1).expand(-1, -1, 3))
        cosine = torch.sum(normals.unsqueeze(1) * close_normals, dim=-1)
        consine_mean = torch.mean(cosine, dim=-1)
        return consine_mean

    def sample_and_fix(poses, segment, obj_verts):

        cano_norm = torch.tensor(cano_axis).float()
        cano_norm = cano_norm / (torch.norm(cano_norm, dim=-1, keepdim=True) + 1e-8)
        best_pose = None
        best_cos = None
        for delta in range(0, 100, 2):
            delta_angle = -110 + 180 / 100 * delta
            delta_rad = np.deg2rad(delta_angle)
            delta_aa = cano_norm * delta_rad
            poses_twist = poses.clone()[segment]
            poses_twist[:, IDX_WRIST] = delta_aa.unsqueeze(0).repeat(len(segment), 1)
            poses_twist = poses_twist.reshape(-1, 156)
            smplx_output = smpl_model(
                pose_body=poses_twist[:, 3:66].float().to(device),
                pose_hand=poses_twist[:, 66:156].float().to(device),
                betas=torch.from_numpy(betas[None, :]).float().to(device),
                root_orient=poses_twist[:, :3].float().to(device),
                trans=torch.from_numpy(trans[segment]).float().to(device)
            )
            delta_Jtr = smplx_output.Jtr.detach()
            consine_mean = calc_segments_consine(delta_Jtr, obj_verts[segment].to(device))
            consine_mean = torch.mean(consine_mean)
            if best_cos is None:
                best_cos = consine_mean
                best_pose = poses_twist.clone()
            else:
                if consine_mean > best_cos:
                    best_cos = consine_mean
                    best_pose = poses_twist
        if best_pose is not None:
            poses[segment] = best_pose.reshape(-1, 52, 3)
        return poses



    # def test_twist():
    #
    #     with np.load(os.path.join("/move/u/dixuan/pj/InterAct/omomo_raw/sequences_canonical", "sub12_vacuum_000", 'human.npz'), allow_pickle=True) as f:
    #         poses, betas, trans, gender = f['poses'], f['betas'], f['trans'], str(f['gender'])
    #
    #     save_path = "/move/u/dixuan/pj/InterAct/save_fix/palm_test"
    #     wrist = joints_t[:, IDX_WRIST, :]  # (T,3)
    #     elbow = joints_t[:, IDX_ELBOW, :]  # (T,3)
    #     axis_wrist = wrist - elbow
    #     save_arrow_ply(elbow[50], axis_wrist[50], out_path=os.path.join(save_path, "elbow.ply"))
    #     axis_norm = axis_wrist / (torch.norm(axis_wrist, dim=-1, keepdim=True) + 1e-8)
    #     cano_norm = torch.tensor(cano_axis).float()
    #     cano_norm = cano_norm / (torch.norm(cano_norm, dim=-1, keepdim=True) + 1e-8)
    #     device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #     for delta in range(100):
    #         delta_angle = -100 + 200 / 100 * delta
    #         delta_rad = np.deg2rad(delta_angle)
    #         delta_aa = cano_norm * delta_rad
    #         poses_50 = poses[50]
    #         poses_50[IDX_WRIST * 3: IDX_WRIST * 3 + 3] = delta_aa
    #         poses_50 = poses_50.reshape(1, -1)
    #
    #         smplx_output = smpl_model(
    #             pose_body=torch.from_numpy(poses_50[:, 3:66]).float().to(device),
    #             pose_hand=torch.from_numpy(poses_50[:, 66:156]).float().to(device),
    #             betas=torch.from_numpy(betas[None, :]).float().to(device),
    #             root_orient=torch.from_numpy(poses_50[:, :3]).float().to(device),
    #             trans=torch.from_numpy(trans[50:51]).float().to(device)
    #         )
    #         verts = smplx_output.v.detach().cpu().numpy()[0]
    #         joints = smplx_output.Jtr.detach().cpu().numpy()[0]
    #
    #
    #         human = trimesh.Trimesh(vertices=verts, faces=smpl_model.f.detach().cpu().numpy())
    #         human.export(os.path.join(save_path, f"human_{int(delta_angle)+100:04d}.ply"))
    #     assert False
    # test_twist()


    contact_segments = split_into_segments(contact_frame_list)

    if len(contact_segments) > 0:
        for segment_i in contact_segments:
            segment_cos = calc_segments_consine(joints_t,object_verts_t.clone(), segment_i)
            if torch.mean(segment_cos) < cosine_thres:
                poses_fixed = sample_and_fix(poses_fixed, segment_i, object_verts_t.clone())
    poses_fixed = poses_fixed.reshape(-1, 156)

    # smplx_output = smpl_model(
    #     pose_body=poses_fixed[:, 3:66].float().to(device),
    #     pose_hand=poses_fixed[:, 66:156].float().to(device),
    #     betas=torch.from_numpy(betas[None, :]).float().to(device),
    #     root_orient=poses_fixed[:, :3].float().to(device),
    #     trans=torch.from_numpy(trans).float().to(device)
    # )
    # verts = smplx_output.v.detach().cpu().numpy()
    # jointsss = smplx_output.Jtr
    # show_res(jointsss, verts, object_verts_t)

    return poses_fixed.detach().cpu().numpy()