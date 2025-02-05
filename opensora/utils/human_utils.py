
from argparse import Namespace
import copy
import json
import numpy as np
import os
import torch
from smplx import SMPL as _SMPL
from smplx.lbs import vertices2joints
from smplx.utils import SMPLOutput
from opensora.dataset.transform import get_kpmaps
from third_parties.humanml3d.common.quaternion import qinv, qrot
from third_parties.humanml3d.motion_representation import process_file, uniform_skeleton
from third_parties.motiongpt.mGPT.render.pyrender.hybrik_loc2rot import HybrIKJointsToRotmat
from third_parties.smpler_x.common.utils.human_models import smpl_x


"""
We create a superset of joints containing the OpenPose joints together with the ones that each dataset provides.
We keep a superset of 24 joints such that we include all joints from every dataset.
If a dataset doesn't provide annotations for a specific joint, we simply ignore it.
The joints used here are the following:
"""
JOINT_NAMES = [
# 25 OpenPose joints (in the order provided by OpenPose)
'OP Nose',
'OP Neck',
'OP RShoulder',
'OP RElbow',
'OP RWrist',
'OP LShoulder',
'OP LElbow',
'OP LWrist',
'OP MidHip',
'OP RHip',
'OP RKnee',
'OP RAnkle',
'OP LHip',
'OP LKnee',
'OP LAnkle',
'OP REye',
'OP LEye',
'OP REar',
'OP LEar',
'OP LBigToe',
'OP LSmallToe',
'OP LHeel',
'OP RBigToe',
'OP RSmallToe',
'OP RHeel',
# 24 Ground Truth joints (superset of joints from different datasets)
'Right Ankle',
'Right Knee',
'Right Hip',
'Left Hip',
'Left Knee',
'Left Ankle',
'Right Wrist',
'Right Elbow',
'Right Shoulder',
'Left Shoulder',
'Left Elbow',
'Left Wrist',
'Neck (LSP)',
'Top of Head (LSP)',
'Pelvis (MPII)',
'Thorax (MPII)',
'Spine (H36M)',
'Jaw (H36M)',
'Head (H36M)',
'Nose',
'Left Eye',
'Right Eye',
'Left Ear',
'Right Ear'
]

# Dict containing the joints in numerical order
JOINT_IDS = {JOINT_NAMES[i]: i for i in range(len(JOINT_NAMES))}

# Map joints to SMPL joints
JOINT_MAP = {
'OP Nose': 24, 'OP Neck': 12, 'OP RShoulder': 17,
'OP RElbow': 19, 'OP RWrist': 21, 'OP LShoulder': 16,
'OP LElbow': 18, 'OP LWrist': 20, 'OP MidHip': 0,
'OP RHip': 2, 'OP RKnee': 5, 'OP RAnkle': 8,
'OP LHip': 1, 'OP LKnee': 4, 'OP LAnkle': 7,
'OP REye': 25, 'OP LEye': 26, 'OP REar': 27,
'OP LEar': 28, 'OP LBigToe': 29, 'OP LSmallToe': 30,
'OP LHeel': 31, 'OP RBigToe': 32, 'OP RSmallToe': 33, 'OP RHeel': 34,
'Right Ankle': 8, 'Right Knee': 5, 'Right Hip': 45,
'Left Hip': 46, 'Left Knee': 4, 'Left Ankle': 7,
'Right Wrist': 21, 'Right Elbow': 19, 'Right Shoulder': 17,
'Left Shoulder': 16, 'Left Elbow': 18, 'Left Wrist': 20,
'Neck (LSP)': 47, 'Top of Head (LSP)': 48,
'Pelvis (MPII)': 49, 'Thorax (MPII)': 50,
'Spine (H36M)': 51, 'Jaw (H36M)': 52,
'Head (H36M)': 53, 'Nose': 24, 'Left Eye': 26,
'Right Eye': 25, 'Left Ear': 28, 'Right Ear': 27
}

SMPL_JOINT_NAMES = [
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hand",
    "right_hand",
]

ASSETS_DIR = '/mnt/data/rongyu/projects/Open-Sora-Plan/assets/'
SMPLX_OP25 = json.load(open(os.path.join(ASSETS_DIR, 'SMPLX_OpenPose_mapping/smplx_openpose25.json')))

SMPL_KS = ['global_orient', 'body_pose', 'betas', 'transl']


class SMPL_SPIN(_SMPL):
    """ Extension of the official SMPL implementation to support more joints """

    def __init__(self, *args, **kwargs):
        super(SMPL_SPIN, self).__init__(*args, **kwargs)
        joints = [JOINT_MAP[i] for i in JOINT_NAMES]
        JOINT_REGRESSOR_TRAIN_EXTRA = os.path.join(args[0], 'J_regressor_extra.npy')
        J_regressor_extra = np.load(JOINT_REGRESSOR_TRAIN_EXTRA)
        self.register_buffer('J_regressor_extra', torch.tensor(J_regressor_extra, dtype=torch.float32))
        self.joint_map = torch.tensor(joints, dtype=torch.long)

    def forward(self, *args, **kwargs):
        kwargs['get_skin'] = True
        smpl_output = super(SMPL_SPIN, self).forward(*args, **kwargs)
        extra_joints = vertices2joints(self.J_regressor_extra, smpl_output.vertices)
        joints = torch.cat([smpl_output.joints, extra_joints], dim=1)
        joints = joints[:, self.joint_map, :]
        output = SMPLOutput(vertices=smpl_output.vertices,
                            global_orient=smpl_output.global_orient,
                            body_pose=smpl_output.body_pose,
                            joints=joints,
                            betas=smpl_output.betas,
                            full_pose=smpl_output.full_pose,
                            j45=smpl_output.joints)
        return output


_smpl_spin = SMPL_SPIN(os.path.join(ASSETS_DIR, 'spin_data/'))  # for supervision
# 'assets/body_models/SMPL_python_v.1.1.0/smpl/models/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl')

_smplx_orig = copy.deepcopy(smpl_x.layer['neutral']).cuda()

mot263_mean = torch.from_numpy(np.load(os.path.join(ASSETS_DIR, 't2m/meta/mean.npy'))).float()  #TODO: incl gender
mot263_std = torch.from_numpy(np.load(os.path.join(ASSETS_DIR, 't2m/meta/std.npy'))).float()


def convert_crop_cam_to_orig_img(cam, bbox, img_width, img_height):
    ''' https://github.com/mkocabas/VIBE/blob/851f779407445b75cd1926402f61c931568c6947/lib/utils/demo_utils.py#L242
    Convert predicted camera from cropped image coordinates
    to original image coordinates
    :param cam (shape=(B, 3,)): weak perspective camera in cropped img coordinates
    :param bbox (shape=(B, 4,)): bbox coordinates (c_x, c_y, h)
    :param img_width (?, 1): original image width
    :param img_height (?, 1): original image height
    :return:
    '''
    cx, cy, w, h = bbox[:,0], bbox[:,1], bbox[:,2], bbox[:,3]
    hw, hh = img_width / 2., img_height / 2.
    s = sx = cam[:,0] * (1. / (img_width / torch.max(w, h)))  # z << tz
    sy = cam[:,0] * (1. / (img_height / torch.max(h, w)))
    tx = ((cx - hw) / hw / sx) + cam[:,-2]
    ty = ((cy - hh) / hh / sy) + cam[:,-1]
    orig_cam = torch.stack([s, tx, ty]).T
    return orig_cam


def perspective_projection(points, rotation=None, translation=None,
                           focal_length=None, camera_center=None, imsz=None, norm=False,
                           orig_imsz=None):
    """
    https://github.com/mkocabas/VIBE/blob/851f779407445b75cd1926402f61c931568c6947/lib/models/spin.py#L325
    This function computes the perspective projection of a set of points.
    Input:
        points (bs, N, 3): 3D points
        rotation (bs, 3, 3): Camera rotation
        translation (bs, 3): Camera translation
        focal_length (?, 2) or scalar: Focal length
        camera_center (?, 2): Camera center
    """
    orig_shape = points.shape
    if len(points.shape) == 4:
        points = points.flatten(end_dim=1)
    if focal_length is None:
        if orig_imsz is not None:
            focal_length = float(
                (orig_imsz[0] ** 2 + orig_imsz[1] ** 2) ** 0.5)  # CLIFF
        else:
            focal_length = 5000
    if not (isinstance(focal_length, torch.Tensor) and len(focal_length.shape) == 2):
        focal_length = torch.tensor(
            [[focal_length, focal_length]], dtype=points.dtype, device=points.device)
    if imsz is None:
        assert orig_imsz is not None
        imsz = orig_imsz
    camera_center = imsz.flip(-1) / 2

    batch_size = points.shape[0]
    K = torch.zeros([batch_size, 3, 3], device=points.device)
    K[:,0,0] = focal_length[:, 0]
    K[:,1,1] = focal_length[:, 1]
    K[:,2,2] = 1.
    K[:,:-1, -1] = camera_center

    # Transform points
    if rotation is not None:
        points = torch.einsum('bij,bkj->bki', rotation, points)
    if translation is not None:
        points = points + translation.unsqueeze(1)

    # Apply perspective distortion
    projected_points = points / points[:,:,-1].unsqueeze(-1)

    # Apply camera intrinsics
    projected_points = torch.einsum('bij,bkj->bki', K, projected_points)

    if norm:
        projected_points = projected_points[..., : -1].clone() / camera_center - 1
    else:
        projected_points = projected_points[:, :, :-1]

    return projected_points.reshape(orig_shape[: -1] + (2,))


def smpl2coco(j22s, betas):
    """
    Args:
        smpls: offers betas
    """
    assert len(j22s.shape) == 4
    assert betas.shape[1] == 1
    B, N = j22s.shape[: 2]
    dtype = j22s.dtype
    device = j22s.device
    j22s = j22s.flatten(end_dim=1)
    # rel_j22s = j22s - j22s[:, 0: 1]
    poses = HybrIKJointsToRotmat()(j22s.detach().cpu().numpy())  # or smplify3d to get eyes
    poses = torch.from_numpy(poses).to(dtype=dtype, device=device)
    smpls = {}
    smpls['global_orient'] = poses[:, : 1]
    smpls['body_pose'] = torch.cat([poses[:, 1:], torch.eye(
        3, dtype=dtype, device=device)[None, None].repeat(B * N, 2, 1, 1)], dim=1)
    smpls['betas'] = betas.repeat(1, N, 1).flatten(end_dim=1)
    smpl_outs = _smpl_spin.to(device=device)(pose2rot=False, **smpls)
    rec = smpl_outs.j45[:, : 22] - smpl_outs.j45[:, : 1] + j22s[:, : 1]
    rec_err = (rec - j22s).norm(dim=-1).mean()  #TODO: 0.05. HybrIK
    smpl_joints = smpl_outs.joints - smpl_outs.j45[:, : 1] + j22s[:, : 1]
    _coco2smpl = [0, 16, 15, 18, 17,
                  5, 2,
                  6, 3,
                  7, 4,
                  12, 9,
                  13, 10,
                  14, 11]
    # smpl_outs = _smplx_orig.to(device=device)(**smpls)
    # _coco2smpl = [SMPLX_OP25['smplx_idxs'][i] for i in _coco2smpl]
    j17s = smpl_joints[:, _coco2smpl].reshape(B, -1, 17, 3)
    return j17s


def raw2mot263(smpls=None, max_mot_len=196, glob_hmr=False):
    assert 'transl' in smpls  # coords in cam space
    dtype = smpls['body_pose'].dtype
    device = smpls['body_pose'].device
    B = smpls['body_pose'].shape[0]
    assert B == 1, 'Need to pad each len!'
    mot263s0_l, PA_ld, raw_mots_l = [], {}, []
    # smpl_outs = _smpl_spin.to(device)(pose2rot=smpls['body_pose'].shape[-1] == 23 * 3,
    #                                   **{k: v[didx] for k, v in smpls.items()})
    # j22s = smpl_outs['j45'][:, : 22]
    smpl_outs = _smplx_orig.to(device)(pose2rot=smpls['body_pose'].shape[-1] == 21 * 3,
                                       **{k: v.flatten(end_dim=1) for k, v in smpls.items()})  #TODO: check smplh
    j22s = smpl_outs['joints'].reshape(B, -1, 144, 3)[:, :, : 22]

    PA = {}
    ######### local -> global (z)?
    pse_j22s = j22s.clone()
    # if pse_j22s[..., 1].min() < -0.4:
    if not glob_hmr:
        pse_j22s[..., 1] = -pse_j22s[..., 1].clone()
    else:
        # amass
        print(f"{'>>>' * 10} Debugging AMASS!")
        swap_yz = torch.tensor([[1.0, 0.0, 0.0],
                                [0.0, 0.0, 1.0],
                                [0.0, 1.0, 0.0]], dtype=dtype, device=device)
        pse_j22s = pse_j22s @ swap_yz.T
    pse_j22s[..., 0] *= -1  # important to face z+! idk
    if True:
        PA['floor_y'] = floor_y = pse_j22s[..., 1].flatten(start_dim=1).min(-1)[0]
        pse_j22s[..., 1] = pse_j22s[..., 1].clone() - floor_y[:, None, None]  # ry, 1st on ground
    if False:
        PA['z_mean'] = z_mean = pse_j22s[..., -1].mean((1, 2))
        pse_j22s[..., -1] = pse_j22s[..., -1].clone() - z_mean[:, None, None] + 1  # stats. #TODO: traj scale

    #########

    for didx in range(B):
        mot263s0_t, PA_t = process_file(
            # torch.cat([pse_j22s[didx], pse_j22s[didx, -1:].repeat(max_mot_len +
            #           1 - pse_j22s.shape[1], 1, 1)]).detach().cpu().numpy(),
            pse_j22s[didx].detach().cpu().numpy(),
            returnPA=True)  # N + 1 -> N. #TODO: reduce ang err
        mot263s0_l.append(mot263s0_t[None])
        for k, v in PA_t.items():
            if didx == 0:
                PA_ld[k] = [v[None]]
            else:
                PA_ld[k].append(v[None])
    mot263s0 = torch.from_numpy(np.concatenate(
        mot263s0_l)).to(dtype=dtype, device=device)
    PA.update({k: torch.from_numpy(np.concatenate(v)).to(dtype=dtype, device=device)
               for k, v in PA_ld.items()})
    mot263s0 = (mot263s0 - mot263_mean) / mot263_std  # (mot263_std + 1e-9)
    mot263s0 = torch.cat([mot263s0, torch.zeros(B, max_mot_len - mot263s0.shape[1], mot263s0.shape[2],
                                                dtype=dtype, device=device)], dim=1)
    return mot263s0, PA, j22s


def front_mots2raw(front_mots, PA):
    dtype = front_mots.dtype
    device = front_mots.device
    B = front_mots.shape[0]
    mots = qrot(qinv(PA['R_quat'])[:, None, None].expand(front_mots.shape[: -1] + (4,)),
                front_mots)
    mots = mots + PA['t'][:, None, None]
    raw_mots_l = []
    for didx in range(B):
        raw_mots_t = uniform_skeleton(mots[didx].detach().cpu().numpy(),
                                      PA['offsets'][didx])
        raw_mots_l.append(raw_mots_t[None])
    raw_mots = torch.from_numpy(np.concatenate(
        raw_mots_l)).to(dtype=dtype, device=device)
    
    # local -> global
    raw_mots = raw_mots.clone()
    if 'z_mean' in PA:
        raw_mots[..., -1] = raw_mots[..., -1].clone() - 1 + PA['z_mean'][:, None, None]
    raw_mots[..., 1] = raw_mots[..., 1].clone() + PA['floor_y'][:, None, None]
    # if raw_mots[..., 1].max() > 0.4:
    raw_mots[..., 1] = -raw_mots[..., 1].clone()
    raw_mots[..., 0] = -raw_mots[..., 0].clone()

    return raw_mots


def j3d2kpmap(j3ds, betas, hw, orig_imsz=None, screen_bboxes=None):
    dtype = j3ds.dtype
    device = j3ds.device
    if orig_imsz is None:
        orig_imsz = hw
    if screen_bboxes is None:
        screen_bboxes = torch.cat(
            [hw.flip(0) / 2, hw.flip(0)]).to(dtype=dtype, device=device)[None]
    B = j3ds.shape[0]
    j17s = smpl2coco(j3ds, betas)
    j2ds = perspective_projection(j17s, orig_imsz=orig_imsz)
    j2ds = (j2ds - screen_bboxes[:, None, : 2]) / \
        screen_bboxes[:, None, 2:] * hw.flip(0) + hw.flip(0) / 2
    kpmaps_l = []
    for didx in range(B):
        kpmaps_t = get_kpmaps(j2ds[didx].detach(), *hw.tolist())
        kpmaps_l.append(kpmaps_t[None])
    kpmaps = torch.cat(kpmaps_l).to(dtype=dtype, device=device)
    return kpmaps


# import torch
# import numpy as np
from typing import Optional, Dict, List, Tuple


def compute_similarity_transform(S1: torch.Tensor, S2: torch.Tensor) -> torch.Tensor:
    """
    Computes a similarity transform (sR, t) in a batched way that takes
    a set of 3D points S1 (B, N, 3) closest to a set of 3D points S2 (B, N, 3),
    where R is a 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    Args:
        S1 (torch.Tensor): First set of points of shape (B, N, 3).
        S2 (torch.Tensor): Second set of points of shape (B, N, 3).
    Returns:
        (torch.Tensor): The first set of points after applying the similarity transformation.
    """

    batch_size = S1.shape[0]
    S1 = S1.permute(0, 2, 1)
    S2 = S2.permute(0, 2, 1)
    # 1. Remove mean.
    mu1 = S1.mean(dim=2, keepdim=True)
    mu2 = S2.mean(dim=2, keepdim=True)
    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = (X1**2).sum(dim=(1,2))

    # 3. The outer product of X1 and X2.
    K = torch.matmul(X1, X2.permute(0, 2, 1))

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are singular vectors of K.
    U, s, V = torch.svd(K)
    Vh = V.permute(0, 2, 1)

    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = torch.eye(U.shape[1]).unsqueeze(0).repeat(batch_size, 1, 1)
    Z[:, -1, -1] *= torch.sign(torch.linalg.det(torch.matmul(U, Vh)))

    # Construct R.
    R = torch.matmul(torch.matmul(V, Z), U.permute(0, 2, 1))

    # 5. Recover scale.
    trace = torch.matmul(R, K).diagonal(offset=0, dim1=-1, dim2=-2).sum(dim=-1)
    scale = (trace / var1).unsqueeze(dim=-1).unsqueeze(dim=-1)

    # 6. Recover translation.
    t = mu2 - scale*torch.matmul(R, mu1)

    # 7. Error:
    S1_hat = scale*torch.matmul(R, S1) + t

    return S1_hat.permute(0, 2, 1)

def reconstruction_error(S1, S2) -> np.array:
    """
    Computes the mean Euclidean distance of 2 set of points S1, S2 after performing Procrustes alignment.
    Args:
        S1 (torch.Tensor): First set of points of shape (B, N, 3).
        S2 (torch.Tensor): Second set of points of shape (B, N, 3).
    Returns:
        (np.array): Reconstruction error.
    """
    S1_hat = compute_similarity_transform(S1, S2)
    re = torch.sqrt( ((S1_hat - S2)** 2).sum(dim=-1)).mean(dim=-1)
    return re.cpu().numpy()
   