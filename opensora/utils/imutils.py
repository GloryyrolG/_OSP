### https://github.com/nkolot/SPIN/blob/master/utils/imutils.py
import numpy as np


def get_transform(center, scale, res, rot=0):
    """Generate transformation matrix."""
    if type(scale) == float:
        scale = np.array([scale, scale])
    h = 200 * scale
    t = np.zeros((3, 3))
    t[0, 0] = float(res[1]) / h[1]
    t[1, 1] = float(res[0]) / h[0]
    t[0, 2] = res[1] * (-float(center[0]) / h[1] + .5)
    t[1, 2] = res[0] * (-float(center[1]) / h[0] + .5)
    t[2, 2] = 1
    if not rot == 0:
        rot = -rot # To match direction of rotation from cropping
        rot_mat = np.zeros((3,3))
        rot_rad = rot * np.pi / 180
        sn,cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat[0,:2] = [cs, -sn]
        rot_mat[1,:2] = [sn, cs]
        rot_mat[2,2] = 1
        # Need to rotate around center
        t_mat = np.eye(3)
        t_mat[0,2] = -res[1]/2
        t_mat[1,2] = -res[0]/2
        t_inv = t_mat.copy()
        t_inv[:2,2] *= -1
        t = np.dot(t_inv,np.dot(rot_mat,np.dot(t_mat,t)))
    return t


def transform(pt, t, invert=0):
    """Transform pixel location to different reference."""
    if invert:
        t = np.linalg.inv(t)
    new_pt = np.stack([pt[..., 0]-1, pt[..., 1]-1, np.ones(pt.shape[: -1])],
                      axis=-1)
    new_pt = np.dot(new_pt, t.T)
    return new_pt[..., :2].astype(int)+1
