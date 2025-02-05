import copy
import numpy as np
import torch
import random
import numbers
from torchvision.transforms import RandomCrop, RandomResizedCrop, Compose
from torchvision.transforms import functional as vF
from opensora.utils.classes_and_palettes import GOLIATH_CLASSES, GOLIATH_PALETTE
from third_parties.open_animateanyone.DWPose.dwpose_utils.util import draw_bodypose


def _is_tensor_video_clip(clip):
    if not torch.is_tensor(clip):
        raise TypeError("clip should be Tensor. Got %s" % type(clip))

    if not clip.ndimension() == 4:
        raise ValueError("clip should be 4D. Got %dD" % clip.dim())

    return True


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


def crop(clip, i, j, h, w):
    """
    Args:
        clip (torch.tensor): Video clip to be cropped. Size is (T, C, H, W)
    """
    if len(clip.size()) != 4:
        raise ValueError("clip should be a 4D tensor")
    return clip[..., i: i + h, j: j + w]


def resize(clip, target_size, interpolation_mode):
    if len(target_size) != 2:
        raise ValueError(f"target size should be tuple (height, width), instead got {target_size}")
    return torch.nn.functional.interpolate(clip, size=target_size, mode=interpolation_mode, align_corners=True, antialias=True)


def resize_scale(clip, target_size, interpolation_mode):
    if len(target_size) != 2:
        raise ValueError(f"target size should be tuple (height, width), instead got {target_size}")
    H, W = clip.size(-2), clip.size(-1)
    scale_ = target_size[0] / min(H, W)
    return torch.nn.functional.interpolate(clip, scale_factor=scale_, mode=interpolation_mode, align_corners=True, antialias=True)


def resized_crop(clip, i, j, h, w, size, interpolation_mode="bilinear"):
    """
    Do spatial cropping and resizing to the video clip
    Args:
        clip (torch.tensor): Video clip to be cropped. Size is (T, C, H, W)
        i (int): i in (i,j) i.e coordinates of the upper left corner.
        j (int): j in (i,j) i.e coordinates of the upper left corner.
        h (int): Height of the cropped region.
        w (int): Width of the cropped region.
        size (tuple(int, int)): height and width of resized clip
    Returns:
        clip (torch.tensor): Resized and cropped clip. Size is (T, C, H, W)
    """
    if not _is_tensor_video_clip(clip):
        raise ValueError("clip should be a 4D torch.tensor")
    clip = crop(clip, i, j, h, w)
    clip = resize(clip, size, interpolation_mode)
    return clip


def center_crop(clip, crop_size):
    if not _is_tensor_video_clip(clip):
        raise ValueError("clip should be a 4D torch.tensor")
    h, w = clip.size(-2), clip.size(-1)
    th, tw = crop_size
    if h < th or w < tw:
        raise ValueError("height and width must be no smaller than crop_size")

    i = int(round((h - th) / 2.0))
    j = int(round((w - tw) / 2.0))
    return crop(clip, i, j, th, tw)


def center_crop_using_short_edge(clip):
    if not _is_tensor_video_clip(clip):
        raise ValueError("clip should be a 4D torch.tensor")
    h, w = clip.size(-2), clip.size(-1)
    if h < w:
        th, tw = h, h
        i = 0
        j = int(round((w - tw) / 2.0))
    else:
        th, tw = w, w
        i = int(round((h - th) / 2.0))
        j = 0
    return crop(clip, i, j, th, tw)



def center_crop_th_tw(clip, th, tw, top_crop):
    if not _is_tensor_video_clip(clip):
        raise ValueError("clip should be a 4D torch.tensor")
    
    # import ipdb;ipdb.set_trace()
    h, w = clip.size(-2), clip.size(-1)
    tr = th / tw
    if h / w > tr:
        new_h = int(w * tr)
        new_w = w
    else:
        new_h = h
        new_w = int(h / tr)
    
    i = 0 if top_crop else int(round((h - new_h) / 2.0))
    j = int(round((w - new_w) / 2.0))
    return crop(clip, i, j, new_h, new_w)

def random_shift_crop(clip):
    '''
    Slide along the long edge, with the short edge as crop size
    '''
    if not _is_tensor_video_clip(clip):
        raise ValueError("clip should be a 4D torch.tensor")
    h, w = clip.size(-2), clip.size(-1)

    if h <= w:
        long_edge = w
        short_edge = h
    else:
        long_edge = h
        short_edge = w

    th, tw = short_edge, short_edge

    i = torch.randint(0, h - th + 1, size=(1,)).item()
    j = torch.randint(0, w - tw + 1, size=(1,)).item()
    return crop(clip, i, j, th, tw)


def to_tensor(clip):
    """
    Convert tensor data type from uint8 to float, divide value by 255.0 and
    permute the dimensions of clip tensor
    Args:
        clip (torch.tensor, dtype=torch.uint8): Size is (T, C, H, W)
    Return:
        clip (torch.tensor, dtype=torch.float): Size is (T, C, H, W)
    """
    _is_tensor_video_clip(clip)
    if not clip.dtype == torch.uint8:
        raise TypeError("clip tensor should have data type uint8. Got %s" % str(clip.dtype))
    # return clip.float().permute(3, 0, 1, 2) / 255.0
    return clip.float() / 255.0


def normalize(clip, mean, std, inplace=False):
    """
    Args:
        clip (torch.tensor): Video clip to be normalized. Size is (T, C, H, W)
        mean (tuple): pixel RGB mean. Size is (3)
        std (tuple): pixel standard deviation. Size is (3)
    Returns:
        normalized clip (torch.tensor): Size is (T, C, H, W)
    """
    if not _is_tensor_video_clip(clip):
        raise ValueError("clip should be a 4D torch.tensor")
    if not inplace:
        clip = clip.clone()
    mean = torch.as_tensor(mean, dtype=clip.dtype, device=clip.device)
    # print(mean)
    std = torch.as_tensor(std, dtype=clip.dtype, device=clip.device)
    clip.sub_(mean[:, None, None, None]).div_(std[:, None, None, None])
    return clip


def hflip(clip):
    """
    Args:
        clip (torch.tensor): Video clip to be normalized. Size is (T, C, H, W)
    Returns:
        flipped clip (torch.tensor): Size is (T, C, H, W)
    """
    if not _is_tensor_video_clip(clip):
        raise ValueError("clip should be a 4D torch.tensor")
    return clip.flip(-1)


class RandomCropVideo:
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor): Video clip to be cropped. Size is (T, C, H, W)
        Returns:
            torch.tensor: randomly cropped video clip.
                size is (T, C, OH, OW)
        """
        i, j, h, w = self.get_params(clip)
        return crop(clip, i, j, h, w)

    def get_params(self, clip):
        h, w = clip.shape[-2:]
        th, tw = self.size

        if h < th or w < tw:
            raise ValueError(f"Required crop size {(th, tw)} is larger than input image size {(h, w)}")

        if w == tw and h == th:
            return 0, 0, h, w

        i = torch.randint(0, h - th + 1, size=(1,)).item()
        j = torch.randint(0, w - tw + 1, size=(1,)).item()

        return i, j, th, tw

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size})"


class SpatialStrideCropVideo:
    def __init__(self, stride):
            self.stride = stride

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor): Video clip to be cropped. Size is (T, C, H, W)
        Returns:
            torch.tensor: cropped video clip by stride.
                size is (T, C, OH, OW)
        """
        i, j, h, w = self.get_params(clip)
        return crop(clip, i, j, h, w)

    def get_params(self, clip):
        h, w = clip.shape[-2:]

        th, tw = h // self.stride * self.stride, w // self.stride * self.stride

        return 0, 0, th, tw  # from top-left

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size})"

class LongSideResizeVideo:
    '''
    First use the long side,
    then resize to the specified size
    '''

    def __init__(
            self,
            size,
            skip_low_resolution=False, 
            interpolation_mode="bilinear",
    ):
        self.size = size
        self.skip_low_resolution = skip_low_resolution
        self.interpolation_mode = interpolation_mode

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor): Video clip to be cropped. Size is (T, C, H, W)
        Returns:
            torch.tensor: scale resized video clip.
                size is (T, C, 512, *) or (T, C, *, 512)
        """
        _, _, h, w = clip.shape
        if self.skip_low_resolution and max(h, w) <= self.size:
            return clip
        if h > w:
            w = int(w * self.size / h)
            h = self.size
        else:
            h = int(h * self.size / w)
            w = self.size
        resize_clip = resize(clip, target_size=(h, w),
                                         interpolation_mode=self.interpolation_mode)
        return resize_clip

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size}, interpolation_mode={self.interpolation_mode}"

class CenterCropResizeVideo:
    '''
    First use the short side for cropping length,
    center crop video, then resize to the specified size
    '''

    def __init__(
            self,
            size,
            top_crop=False, 
            interpolation_mode="bilinear",
    ):
        if len(size) != 2:
            raise ValueError(f"size should be tuple (height, width), instead got {size}")
        self.size = size
        self.top_crop = top_crop
        self.interpolation_mode = interpolation_mode

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor): Video clip to be cropped. Size is (T, C, H, W)
        Returns:
            torch.tensor: scale resized / center cropped video clip.
                size is (T, C, crop_size, crop_size)
        """
        # clip_center_crop = center_crop_using_short_edge(clip)
        clip_center_crop = center_crop_th_tw(clip, self.size[0], self.size[1], top_crop=self.top_crop)
        # import ipdb;ipdb.set_trace()
        clip_center_crop_resize = resize(clip_center_crop, target_size=self.size,
                                         interpolation_mode=self.interpolation_mode)
        return clip_center_crop_resize

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size}, interpolation_mode={self.interpolation_mode}"


def bbox_crop(clip, out_ratio, kp2ds=None, bboxes=None, rescale=1.3):
    assert isinstance(clip, torch.Tensor)
    clip = clip.clone()
    out_ratio = torch.as_tensor(out_ratio)
    T, C = clip.shape[:2]

    confs = kp2ds[..., :, 2]  # all frames  #TODO: bbox
    right, top = kp2ds[confs == 1][:, : 2].max(0)[0]
    left, bottom = kp2ds[confs == 1][:, : 2].min(0)[0]
    # top, bottom = clip.shape[2] - top, clip.shape[2] - bottom  # orig shape

    center = torch.tensor([(right + left) / 2, (top + bottom) / 2])
    hw = torch.tensor([abs(bottom - top), right - left])
    out_hw = rescale * torch.cat([hw, hw.flip(0) / out_ratio.flip(0) * out_ratio]).reshape(2, 2).max(0)[0]
    top, left = center.flip(0) - out_hw / 2
    
    # for t in range(clip.shape[0]):  #TODO: too few kps
    clip_crop = vF.crop(clip.flatten(end_dim=1), int(top), int(left), int(out_hw[0]), int(out_hw[1]))
    clip_crop = clip_crop.reshape(T, C, *clip_crop.shape[-2:])
    kp2ds_crop = kp2ds.clone()
    kp2ds_crop[..., :2] -= torch.tensor([left, top])
    kp2ds_crop[kp2ds[..., 2] == 0] = 0
    return clip_crop, kp2ds_crop


class BboxCropResizeVideo(CenterCropResizeVideo):
    '''
    First use the short side for cropping length,
    center crop video, then resize to the specified size
    '''

    def __init__(
            self,
            size,
            top_crop=False, 
            interpolation_mode="bilinear",
    ):
        super(BboxCropResizeVideo, self).__init__(size, top_crop=top_crop,
                                                  interpolation_mode=interpolation_mode)
        self.kwargs = True

    def __call__(self, clip, kwargs=None):
        """
        Args:
            clip (torch.tensor): Video clip to be cropped. Size is (T, C, H, W)
        Returns:
            torch.tensor: scale resized / center cropped video clip.
                size is (T, C, crop_size, crop_size)
        """
        # clip_center_crop = center_crop_using_short_edge(clip)
        # clip_center_crop = center_crop_th_tw(clip, self.size[0], self.size[1], top_crop=self.top_crop)
        clip_crop, kp2ds_crop = bbox_crop(clip, self.size, kp2ds=kwargs['kp2ds'])
        # import ipdb;ipdb.set_trace()
        clip_crop_resize = resize(clip_crop, target_size=self.size,
                                         interpolation_mode=self.interpolation_mode)
        kp2ds_crop[..., : 2] = kp2ds_crop[..., : 2] / clip_crop.shape[-1] * self.size[-1]
        return clip_crop_resize, {'kp2ds': kp2ds_crop}


class UCFCenterCropVideo:
    '''
    First scale to the specified size in equal proportion to the short edge,
    then center cropping
    '''

    def __init__(
            self,
            size,
            interpolation_mode="bilinear",
    ):
        if isinstance(size, tuple):
            if len(size) != 2:
                raise ValueError(f"size should be tuple (height, width), instead got {size}")
            self.size = size
        else:
            self.size = (size, size)

        self.interpolation_mode = interpolation_mode

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor): Video clip to be cropped. Size is (T, C, H, W)
        Returns:
            torch.tensor: scale resized / center cropped video clip.
                size is (T, C, crop_size, crop_size)
        """
        clip_resize = resize_scale(clip=clip, target_size=self.size, interpolation_mode=self.interpolation_mode)
        clip_center_crop = center_crop(clip_resize, self.size)
        return clip_center_crop

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size}, interpolation_mode={self.interpolation_mode}"


class KineticsRandomCropResizeVideo:
    '''
    Slide along the long edge, with the short edge as crop size. And resie to the desired size.
    '''

    def __init__(
            self,
            size,
            interpolation_mode="bilinear",
    ):
        if isinstance(size, tuple):
            if len(size) != 2:
                raise ValueError(f"size should be tuple (height, width), instead got {size}")
            self.size = size
        else:
            self.size = (size, size)

        self.interpolation_mode = interpolation_mode

    def __call__(self, clip):
        clip_random_crop = random_shift_crop(clip)
        clip_resize = resize(clip_random_crop, self.size, self.interpolation_mode)
        return clip_resize


class CenterCropVideo:
    def __init__(
            self,
            size,
            interpolation_mode="bilinear",
    ):
        if isinstance(size, tuple):
            if len(size) != 2:
                raise ValueError(f"size should be tuple (height, width), instead got {size}")
            self.size = size
        else:
            self.size = (size, size)

        self.interpolation_mode = interpolation_mode

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor): Video clip to be cropped. Size is (T, C, H, W)
        Returns:
            torch.tensor: center cropped video clip.
                size is (T, C, crop_size, crop_size)
        """
        clip_center_crop = center_crop(clip, self.size)
        return clip_center_crop

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size}, interpolation_mode={self.interpolation_mode}"


class NormalizeVideo:
    """
    Normalize the video clip by mean subtraction and division by standard deviation
    Args:
        mean (3-tuple): pixel RGB mean
        std (3-tuple): pixel RGB standard deviation
        inplace (boolean): whether do in-place normalization
    """

    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor): video clip must be normalized. Size is (C, T, H, W)
        """
        return normalize(clip, self.mean, self.std, self.inplace)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std}, inplace={self.inplace})"


class ToTensorVideo:
    """
    Convert tensor data type from uint8 to float, divide value by 255.0 and
    permute the dimensions of clip tensor
    """

    def __init__(self):
        pass

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor, dtype=torch.uint8): Size is (T, C, H, W)
        Return:
            clip (torch.tensor, dtype=torch.float): Size is (T, C, H, W)
        """
        return to_tensor(clip)

    def __repr__(self) -> str:
        return self.__class__.__name__


class RandomHorizontalFlipVideo:
    """
    Flip the video clip along the horizontal direction with a given probability
    Args:
        p (float): probability of the clip being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor): Size is (T, C, H, W)
        Return:
            clip (torch.tensor): Size is (T, C, H, W)
        """
        if random.random() < self.p:
            clip = hflip(clip)
        return clip

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"


#  ------------------------------------------------------------
#  ---------------------  Sampling  ---------------------------
#  ------------------------------------------------------------
class TemporalRandomCrop(object):
    """Temporally crop the given frame indices at a random location.

    Args:
        size (int): Desired length of frames will be seen in the model.
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, total_frames):
        rand_end = max(0, total_frames - self.size - 1)
        begin_index = random.randint(0, rand_end)
        end_index = min(begin_index + self.size, total_frames)
        return begin_index, end_index

class DynamicSampleDuration(object):
    """Temporally crop the given frame indices at a random location.

    Args:
        size (int): Desired length of frames will be seen in the model.
    """

    def __init__(self, t_stride, extra_1):
        self.t_stride = t_stride
        self.extra_1 = extra_1

    def __call__(self, t, h, w):
        if self.extra_1:
            t = t - 1
        truncate_t_list = list(range(t+1))[t//2:][::self.t_stride]  # need half at least
        truncate_t = random.choice(truncate_t_list)
        if self.extra_1:
            truncate_t = truncate_t + 1
        return 0, truncate_t


# joints_name = ('nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',    # 4
#                'left_shoulder', 'right_shoulder',                           # 6
#                'left_elbow', 'right_elbow',                                 # 8
#                'left_wrist', 'right_wrist',                                 # 10
#                'left_hip', 'right_hip',                                     # 12
#                'left_knee', 'right_knee',                                   # 14
#                'left_ankle', 'right_ankle')                                 # 16

# OP:
# Nose, Neck, RShoulder, RElbow, RWrist, LShoulder, LElbow, LWrist, RHip,  # 8
# RKnee, RAnkle, LHip, LKnee, LAnkle, REye, LEye, REar, LEar  # 17


def get_kpmaps(kp17_2ds, h, w):
    T = kp17_2ds.shape[0]
    kpmaps = np.zeros((T, h, w, 3), dtype=np.uint8)
    kpmap17_to_18 = [0, 15, 14, 17, 16,
                     5, 2,
                     6, 3,
                     7, 4,
                     11, 8,
                     12, 9,
                     13, 10]  # COCO to OP
    kpmap17_to_18 = np.array(kpmap17_to_18)

    kp18_2ds = np.zeros((T, 18, 2), dtype=np.float32)
    kp18_2ds[:, kpmap17_to_18] = kp17_2ds[..., : 2]
    kp18_2ds[:, 1] = (kp18_2ds[:, 5] + kp18_2ds[:, 2]) / 2  # neck
    norm_kp18_2ds = kp18_2ds / np.array([w, h])

    for t in range(T):
        subset = np.arange(18)[None]  # single-person
        miss_kps = np.nonzero(kp17_2ds[t, :, -1].numpy() == 0)[0]
        subset[:, kpmap17_to_18[miss_kps]] = -1

        kpmaps[t] = draw_bodypose(kpmaps[t], norm_kp18_2ds[t], subset)
    return torch.from_numpy(kpmaps).permute(0, 3, 1, 2)


def get_dps(pred_sem_seg):
    classes = GOLIATH_CLASSES
    palette = GOLIATH_PALETTE
    num_classes = len(classes)
    sem_seg = pred_sem_seg
    ids = np.unique(sem_seg)[::-1]
    legal_indices = ids < num_classes
    ids = ids[legal_indices]
    labels = np.array(ids, dtype=np.int64)

    colors = [palette[label] for label in labels]

    mask = np.zeros((*pred_sem_seg.shape, 3), dtype=np.uint8)
    for label, color in zip(labels, colors):
        mask[sem_seg == label, :] = color
    return mask


def get_dep_normal(depth_map):
    mask = np.ones(depth_map.shape, dtype='bool')
    depth_map[~mask] = np.nan
    depth_foreground = depth_map[mask]  ## value in range [0, 1]
    processed_depth = np.full((mask.shape[0], mask.shape[1], 3), 100, dtype=np.uint8)

    if len(depth_foreground) > 0:
        min_val, max_val = np.min(depth_foreground), np.max(depth_foreground)
        depth_normalized_foreground = 1 - (
            (depth_foreground - min_val) / (max_val - min_val)
        )  ## for visualization, foreground is 1 (white), background is 0 (black)
        depth_normalized_foreground = (depth_normalized_foreground * 255.0).astype(
            np.uint8
        )

        depth_colored_foreground = cv2.applyColorMap(
            depth_normalized_foreground, cv2.COLORMAP_INFERNO
        )
        depth_colored_foreground = depth_colored_foreground.reshape(-1, 3)
        processed_depth[mask] = depth_colored_foreground

    ##---------get surface normal from depth map---------------
    depth_normalized = np.full((mask.shape[0], mask.shape[1]), np.inf)
    depth_normalized[mask > 0] = 1 - (
        (depth_foreground - min_val) / (max_val - min_val)
    )

    kernel_size = 7
    grad_x = cv2.Sobel(
        depth_normalized.astype(np.float32),
        cv2.CV_32F,
        1,
        0,
        ksize=kernel_size,
    )
    grad_y = cv2.Sobel(
        depth_normalized.astype(np.float32),
        cv2.CV_32F,
        0,
        1,
        ksize=kernel_size,
    )
    z = np.full(grad_x.shape, -1)
    normals = np.dstack((-grad_x, -grad_y, z))

    # Normalize the normals
    normals_mag = np.linalg.norm(normals, axis=2, keepdims=True)

    ## background pixels are nan.
    with np.errstate(divide="ignore", invalid="ignore"):
        normals_normalized = normals / (
            normals_mag + 1e-5
        )  # Add a small epsilon to avoid division by zero

    # Convert normals to a 0-255 scale for visualization
    normals_normalized = np.nan_to_num(
        normals_normalized, nan=-1, posinf=-1, neginf=-1
    )  ## visualize background (nan) as black
    normal_from_depth = ((normals_normalized + 1) / 2 * 255).astype(np.uint8)

    ## RGB to BGR for cv2
    normal_from_depth = normal_from_depth[:, :, ::-1]

    # vis_image = np.concatenate([image, processed_depth, normal_from_depth], axis=1)
    # cv2.imwrite(output_path, vis_image)
    return processed_depth, normal_from_depth


def _msra_generate_target(self, cfg, joints_3d, joints_3d_visible, sigma):
    """Generate the target heatmap via "MSRA" approach.

    Args:
        cfg (dict): data config
        joints_3d: np.ndarray ([num_joints, 3])
        joints_3d_visible: np.ndarray ([num_joints, 3])
        sigma: Sigma of heatmap gaussian
    Returns:
        tuple: A tuple containing targets.

        - target: Target heatmaps.
        - target_weight: (1: visible, 0: invisible)
    """
    num_joints = len(joints_3d)
    image_size = cfg['image_size']
    W, H = cfg['heatmap_size']
    joint_weights = cfg['joint_weights']
    use_different_joint_weights = cfg['use_different_joint_weights']
    assert not use_different_joint_weights

    target_weight = np.zeros((num_joints, 1), dtype=np.float32)
    target = np.zeros((num_joints, H, W), dtype=np.float32)

    # 3-sigma rule
    tmp_size = sigma * 3

    if True:  # self.unbiased_encoding:
        for joint_id in range(num_joints):
            target_weight[joint_id] = joints_3d_visible[joint_id, 0]

            feat_stride = image_size / [W, H]
            mu_x = joints_3d[joint_id][0] / feat_stride[0]
            mu_y = joints_3d[joint_id][1] / feat_stride[1]
            # Check that any part of the gaussian is in-bounds
            ul = [mu_x - tmp_size, mu_y - tmp_size]
            br = [mu_x + tmp_size + 1, mu_y + tmp_size + 1]
            if ul[0] >= W or ul[1] >= H or br[0] < 0 or br[1] < 0:
                target_weight[joint_id] = 0

            if target_weight[joint_id] == 0:
                continue

            x = np.arange(0, W, 1, np.float32)
            y = np.arange(0, H, 1, np.float32)
            y = y[:, None]

            if target_weight[joint_id] > 0.5:
                target[joint_id] = np.exp(-((x - mu_x)**2 +
                                            (y - mu_y)**2) /
                                          (2 * sigma**2))


class ComposeKwargs(Compose):
    def __init__(self, transforms):
        super(ComposeKwargs, self).__init__(transforms)
    
    def __call__(self, img, kwargs=None):
        returns = copy.deepcopy(kwargs)
        for t in self.transforms:
            if getattr(t, 'kwargs', False):  # data aug
                img, returns_t = t(img, kwargs=kwargs)
                if returns is None:
                    returns = {}
                returns.update(returns_t)
            else:
                img = t(img)
        return img, returns


if __name__ == '__main__':
    from torchvision import transforms
    import torchvision.io as io
    import numpy as np
    from torchvision.utils import save_image
    import os

    vframes, aframes, info = io.read_video(
        filename='./v_Archery_g01_c03.avi',
        pts_unit='sec',
        output_format='TCHW'
    )

    trans = transforms.Compose([
        ToTensorVideo(),
        RandomHorizontalFlipVideo(),
        UCFCenterCropVideo(512),
        # NormalizeVideo(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])

    target_video_len = 32
    frame_interval = 1
    total_frames = len(vframes)
    print(total_frames)

    temporal_sample = TemporalRandomCrop(target_video_len * frame_interval)

    # Sampling video frames
    start_frame_ind, end_frame_ind = temporal_sample(total_frames)
    # print(start_frame_ind)
    # print(end_frame_ind)
    assert end_frame_ind - start_frame_ind >= target_video_len
    frame_indice = np.linspace(start_frame_ind, end_frame_ind - 1, target_video_len, dtype=int)
    print(frame_indice)

    select_vframes = vframes[frame_indice]
    print(select_vframes.shape)
    print(select_vframes.dtype)

    select_vframes_trans = trans(select_vframes)
    print(select_vframes_trans.shape)
    print(select_vframes_trans.dtype)

    select_vframes_trans_int = ((select_vframes_trans * 0.5 + 0.5) * 255).to(dtype=torch.uint8)
    print(select_vframes_trans_int.dtype)
    print(select_vframes_trans_int.permute(0, 2, 3, 1).shape)

    io.write_video('./test.avi', select_vframes_trans_int.permute(0, 2, 3, 1), fps=8)

    for i in range(target_video_len):
        save_image(select_vframes_trans[i], os.path.join('./test000', '%04d.png' % i), normalize=True,
                   value_range=(-1, 1))
