from argparse import Namespace
import copy
import os
import numpy as np
from torch import nn
import torch
from einops import rearrange, repeat
from typing import Any, Dict, Optional, Tuple
from torch.nn import functional as F
from diffusers.models.transformer_2d import Transformer2DModelOutput
from diffusers.utils import is_torch_version, deprecate
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import AdaLayerNormSingle
from diffusers.models.embeddings import PixArtAlphaTextProjection
from opensora.models.diffusion.opensora.modules import (
    OverlapPatchEmbed3D, OverlapPatchEmbed2D, PatchEmbed2D, BasicTransformerBlock,
    _UnpatchifyOutTime, ModuleList, ModalSpatialAttnBlk, EmbLayers, zero_module)
from opensora.models.decs.aggregation_network import AggregationNetwork, AggTsfm
from opensora.models.encs.PoseGuider import PoseGuider
from opensora.utils.human_utils import SMPL_KS, raw2mot263, front_mots2raw, j3d2kpmap
from opensora.utils.utils import to_2tuple
try:
    import torch_npu
    from opensora.npu_config import npu_config
    from opensora.acceleration.parallel_states import get_sequence_parallel_state, hccl_info
except:
    torch_npu = None
    npu_config = None
    from opensora.utils.parallel_states import get_sequence_parallel_state, nccl_info


def ins_with_str(cmd, num=1, **kwargs):
    if num > 1:
        print(f"{'>>>' * 10} {(cmd[0] if type(cmd) == list else cmd).split('(')[0]} use CopyTrain!")
        return CopyTrain(cmd, num=num, **kwargs)
    else:
        assert num == 1
        return eval(cmd)


def copy_zeroout(m):  # no need cuz no add
    if not isinstance(m, nn.ModuleList):  # pos_embed_masked_video
        m = ModuleList([m])
    lst_m = m[-1]
    if isinstance(lst_m, nn.Linear):  # proj_out, pos_embed_masked_video
        zeroout = nn.Linear(lst_m.out_features, lst_m.out_features, bias=False)
    elif isinstance(lst_m, PatchEmbed2D):  # pos_embed
        zeroout = nn.Linear(lst_m.proj.out_channels, lst_m.proj.out_channels, bias=False)
    elif isinstance(lst_m, BasicTransformerBlock):
        zeroout = nn.Linear(lst_m.ff.net[-1].out_features, lst_m.ff.net[-1].out_features, bias=False)
    else:
        raise RuntimeError
    m.append(zero_module(zeroout))
    return m


class CopyTrain(nn.ModuleList):
    """ Applicable to multi-head output and multi-condition input (T2I-Adapter).
        Instead of copy whole net (MulTask) for saving GPU memory.
        So far supports PatchEmbed2D, ModuleList, Linear, BasicTransformerBlock
    """

    def __init__(self, inscmd, num=2, intype='dup', outtype='add', **kwargs):
        # Instantiate by name
        self.zeroout = kwargs.get('zeroout', outtype == 'add')
        if type(inscmd) == str:
            inscmd = [inscmd] * num
        ms = [eval(inscmd[0])] + [(eval(inscmd[i]) if not self.zeroout
                                   else copy_zeroout(eval(inscmd[i])))
                                  for i in range(1, num)]
        super(CopyTrain, self).__init__(ms)

        self.intype = intype  # for fixed call type
        self.outtype = outtype

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # train: torch load_state_dict.load
        if len(state_dict) < len(self.state_dict()):
            for k, v in {k: v for k, v in state_dict.items()}.items():
                for i in range(len(self)):
                    # k1 = prefix + (f"{i}.0." if i and not k[len(prefix)].isdigit() else f"{i}.") + k[len(prefix):]
                    # itself is not ModuleList
                    if self.zeroout and i and not k[len(prefix)].isdigit():
                        k1 = f"{i}.0."
                    else:
                        k1 = f"{i}."
                    k1 = prefix + k1 + k[len(prefix):]
                    state_dict[k1] = v  # inplace during topdown
                del state_dict[k]
        # else sample: diffusers _load_state_dict_into_model.load
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)
        return

    def split_inputs(self, x, args, B):
        x1 = []
        if type(x) in [tuple, list]:
            # to fill in gradient ckpting
            args = args + [-1] * (len(x) - len(args))
            for x_, args_ in zip(x, args):
                if args_ != -1 and isinstance(x_, torch.Tensor):
                    if type(args_) == list:
                        assert len(args_) == 2 and type(args[1]) == list
                        units = x_.chunk(sum(args_[1]), dim=args_[0])
                        cks = []
                        cumdims = 0
                        for dims in args_[1]:
                            if not dims:
                                cks.append(None)
                            else:
                                cks.append(torch.cat(units[cumdims: cumdims + dims], dim=args_[0]))
                            cumdims += dims
                        x1.append(cks)
                    else:
                        x1.append(x_.chunk(len(self), dim=args_))  # core tensors
                else:
                    x1.append([x_] * len(self))  # dup
            if len(x) == 0:
                x1 = [[]] * len(self)
            else:
                x1 = list(zip(*x1))  # col
        else:
            assert type(x) == dict, type(x)
            for k, v in x.items():
                if k in args:
                    if type(args[k]) == list:
                        assert len(args[k]) == 2 and type(
                            args[k][1]) == list and len(args[k][1]) == len(self)
                        units = v.chunk(sum(args[k][1]), dim=args[k][0])
                        cks = []
                        cumdims = 0
                        for dims in args[k][1]:
                            if not dims:
                                cks.append(None)
                            else:
                                cks.append(torch.cat(units[cumdims: cumdims + dims], dim=args[k][0]))
                            cumdims += dims
                        x1.append(zip([k] * len(self), cks))
                    else:
                        x1.append(zip([k] * len(self),
                                    v.chunk(len(self), dim=args[k])))
                else:
                    x1.append([(k, v)] * len(self))  # conds
            if len(x) == 0:  # e.g. Linear
                x1 = [{}] * len(self)
            else:
                x1 = [dict(items) for items in zip(*x1)]
        return x1

    def forward(self, *args, **kwargs):
        """ Data interface.
        #TODO: func not target single-path """
        if kwargs.get('outtype', None) is None:
            outtype = self.outtype
        if True:  # self.intype != 'dup':
            # first. Accepts mul inputs
            if not len(kwargs):  # {'skips': 2} for grad ckpt
                args1 = self.split_inputs(
                    args, self.intype[0] + list(self.intype[1].values()), args[0].shape[0])
                kwargs1 = [kwargs] * len(self)
            else:
                args1 = self.split_inputs(
                    args, self.intype[0], args[0].shape[0])
                kwargs1 = self.split_inputs(
                    kwargs, self.intype[1], args[0].shape[0])
        # else:
        #     args1 = [args] * len(self)
        #     kwargs1 = [kwargs] * len(self)
        assert len(args1) == len(kwargs1) and len(args1) == len(self)
        # One-to-one correspondence
        outs = [m(*args_, **kwargs_) for args_, kwargs_,
                m in zip(args1, kwargs1, self)]  # "parallel"
        if type(outtype) == int:
            out = torch.cat(outs, dim=outtype)  # lst. Outputs only 1
        elif outtype == 'avg':
            out = sum(outs) / len(self)
        else:
            assert outtype == 'add'
            out = sum(outs)
        return out


# class Linear(nn.Linear):
#     def __init__(self, *args, **kwargs):
#         zeroout = kwargs.pop('zeroout', False)
#         super(Linear, self).__init__(*args, **kwargs)
#         if zeroout:
#             zero_module(self)


class OpenSoraT2V(ModelMixin, ConfigMixin):
    """
    A 2D Transformer model for image-like data.

    Parameters:
        num_attention_heads (`int`, *optional*, defaults to 16): The number of heads to use for multi-head attention.
        attention_head_dim (`int`, *optional*, defaults to 88): The number of channels in each head.
        in_channels (`int`, *optional*):
            The number of channels in the input and output (specify if the input is **continuous**).
        num_layers (`int`, *optional*, defaults to 1): The number of layers of Transformer blocks to use.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        cross_attention_dim (`int`, *optional*): The number of `encoder_hidden_states` dimensions to use.
        sample_size (`int`, *optional*): The width of the latent images (specify if the input is **discrete**).
            This is fixed during training since it is used to learn a number of position embeddings.
        num_vector_embeds (`int`, *optional*):
            The number of classes of the vector embeddings of the latent pixels (specify if the input is **discrete**).
            Includes the class for the masked latent pixel.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to use in feed-forward.
        num_embeds_ada_norm ( `int`, *optional*):
            The number of diffusion steps used during training. Pass if at least one of the norm_layers is
            `AdaLayerNorm`. This is fixed during training since it is used to learn a number of embeddings that are
            added to the hidden states.

            During inference, you can denoise for up to but not more steps than `num_embeds_ada_norm`.
        attention_bias (`bool`, *optional*):
            Configure if the `TransformerBlocks` attention should contain a bias parameter.
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        # args=None,  # TypeError: Object of type Namespace is not JSON serializable
        num_attention_heads: int = 16,
        attention_head_dim: int = 88,
        in_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        num_layers: int = 1,
        dropout: float = 0.0,
        norm_num_groups: int = 32,
        cross_attention_dim: Optional[int] = None,
        attention_bias: bool = False,
        sample_size: Optional[int] = None,
        sample_size_t: Optional[int] = None,
        num_vector_embeds: Optional[int] = None,
        patch_size: Optional[int] = None,
        patch_size_t: Optional[int] = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        use_linear_projection: bool = False,
        only_cross_attention: bool = False,
        double_self_attention: bool = False,
        upcast_attention: bool = False,
        norm_type: str = "layer_norm",  # 'layer_norm', 'ada_norm', 'ada_norm_zero', 'ada_norm_single', 'ada_norm_continuous', 'layer_norm_i2vgen'
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        attention_type: str = "default",
        caption_channels: int = None,
        interpolation_scale_h: float = None,
        interpolation_scale_w: float = None,
        interpolation_scale_t: float = None,
        use_additional_conditions: Optional[bool] = None,
        attention_mode: str = 'xformers', 
        downsampler: str = None, 
        use_rope: bool = False,
        use_stable_fp32: bool = False,

        latent_pose=None,
        multitask=None,
        skips=None,
        mullev=None,
    ):
        super().__init__()

        # Validate inputs.
        if patch_size is not None:
            if norm_type not in ["ada_norm", "ada_norm_zero", "ada_norm_single"]:
                raise NotImplementedError(
                    f"Forward pass is not implemented when `patch_size` is not None and `norm_type` is '{norm_type}'."
                )
            elif norm_type in ["ada_norm", "ada_norm_zero"] and num_embeds_ada_norm is None:
                raise ValueError(
                    f"When using a `patch_size` and this `norm_type` ({norm_type}), `num_embeds_ada_norm` cannot be None."
                )

        # Set some common variables used across the board.
        self.latent_pose = latent_pose
        self.multitask = multitask
        self.nmodals = 1 if multitask not in ['local', 'fuse1', 'idol', 'emb'] else 2  # generative latent
        self.skips = skips
        self.mullev = mullev
        self.use_rope = use_rope
        self.use_linear_projection = use_linear_projection
        self.interpolation_scale_t = interpolation_scale_t
        self.interpolation_scale_h = interpolation_scale_h
        self.interpolation_scale_w = interpolation_scale_w
        self.downsampler = downsampler
        self.caption_channels = caption_channels
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        self.inner_dim = self.config.num_attention_heads * self.config.attention_head_dim
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.gradient_checkpointing = False
        self.config.hidden_size = self.inner_dim
        use_additional_conditions = False
        # if use_additional_conditions is None:
            # if norm_type == "ada_norm_single" and sample_size == 128:
            #     use_additional_conditions = True
            # else:
            # use_additional_conditions = False
        self.use_additional_conditions = use_additional_conditions

        # 1. Transformer2DModel can process both standard continuous images of shape `(batch_size, num_channels, width, height)` as well as quantized image embeddings of shape `(batch_size, num_image_vectors)`
        # Define whether input is continuous or discrete depending on configuration
        assert in_channels is not None and patch_size is not None

        if norm_type == "layer_norm" and num_embeds_ada_norm is not None:
            deprecation_message = (
                f"The configuration file of this model: {self.__class__} is outdated. `norm_type` is either not set or"
                " incorrectly set to `'layer_norm'`. Make sure to set `norm_type` to `'ada_norm'` in the config."
                " Please make sure to update the config accordingly as leaving `norm_type` might led to incorrect"
                " results in future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it"
                " would be very nice if you could open a Pull request for the `transformer/config.json` file"
            )
            deprecate("norm_type!=num_embeds_ada_norm", "1.0.0", deprecation_message, standard_warn=False)
            norm_type = "ada_norm"

        # 2. Initialize the right blocks.
        # Initialize the output blocks and other projection blocks when necessary.
        self._init_patched_inputs(norm_type=norm_type)

        self._create_latent_pose()

    def _init_patched_inputs(self, norm_type):
        assert self.config.sample_size_t is not None, "OpenSoraT2V over patched input must provide sample_size_t"
        assert self.config.sample_size is not None, "OpenSoraT2V over patched input must provide sample_size"
        #assert not (self.config.sample_size_t == 1 and self.config.patch_size_t == 2), "Image do not need patchfy in t-dim"

        self.num_frames = self.config.sample_size_t
        self.config.sample_size = to_2tuple(self.config.sample_size)
        self.height = self.config.sample_size[0]
        self.width = self.config.sample_size[1]
        self.patch_size_t = self.config.patch_size_t
        self.patch_size = self.config.patch_size
        interpolation_scale_t = ((self.config.sample_size_t - 1) // 16 + 1) if self.config.sample_size_t % 2 == 1 else self.config.sample_size_t / 16
        interpolation_scale_t = (
            self.config.interpolation_scale_t if self.config.interpolation_scale_t is not None else interpolation_scale_t
        )
        self.interpolation_scale = interpolation_scale = (
            self.config.interpolation_scale_h if self.config.interpolation_scale_h is not None else self.config.sample_size[0] / 30, 
            self.config.interpolation_scale_w if self.config.interpolation_scale_w is not None else self.config.sample_size[1] / 40, 
        )
        # if self.config.sample_size_t > 1:
        #     self.pos_embed = PatchEmbed3D(
        #         num_frames=self.config.sample_size_t,
        #         height=self.config.sample_size[0],
        #         width=self.config.sample_size[1],
        #         patch_size_t=self.config.patch_size_t,
        #         patch_size=self.config.patch_size,
        #         in_channels=self.in_channels,
        #         embed_dim=self.inner_dim,
        #         interpolation_scale=interpolation_scale, 
        #         interpolation_scale_t=interpolation_scale_t,
        #     )
        # else:
        if self.config.downsampler is not None and len(self.config.downsampler) == 9:
            self.pos_embed = OverlapPatchEmbed3D(
                num_frames=self.config.sample_size_t,
                height=self.config.sample_size[0],
                width=self.config.sample_size[1],
                patch_size_t=self.config.patch_size_t,
                patch_size=self.config.patch_size,
                in_channels=self.in_channels,
                embed_dim=self.inner_dim,
                interpolation_scale=interpolation_scale, 
                interpolation_scale_t=interpolation_scale_t,
                use_abs_pos=not self.config.use_rope, 
            )
        elif self.config.downsampler is not None and len(self.config.downsampler) == 7:
            self.pos_embed = OverlapPatchEmbed2D(
                num_frames=self.config.sample_size_t,
                height=self.config.sample_size[0],
                width=self.config.sample_size[1],
                patch_size_t=self.config.patch_size_t,
                patch_size=self.config.patch_size,
                in_channels=self.in_channels,
                embed_dim=self.inner_dim,
                interpolation_scale=interpolation_scale, 
                interpolation_scale_t=interpolation_scale_t,
                use_abs_pos=not self.config.use_rope, 
            )
        
        else:
            # self.pos_embed = PatchEmbed2D(
            #     num_frames=self.config.sample_size_t,
            #     height=self.config.sample_size[0],
            #     width=self.config.sample_size[1],
            #     patch_size_t=self.config.patch_size_t,
            #     patch_size=self.config.patch_size,
            #     in_channels=self.in_channels,
            #     embed_dim=self.inner_dim,
            #     interpolation_scale=interpolation_scale, 
            #     interpolation_scale_t=interpolation_scale_t,
            #     use_abs_pos=not self.config.use_rope,
            #     zeroout=True, 
            # )
            # S ((C|S) C S C S C) S
            inskwargs = {}
            if self.config.multitask in ['local', 'fuse1', 'idol']:
                inskwargs['num'] = self.nmodals
                inskwargs.update(intype=[
                    [1, -1],
                    {}
                ],
                    outtype=2)  # x, frame, return_img
            else:
                inskwargs['num'] = 1
            self.pos_embed = ins_with_str(f'PatchEmbed2D(\
                num_frames={self.config.sample_size_t},\
                height={self.config.sample_size[0]},\
                width={self.config.sample_size[1]},\
                patch_size_t={self.config.patch_size_t},\
                patch_size={self.config.patch_size},\
                in_channels={self.in_channels},\
                embed_dim={self.inner_dim},\
                interpolation_scale={interpolation_scale},\
                interpolation_scale_t={interpolation_scale_t},\
                use_abs_pos={not self.config.use_rope},\
            )', **inskwargs)
        
        interpolation_scale_thw = (interpolation_scale_t, *interpolation_scale)
                # BasicTransformerBlock(
                #     self.inner_dim,
                #     self.config.num_attention_heads,
                #     self.config.attention_head_dim,
                #     dropout=self.config.dropout,
                #     cross_attention_dim=self.config.cross_attention_dim,
                #     activation_fn=self.config.activation_fn,
                #     num_embeds_ada_norm=self.config.num_embeds_ada_norm,
                #     attention_bias=self.config.attention_bias,
                #     only_cross_attention=self.config.only_cross_attention,
                #     double_self_attention=self.config.double_self_attention,
                #     upcast_attention=self.config.upcast_attention,
                #     norm_type=norm_type,
                #     norm_elementwise_affine=self.config.norm_elementwise_affine,
                #     norm_eps=self.config.norm_eps,
                #     attention_type=self.config.attention_type,
                #     attention_mode=self.config.attention_mode, 
                #     downsampler=self.config.downsampler, 
                #     use_rope=self.config.use_rope, 
                #     interpolation_scale_thw=interpolation_scale_thw, 
                #     zeroout=True,
                # )
        bas_tsfm_blk_cmd = cmd = f'BasicTransformerBlock(\
                    {self.inner_dim},\
                    {self.config.num_attention_heads},\
                    {self.config.attention_head_dim},\
                    dropout={self.config.dropout},\
                    cross_attention_dim={self.config.cross_attention_dim},\
                    activation_fn="{self.config.activation_fn}",\
                    num_embeds_ada_norm={self.config.num_embeds_ada_norm},\
                    attention_bias={self.config.attention_bias},\
                    only_cross_attention={self.config.only_cross_attention},\
                    double_self_attention={self.config.double_self_attention},\
                    upcast_attention={self.config.upcast_attention},\
                    norm_type="{norm_type}",\
                    norm_elementwise_affine={self.config.norm_elementwise_affine},\
                    norm_eps={self.config.norm_eps},\
                    attention_type="{self.config.attention_type}",\
                    attention_mode="{self.config.attention_mode}",\
                    downsampler={self.config.downsampler},\
                    use_rope={self.config.use_rope},\
                    interpolation_scale_thw={interpolation_scale_thw},\
                    skips_in={{skips_in}},\
                )'
                    # ski={{ski}},\
        # Cannot be too few for diff mod pred, large loss@t=0, deviated, discriminative head
        self.nsepblks = nsepblks = 1 if self.nmodals > 1 and self.config.multitask != 'emb' else 0  # u-dit. 1
        # input(f"{'>>>' * 10} nsepblks={self.nsepblks}!")
        tsfm_blks = []
        for _ in range(self.config.num_layers - 2 * nsepblks):
            tsfm_blks.append(eval(cmd.format(skips_in=0)))  # , ski=False))
        self.transformer_blocks = nn.ModuleList(tsfm_blks)
        if self.config.skips == '1':
            self.skip_bidxs = [0]  # 1?
        else:
            self.skip_bidxs = []
        if self.config.mullev == 'true':  # OutBlk
            # assert nsepblks % 2 == 0
            self.lev_idxs = list(range(self.config.num_layers))[1: -self.nsepblks: 2]
        else:
            self.lev_idxs = []

        # InBlks
        # if self.config.skips in ['1', '1.0']:
        #     cmd1 = cmd.format(skips_in=0)  # , ski=True)
        # else:
        cmd1 = cmd.format(skips_in=0)  # , ski=False)
        # No diff@writer side
        for _ in range(nsepblks - 1):
            self.transformer_blocks.insert(
                _, ins_with_str(cmd1, **{**inskwargs, 'intype': [[2], {}], 'outtype': 2}))  # incls num=1 

        if nsepblks:
            if self.config.multitask in ['local', 'idol']:
                # cat along batch for sharing NN
                self.transformer_blocks.insert(
                    nsepblks - 1, ins_with_str(cmd1, **{**inskwargs, 'intype': [[2], {}], 'outtype': 0}))
            elif self.config.multitask == 'fuse1':
                if self.config.skips == '1' or self.config.mullev == 'true':
                    self.transformer_blocks.insert(
                        nsepblks - 1, ins_with_str(cmd1, **{**inskwargs, 'intype': [[2], {}], 'outtype': 2, 'zeroout': True}))
                    inskwargs['zeroout'] = False
                else:
                    # https://github.com/snap-research/HyperHuman/issues/4#issuecomment-1765615972
                    # Pre-act
                    self.transformer_blocks.insert(
                        nsepblks - 1, ins_with_str(cmd1, **{**inskwargs, 'intype': [[2], {}], 'outtype': 'add'}))  # TODO: align distributions
            else:
                self.transformer_blocks.insert(nsepblks - 1, eval(cmd1))

        # OutBlks    
        for _ in range((nsepblks + 1) // 2):
            # 1st blk, fuse
            if _ == 0:
                if self.config.multitask in ['local', 'idol']:
                    args_dim, kwargs_dim = 0, 0
                elif self.config.multitask == 'fuse1':
                    args_dim, kwargs_dim = -1, 2
            else:
                args_dim, kwargs_dim = 2, 2

            # Lst blk. Case 2, mullev
            if _ == 0 and self.config.mullev == 'true':
                self.transformer_blocks.append(ins_with_str(
                    [cmd.format(skips_in=0)]
                    + [cmd.format(skips_in=len(self.lev_idxs)
                                  * self.inner_dim)] * (self.nmodals - 1),
                    **{**inskwargs, 'intype': [[args_dim],
                                               {'skips': [2, [0,
                                                              *[len(self.lev_idxs)] * (self.nmodals - 1)]]}], 'outtype': 2}))  # [1, 1]. Now asymmetric

            # Middle blks. Case 1, skips
            else:  # to be compatible with old version
                outtype = 0 if nsepblks == 1 else 2
                if self.config.skips == '1':  # TODO: asymmetric '1' like mullev
                    self.transformer_blocks.append(ins_with_str(
                        cmd.format(skips_in=self.inner_dim),
                        **{**inskwargs, 'intype': [[args_dim], {'skips': kwargs_dim}], 'outtype': outtype}))
                else:
                    self.transformer_blocks.append(ins_with_str(
                        cmd1, **{**inskwargs, 'intype': [[args_dim], {}], 'outtype': outtype}))

            if nsepblks > 1:
                # TODO: env smaller blks into larger blks
                for i in range(2 - 1):
                    self.transformer_blocks.append(ins_with_str(
                        cmd1, **{**inskwargs, 'intype': [[2], {}],
                                 'outtype': 0 if _ == (nsepblks + 1) // 2 - 1 and i == 2 - 2 else 2}))

        if self.config.multitask == 'idol':
            if not (self.config.skips == 'none' and self.config.mullev == 'none'):
                raise ValueError
            modal_bidxs = [0, 11, 19, len(self.transformer_blocks) - 2]
            mod_spa_attn_blk_cmd = (bas_tsfm_blk_cmd.replace('BasicTransformerBlock', 'ModalSpatialAttnBlk')
                                    # .replace('use_rope=True', 'use_rope=False')
                                    [: -1]
                                    + f'nmodals={self.nmodals},)')
            # print(f"{'>>>' * 10} No RoPE for CMSA!")
            self.modal_spatial_attn_blks = nn.ModuleDict(
                dict(zip([f'{_}' for _ in modal_bidxs],
                         [eval(mod_spa_attn_blk_cmd.format(skips_in=0)) for _ in range(len(modal_bidxs))])))
            for bidx in modal_bidxs:
                blk = self.transformer_blocks[bidx + 1]
                if not isinstance(blk, CopyTrain):
                    blk = [blk]
                for m in blk:
                    m.attn2.processor.attnsaver = True

        if self.config.latent_pose == 'ipi0':
            # self.pose_attn_bidxs = [0, 6, 11, 15, 19, 25, len(self.transformer_blocks) - 2]
            self.pose_attn_bidxs = list(range(16, 32, 4))
            pose_attn_blk_cmd = (bas_tsfm_blk_cmd.format(skips_in=0)[: -1]
                                 + f'pose_attn=True,)')
            for bidx in self.pose_attn_bidxs:
                self.transformer_blocks[bidx] = eval(pose_attn_blk_cmd)

        if self.config.norm_type != "ada_norm_single":
            self.norm_out = nn.LayerNorm(self.inner_dim, elementwise_affine=False, eps=1e-6)
            self.proj_out_1 = nn.Linear(self.inner_dim, 2 * self.inner_dim)
            self.proj_out_2 = nn.Linear(
                self.inner_dim, self.config.patch_size_t * self.config.patch_size * self.config.patch_size * self.out_channels
            )
        elif self.config.norm_type == "ada_norm_single":
            self.scale_shift_table = nn.Parameter(torch.randn(2, self.inner_dim) / self.inner_dim**0.5)  #TODO: ?
            if self.config.multitask in ['local', 'fuse1', 'idol']:
                inskwargs.update(intype=[[0], {}], outtype=2)
            self.norm_out = nn.LayerNorm(self.inner_dim, elementwise_affine=False, eps=1e-6)
            self.proj_out = ins_with_str(f'nn.Linear(\
                {self.inner_dim}, {self.config.patch_size_t * self.config.patch_size * self.config.patch_size * self.out_channels}\
            )', **inskwargs)

        # PixArt-Alpha blocks.
        self.adaln_single = None
        if self.config.norm_type == "ada_norm_single":
            # TODO(Sayak, PVP) clean this, for now we use sample size to determine whether to use
            # additional conditions until we find better name
            self.adaln_single = AdaLayerNormSingle(
                self.inner_dim, use_additional_conditions=self.use_additional_conditions
            )
            
            if self.multitask == 'emb':
                self.mod_emb_proj = EmbLayers(2 * self.nmodals, embedding_dim=self.inner_dim)

        self.caption_projection = None
        if self.caption_channels is not None:
            self.caption_projection = PixArtAlphaTextProjection(
                in_features=self.caption_channels, hidden_size=self.inner_dim
            )
        
        if self.config.multitask in ['rg', 'rg1']:
            # Essence: use multilevel feats. Discriminative
            self.rg_bidxs = list(range(len(self.transformer_blocks)))[1: -4: 2]
            if self.config.multitask == 'rg':
                tsks = ['latent_deps']  # 'deps'
                uprate = [0, 0, 0]
            elif self.config.multitask == 'rg1':
                tsks = ['ds_deps']  # deps']  # , 'ds_normals', 'ds_parts']
                uprate = [0, 0, 0]  # [2, 0, 0]
            self.rgs = {}
            for tsk in tsks:
                # self.rgs[tsk] = AggregationNetwork(  #TODO: sep mul feats
                #     [self.inner_dim] * len(self.rg_bidxs), None, save_timestep=[0], num_timesteps=1000,
                #     use_output_head=True, bottleneck_sequential=False)  # sep nets
                self.rgs[tsk] = AggTsfm(
                    bas_tsfm_blk_cmd, len(self.rg_bidxs), norm_type=self.config.norm_type,
                    inner_dim=self.inner_dim, patch_size_t=self.patch_size_t,
                    patch_size=self.patch_size, out_channels=self.out_channels,
                    uprate=uprate)
            self.rgs = nn.ModuleDict(self.rgs)

    def _create_latent_pose(self):
        if self.config.latent_pose == 'aa':
            self.enc_pose = PoseGuider()
        elif self.config.latent_pose in ['aa_hack', 'ipi0']:
            pos_embed_pose = PatchEmbed2D(
                num_frames=self.config.sample_size_t,
                height=self.config.sample_size[0],
                width=self.config.sample_size[1],
                patch_size_t=self.config.patch_size_t,
                patch_size=self.config.patch_size,
                in_channels=128,
                embed_dim=self.inner_dim,
                interpolation_scale=self.interpolation_scale,
                interpolation_scale_t=self.interpolation_scale_t,
                use_abs_pos=not self.config.use_rope)
            self.enc_pose = PoseGuider(latent_pose='aa_hack', final_proj=pos_embed_pose)
        elif self.config.latent_pose not in [None, 'none']:
            raise ValueError
    
    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    def ti2m(self, data):
        data = copy.deepcopy(data)  # incl inf

        mot263s0, PA = raw2mot263(smpls={k: v for k, v in data.items() if k in SMPL_KS})

        # Cont pred
        batch = {
            "length": [0] * len(data['txt']),
            "text": data['txt'],
            'motion': mot263s0
        }
        outputs = self.t2m_model(batch, task='pred')  # DiT. Incl 0. #TODO: check quota

        raw_mots = front_mots2raw(outputs['joints'], PA)

        data['gen_kpmaps'] = j3d2kpmap(raw_mots['joints'], data['smpls'], data['kpmaps'].shape[-2:],
                                       data['orig_imsz'], data['bboxes'])

        return data

    @staticmethod
    def pred_x0_from_noise(noisy_model_input, model_pred, scheduler, timestep: int,
                           prev_timestep: Optional[int] = None):
        """ latentman/pipelines/scheduling_ddim.py """
        assert 'DDIMScheduler' in str(type(scheduler))
        alpha_prod_t = scheduler.alphas_cumprod.to(noisy_model_input.device)[timestep]
        beta_prod_t = 1 - alpha_prod_t
        alpha_prod_t_prev = None
        if prev_timestep is not None:
            raise NotImplementedError
            alpha_prod_t_prev = scheduler.alphas_cumprod[
                prev_timestep] if prev_timestep >= 0 else scheduler.final_alpha_cumprod

        pred_x0 = (noisy_model_input - beta_prod_t ** (0.5)
                   * model_pred) / alpha_prod_t ** (0.5)
        return pred_x0

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: Optional[torch.LongTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        added_cond_kwargs: Dict[str, torch.Tensor] = None,
        class_labels: Optional[torch.LongTensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        use_image_num: Optional[int] = 0,
        
        data=None,
        return_dict: bool = True,
    ):
        """
        The [`Transformer2DModel`] forward method.

        Args:
            hidden_states (`torch.LongTensor` of shape `(batch size, num latent pixels)` if discrete, `torch.FloatTensor` of shape `(batch size, channel, height, width)` if continuous):
                Input `hidden_states`.
            encoder_hidden_states ( `torch.FloatTensor` of shape `(batch size, sequence len, embed dims)`, *optional*):
                Conditional embeddings for cross attention layer. If not given, cross-attention defaults to
                self-attention.
            timestep ( `torch.LongTensor`, *optional*):
                Used to indicate denoising step. Optional timestep to be applied as an embedding in `AdaLayerNorm`.
            class_labels ( `torch.LongTensor` of shape `(batch size, num classes)`, *optional*):
                Used to indicate class labels conditioning. Optional class labels to be applied as an embedding in
                `AdaLayerZeroNorm`.
            cross_attention_kwargs ( `Dict[str, Any]`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            attention_mask ( `torch.Tensor`, *optional*):
                An attention mask of shape `(batch, key_tokens)` is applied to `encoder_hidden_states`. If `1` the mask
                is kept, otherwise if `0` it is discarded. Mask will be converted into a bias, which adds large
                negative values to the attention scores corresponding to "discard" tokens.
            encoder_attention_mask ( `torch.Tensor`, *optional*):
                Cross-attention mask applied to `encoder_hidden_states`. Two formats supported:

                    * Mask `(batch, sequence_length)` True = keep, False = discard.
                    * Bias `(batch, 1, sequence_length)` 0 = keep, -10000 = discard.

                If `ndim == 2`: will be interpreted as a mask, then converted into a bias consistent with the format
                above. This bias will be added to the cross-attention scores.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.unets.unet_2d_condition.UNet2DConditionOutput`] instead of a plain
                tuple.

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """
        ### Pre hook
        # if self.pipe == 'ti2mi2v':  # and 'pred_kpmaps' not in data:
        #     data = self.ti2m(data)

        batch_size, c, frame, h, w = hidden_states.shape
        # print('hidden_states.shape', hidden_states.shape)
        frame = frame - use_image_num  # 21-4=17
        if cross_attention_kwargs is not None:
            if cross_attention_kwargs.get("scale", None) is not None:
                print.warning("Passing `scale` to `cross_attention_kwargs` is deprecated. `scale` will be ignored.")
        # ensure attention_mask is a bias, and give it a singleton query_tokens dimension.
        #   we may have done this conversion already, e.g. if we came here via UNet2DConditionModel#forward.
        #   we can tell by counting dims; if ndim == 2: it's a mask rather than a bias.
        # expects mask of shape:
        #   [batch, key_tokens]
        # adds singleton query_tokens dimension:
        #   [batch,                    1, key_tokens]
        # this helps to broadcast it as a bias over attention scores, which will be in one of the following shapes:
        #   [batch,  heads, query_tokens, key_tokens] (e.g. torch sdp attn)
        #   [batch * heads, query_tokens, key_tokens] (e.g. xformers or classic attn)
        attention_mask_vid, attention_mask_img = None, None
        if attention_mask is not None and attention_mask.ndim == 4:
            # assume that mask is expressed as:
            #   (1 = keep,      0 = discard)
            # convert mask into a bias that can be added to attention scores:
            #   (keep = +0,     discard = -10000.0)
            # b, frame+use_image_num, h, w -> a video with images
            # b, 1, h, w -> only images
            attention_mask = attention_mask.to(self.dtype)
            if get_sequence_parallel_state():
                if npu_config is not None:
                    attention_mask_vid = attention_mask[:, :frame * hccl_info.world_size]  # b, frame, h, w
                    attention_mask_img = attention_mask[:, frame * hccl_info.world_size:]  # b, use_image_num, h, w
                else:
                    # print('before attention_mask.shape', attention_mask.shape)
                    attention_mask_vid = attention_mask[:, :frame * nccl_info.world_size]  # b, frame, h, w
                    attention_mask_img = attention_mask[:, frame * nccl_info.world_size:]  # b, use_image_num, h, w
                    # print('after attention_mask.shape', attention_mask_vid.shape)
            else:
                attention_mask_vid = attention_mask[:, :frame]  # b, frame, h, w
                attention_mask_img = attention_mask[:, frame:]  # b, use_image_num, h, w

            if attention_mask_vid.numel() > 0:
                attention_mask_vid_first_frame = attention_mask_vid[:, :1].repeat(1, self.patch_size_t-1, 1, 1)
                attention_mask_vid = torch.cat([attention_mask_vid_first_frame, attention_mask_vid], dim=1)
                attention_mask_vid = attention_mask_vid.unsqueeze(1)  # b 1 t h w
                attention_mask_vid = F.max_pool3d(attention_mask_vid, kernel_size=(self.patch_size_t, self.patch_size, self.patch_size), 
                                                  stride=(self.patch_size_t, self.patch_size, self.patch_size))
                attention_mask_vid = rearrange(attention_mask_vid, 'b 1 t h w -> (b 1) 1 (t h w)') 
            if attention_mask_img.numel() > 0:
                attention_mask_img = F.max_pool2d(attention_mask_img, kernel_size=(self.patch_size, self.patch_size), stride=(self.patch_size, self.patch_size))
                attention_mask_img = rearrange(attention_mask_img, 'b i h w -> (b i) 1 (h w)') 

            attention_mask_vid = (1 - attention_mask_vid.bool().to(self.dtype)) * -10000.0 if attention_mask_vid.numel() > 0 else None
            attention_mask_img = (1 - attention_mask_img.bool().to(self.dtype)) * -10000.0 if attention_mask_img.numel() > 0 else None

            if frame == 1 and use_image_num == 0 and not get_sequence_parallel_state():
                attention_mask_img = attention_mask_vid
                attention_mask_vid = None
        # convert encoder_attention_mask to a bias the same way we do for attention_mask
        # import ipdb;ipdb.set_trace()
        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 3:  
            # b, 1+use_image_num, l -> a video with images
            # b, 1, l -> only images
            encoder_attention_mask = (1 - encoder_attention_mask.to(self.dtype)) * -10000.0
            in_t = encoder_attention_mask.shape[1]
            encoder_attention_mask_vid = encoder_attention_mask[:, :in_t-use_image_num]  # b, 1, l
            encoder_attention_mask_vid = rearrange(encoder_attention_mask_vid, 'b 1 l -> (b 1) 1 l') if encoder_attention_mask_vid.numel() > 0 else None

            encoder_attention_mask_img = encoder_attention_mask[:, in_t-use_image_num:]  # b, use_image_num, l
            encoder_attention_mask_img = rearrange(encoder_attention_mask_img, 'b i l -> (b i) 1 l') if encoder_attention_mask_img.numel() > 0 else None

            if frame == 1 and use_image_num == 0 and not get_sequence_parallel_state():
                encoder_attention_mask_img = encoder_attention_mask_vid
                encoder_attention_mask_vid = None

        if npu_config is not None and attention_mask_vid is not None:
            attention_mask_vid = npu_config.get_attention_mask(attention_mask_vid, attention_mask_vid.shape[-1])
            encoder_attention_mask_vid = npu_config.get_attention_mask(encoder_attention_mask_vid,
                                                                       attention_mask_vid.shape[-2])
        if npu_config is not None and attention_mask_img is not None:
            attention_mask_img = npu_config.get_attention_mask(attention_mask_img, attention_mask_img.shape[-1])
            encoder_attention_mask_img = npu_config.get_attention_mask(encoder_attention_mask_img,
                                                                       attention_mask_img.shape[-2])


        # 1. Input
        frame = ((frame - 1) // self.patch_size_t + 1) if frame % 2 == 1 else frame // self.patch_size_t  # patchfy
        # print('frame', frame)
        height, width = hidden_states.shape[-2] // self.patch_size, hidden_states.shape[-1] // self.patch_size

        # if self.config.latent_pose == 'aa':
        #     orig_hidden_states = hidden_states.clone()
        #     latent_pose = self.enc_pose(data['kpmaps'])
        #     hidden_states = hidden_states + latent_pose  # pre-process. https://github.com/guoqincode/Open-AnimateAnyone/issues/56#issuecomment-1868835223
        #     # https://github.com/guoqincode/Open-AnimateAnyone/issues/80#issuecomment-1880485546

        added_cond_kwargs = {"resolution": None, "aspect_ratio": None}
        hidden_states_vid, hidden_states_img, encoder_hidden_states_vid, encoder_hidden_states_img, \
        timestep_vid, timestep_img, embedded_timestep_vid, embedded_timestep_img = self._operate_on_patched_inputs(
            hidden_states, encoder_hidden_states, timestep, added_cond_kwargs, batch_size, frame, use_image_num,
            **data
        )

        if self.config.latent_pose in ['aa_hack', 'ipi0']:
            orig_hidden_states = hidden_states_vid.clone()
            latent_pose = self.enc_pose(data['kpmaps'], **vars(Namespace(num_frames=frame)))  #TODO: move outside timestep to show T2I-Adapter's advtgs over CtrlNet's
            latent_pose = latent_pose.repeat(hidden_states_vid.shape[0] // latent_pose.shape[0], 1, hidden_states_vid.shape[2] // latent_pose.shape[2])
            
            if self.config.latent_pose == 'aa_hack':
                hidden_states_vid = hidden_states_vid + latent_pose
        else:
            latent_pose = None

        # 2. Blocks
        # import ipdb;ipdb.set_trace()
        if get_sequence_parallel_state():
            if hidden_states_vid is not None:
                # print(333333333333333)
                hidden_states_vid = rearrange(hidden_states_vid, 'b s h -> s b h', b=batch_size).contiguous()
                encoder_hidden_states_vid = rearrange(encoder_hidden_states_vid, 'b s h -> s b h',
                                                      b=batch_size).contiguous()
                timestep_vid = timestep_vid.view(batch_size, 6, -1).transpose(0, 1).contiguous()
                # print('timestep_vid', timestep_vid.shape)

        # 1. Save feats during FF
        # 2. Save feats by 1st FF. More suitable for 2 nets
        skips_l = [hidden_states_vid]
        levs_l = [hidden_states_vid[..., self.inner_dim:]]
        feats_l = [hidden_states_vid]
        for bidx, block in enumerate(self.transformer_blocks):
            while True:
                if hidden_states_vid is not None:
                    N = (hidden_states_vid.shape[0] // batch_size
                        if not isinstance(block, CopyTrain) else 1)  # for sharing blocks, change along batch dim
                    if (getattr(block, 'skips_in', None)
                            or isinstance(block, CopyTrain) and getattr(block[1], 'skips_in', None)):
                        if self.config.mullev == 'true' and bidx > self.lev_idxs[-1]:  # wont appear togther w/ '1'
                            # Horizontal -> vertical slices. Lst one is input latent
                            skips = rearrange(torch.cat(levs_l[: -1], dim=-1), 'b n (l m c) -> b n (m l c)',
                                            l=len(levs_l) - 1, c=self.inner_dim)
                        elif self.config.skips == '1':
                            skips = skips_l.pop(-1)
                    else:
                        skips = None

                if self.training and self.gradient_checkpointing:

                    def create_custom_forward(module, return_dict=None):
                        def custom_forward(*inputs):
                            if return_dict is not None:
                                return module(*inputs, return_dict=return_dict)
                            else:
                                return module(*inputs)

                        return custom_forward

                    ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                    # import ipdb;ipdb.set_trace()
                    if hidden_states_vid is not None:
                        hidden_states_vid = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(block),
                            hidden_states_vid,
                            skips,
                            latent_pose,
                            attention_mask_vid.repeat(N, 1, 1),
                            encoder_hidden_states_vid.repeat(N, 1, 1),
                            encoder_attention_mask_vid.repeat(N, 1, 1),
                            timestep_vid.repeat(batch_size * N // timestep_vid.shape[0], 1),
                            cross_attention_kwargs,
                            class_labels,
                            frame, 
                            height, 
                            width,
                            batch_size, 
                            **ckpt_kwargs,
                        )
                    # import ipdb;ipdb.set_trace()
                    if hidden_states_img is not None:
                        hidden_states_img = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(block),
                            hidden_states_img,
                            attention_mask_img,
                            encoder_hidden_states_img,
                            encoder_attention_mask_img,
                            timestep_img,
                            cross_attention_kwargs,
                            class_labels,
                            1, 
                            height, 
                            width,
                            **ckpt_kwargs,
                        )
                else:
                    # print(bidx)
                    if hidden_states_vid is not None:
                        hidden_states_vid = block(
                            hidden_states_vid,  # (B, 9600, 2304)
                            skips=skips,
                            latent_poses=latent_pose,
                            attention_mask=attention_mask_vid.repeat(N, 1, 1),  # (B, 1, 9600)
                            encoder_hidden_states=encoder_hidden_states_vid.repeat(N, 1, 1),  # (B, 512, 2304)
                            encoder_attention_mask=encoder_attention_mask_vid.repeat(N, 1, 1),  # (B, 1, 512)
                            timestep=timestep_vid.repeat(batch_size * N // timestep_vid.shape[0], 1),  # (B, 6*2304)
                            cross_attention_kwargs=cross_attention_kwargs,  # None
                            class_labels=class_labels,  # None
                            frame=frame,  # 8
                            height=height,  # 30
                            width=width,  # 40
                            batch_size=batch_size,
                        )
                    if hidden_states_img is not None:
                        hidden_states_img = block(
                            hidden_states_img,
                            attention_mask=attention_mask_img,
                            encoder_hidden_states=encoder_hidden_states_img,
                            encoder_attention_mask=encoder_attention_mask_img,
                            timestep=timestep_img,
                            cross_attention_kwargs=cross_attention_kwargs,
                            class_labels=class_labels,
                            frame=1, 
                            height=height, 
                            width=width, 
                        )

                if hidden_states_vid is not None:
                    # if (getattr(block, 'ski', None)
                    #         or isinstance(block, CopyTrain) and block[0].ski):
                    if self.config.skips == '1' and bidx in self.skip_bidxs:
                        skips_l.append(hidden_states_vid)
                    if self.config.mullev == 'true' and bidx in self.lev_idxs:
                        # Whole modality as unit on or off
                        if bidx >= self.nsepblks and bidx < len(self.transformer_blocks) - self.nsepblks:
                            # Shared
                            assert hidden_states_vid.shape[-1] == self.inner_dim
                            levs_l.append(hidden_states_vid.repeat(1, 1, self.nmodals - 1))
                        elif bidx < self.nsepblks:
                            levs_l.append(hidden_states_vid[..., self.inner_dim:])
                        else:  # -1
                            pass
                    # Add back for following sharing
                    if isinstance(block, CopyTrain) and block.zeroout and type(block.outtype) == int:
                        hidden_states_vid = sum(hidden_states_vid.chunk(
                            len(block), dim=block.outtype))

                    if self.config.multitask in ['rg', 'rg1']:
                        if bidx in self.rg_bidxs:
                            feats_l.append(hidden_states_vid)
        
                if self.config.multitask == 'idol' and f'{bidx}' in self.modal_spatial_attn_blks:
                    if (isinstance(block, BasicTransformerBlock)
                            or isinstance(block, CopyTrain) and isinstance(block[0], BasicTransformerBlock)):
                        block = self.modal_spatial_attn_blks[f'{bidx}']
                    else:
                        break
                else:
                    break
        assert self.skips != '1' or len(skips_l) <= 1

        if get_sequence_parallel_state():
            if hidden_states_vid is not None:
                hidden_states_vid = rearrange(hidden_states_vid, 's b h -> b s h', b=batch_size).contiguous()

        # 3. Output
        output_vid, output_img = None, None 
        if hidden_states_vid is not None:
            N = hidden_states_vid.shape[0] // timestep_vid.shape[0]
            output_vid = self._get_output_for_patched_inputs(
                hidden_states=hidden_states_vid,
                timestep=timestep_vid.repeat(N, 1),
                class_labels=class_labels,
                embedded_timestep=embedded_timestep_vid.repeat(N, 1),
                num_frames=frame, 
                height=height,
                width=width,
            )  # b c t h w
        if hidden_states_img is not None:
            output_img = self._get_output_for_patched_inputs(
                hidden_states=hidden_states_img,
                timestep=timestep_img,
                class_labels=class_labels,
                embedded_timestep=embedded_timestep_img,
                num_frames=1, 
                height=height,
                width=width,
            )  # b c 1 h w
            if use_image_num != 0:
                output_img = rearrange(output_img, '(b i) c 1 h w -> b c i h w', i=use_image_num)

        if output_vid is not None and output_img is not None:
            output = torch.cat([output_vid, output_img], dim=2)
        elif output_vid is not None:
            output = output_vid
        elif output_img is not None:
            output = output_img
        
        if self.multitask == 'fuse1' and self.mullev == 'none' and self.skips == 'none':
            def get_alpha_beta(self, timestep: int, prev_timestep: Optional[int] = None):
                alpha_prod_t = self.alphas_cumprod[timestep]
                beta_prod_t = 1 - alpha_prod_t
                alpha_prod_t_prev = None
                if prev_timestep is not None:
                    raise NotImplementedError
                    alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod
                return alpha_prod_t, beta_prod_t, alpha_prod_t_prev
            
            alpha_prod_t, beta_prod_t, _ = get_alpha_beta(noise_scheduler, timesteps)
            # pred_x0 = (noisy_model_input - beta_prod_t ** (0.5) * model_pred) / alpha_prod_t ** (0.5)
            pred_noise = (hidden_states - alpha_prod_t ** 0.5 * output.clone()) / beta_prod_t ** 0.5
            # output[:, self.in_channels:] = pred_noise[:, self.in_channels:]  # noisy
            output[:, self.in_channels:] = (hidden_states - output.clone())[:, self.in_channels:]
        
        if self.config.multitask in ['rg', 'rg1']:
            feats = torch.cat(feats_l[: -1], dim=-1)
            self._rg_outs = {}
            for tsk, rg in self.rgs.items():
                self._rg_outs[tsk] = rg(feats_l[-1],
                                        feats,
                                        attention_mask_vid,
                                        encoder_hidden_states_vid,
                                        encoder_attention_mask_vid,
                                        timestep_vid,
                                        cross_attention_kwargs=cross_attention_kwargs,
                                        class_labels=class_labels,
                                        frame=frame,
                                        height=height,
                                        width=width,
                                        embedded_timestep_vid=embedded_timestep_vid,
                                        gradient_checkpointing=self.gradient_checkpointing)

        if self.config.multitask == 'emb':
            output = rearrange(output, '(m b) d t h w -> b (m d) t h w', b=batch_size)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)
    
    @classmethod
    def _viz_attn(cls):
        """ For 3D attention (spatialtemporal) in the DiT (Diffusion Transformer) architecture, for example,
            after the latent video is flatten, seqlen=#frames xhxw=8x30x40,
            how can we visualize the area of ​​attention of a certain area in the second
            frame on other frames? If there is a good github repo package, we can
            use it. Please give the code as concisely as possible.
        """
        pass  # placeholder

    def _operate_on_patched_inputs(self, hidden_states, encoder_hidden_states, timestep, added_cond_kwargs, batch_size, frame, use_image_num):
        # batch_size = hidden_states.shape[0]
        hidden_states_vid, hidden_states_img = self.pos_embed(hidden_states.to(self.dtype), frame)
        timestep_vid, timestep_img = None, None
        embedded_timestep_vid, embedded_timestep_img = None, None
        encoder_hidden_states_vid, encoder_hidden_states_img = None, None

        if self.adaln_single is not None:
            if self.use_additional_conditions and added_cond_kwargs is None:
                raise ValueError(
                    "`added_cond_kwargs` cannot be None when using additional conditions for `adaln_single`."
                )
            timestep, embedded_timestep = self.adaln_single(
                timestep, added_cond_kwargs, batch_size=batch_size, hidden_dtype=self.dtype
            )  # b 6d, b d
            if hidden_states_vid is None:
                timestep_img = timestep
                embedded_timestep_img = embedded_timestep
            else:
                timestep_vid = timestep
                embedded_timestep_vid = embedded_timestep
                if hidden_states_img is not None:
                    timestep_img = repeat(timestep, 'b d -> (b i) d', i=use_image_num).contiguous()
                    embedded_timestep_img = repeat(embedded_timestep, 'b d -> (b i) d', i=use_image_num).contiguous()

        if self.caption_projection is not None:
            encoder_hidden_states = self.caption_projection(encoder_hidden_states)  # b, 1+use_image_num, l, d or b, 1, l, d
            if hidden_states_vid is None:
                encoder_hidden_states_img = rearrange(encoder_hidden_states, 'b 1 l d -> (b 1) l d')
            else:
                encoder_hidden_states_vid = rearrange(encoder_hidden_states[:, :1], 'b 1 l d -> (b 1) l d')
                if hidden_states_img is not None:
                    encoder_hidden_states_img = rearrange(encoder_hidden_states[:, 1:], 'b i l d -> (b i) l d')


        return hidden_states_vid, hidden_states_img, encoder_hidden_states_vid, encoder_hidden_states_img, timestep_vid, timestep_img, embedded_timestep_vid, embedded_timestep_img

    
    
    def _get_output_for_patched_inputs(
        self, hidden_states, timestep, class_labels, embedded_timestep, num_frames, height=None, width=None
    ):  
        # import ipdb;ipdb.set_trace()
        if self.config.norm_type != "ada_norm_single":
            conditioning = self.transformer_blocks[0].norm1.emb(
                timestep, class_labels, hidden_dtype=self.dtype
            )
            shift, scale = self.proj_out_1(F.silu(conditioning)).chunk(2, dim=1)
            hidden_states = self.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
            hidden_states = self.proj_out_2(hidden_states)
        elif self.config.norm_type == "ada_norm_single":
            shift, scale = (self.scale_shift_table[None] + embedded_timestep[:, None]).chunk(2, dim=1)
            hidden_states = self.norm_out(hidden_states)  # nothing to do with batch dim
            # Modulation
            hidden_states = hidden_states * (1 + scale) + shift
            hidden_states = self.proj_out(hidden_states)
            hidden_states = hidden_states.squeeze(1)

        # unpatchify
        if self.adaln_single is None:
            height = width = int(hidden_states.shape[1] ** 0.5)
        
        # (B, 4, 8, 60, 80)
        output = rearrange(hidden_states, 'b (t h w) (m o p q d) -> b (m d) (t o) (h p) (w q)',
                           t=num_frames, h=height, o=self.patch_size_t, p=self.patch_size,
                           q=self.patch_size, d=self.out_channels)

        # import ipdb;ipdb.set_trace()
        # if output.shape[2] % 2 == 0:
        #     output = output[:, :, 1:]
        return output
     
def OpenSoraT2V_S_122(**kwargs):
    return OpenSoraT2V(num_layers=28, attention_head_dim=96, num_attention_heads=16, patch_size_t=1, patch_size=2,
                       norm_type="ada_norm_single", caption_channels=4096, cross_attention_dim=1536, **kwargs)

def OpenSoraT2V_B_122(**kwargs):
    return OpenSoraT2V(num_layers=32, attention_head_dim=96, num_attention_heads=16, patch_size_t=1, patch_size=2,
                       norm_type="ada_norm_single", caption_channels=4096, cross_attention_dim=1920, **kwargs)

def OpenSoraT2V_L_122(**kwargs):
    return OpenSoraT2V(num_layers=40, attention_head_dim=128, num_attention_heads=16, patch_size_t=1, patch_size=2,
                       norm_type="ada_norm_single", caption_channels=4096, cross_attention_dim=2048, **kwargs)

def OpenSoraT2V_ROPE_L_122(**kwargs):
    return OpenSoraT2V(num_layers=32, attention_head_dim=96, num_attention_heads=24, patch_size_t=1, patch_size=2,
                       norm_type="ada_norm_single", caption_channels=4096, cross_attention_dim=2304, **kwargs)

OpenSora_models = {
    "OpenSoraT2V-S/122": OpenSoraT2V_S_122,  #       1.1B
    "OpenSoraT2V-B/122": OpenSoraT2V_B_122,
    "OpenSoraT2V-L/122": OpenSoraT2V_L_122,
    "OpenSoraT2V-ROPE-L/122": OpenSoraT2V_ROPE_L_122,
}

OpenSora_models_class = {
    "OpenSoraT2V-S/122": OpenSoraT2V,
    "OpenSoraT2V-B/122": OpenSoraT2V,
    "OpenSoraT2V-L/122": OpenSoraT2V,
    "OpenSoraT2V-ROPE-L/122": OpenSoraT2V,
}

if __name__ == '__main__':
    from opensora.models.causalvideovae import ae_channel_config, ae_stride_config
    from opensora.models.causalvideovae import getae, getae_wrapper
    from opensora.models.causalvideovae import CausalVAEModelWrapper

    args = type('args', (), 
    {
        'ae': 'CausalVAEModel_4x8x8', 
        'attention_mode': 'xformers', 
        'use_rope': True, 
        'model_max_length': 300, 
        'max_height': 320,
        'max_width': 240,
        'num_frames': 1,
        'use_image_num': 0, 
        'compress_kv_factor': 1, 
        'interpolation_scale_t': 1,
        'interpolation_scale_h': 1,
        'interpolation_scale_w': 1,
    }
    )
    b = 16
    c = 8
    cond_c = 4096
    num_timesteps = 1000
    ae_stride_t, ae_stride_h, ae_stride_w = ae_stride_config[args.ae]
    latent_size = (args.max_height // ae_stride_h, args.max_width // ae_stride_w)
    num_frames = (args.num_frames - 1) // ae_stride_t + 1

    device = torch.device('cuda:0')
    model = OpenSoraT2V_ROPE_L_122(in_channels=c, 
                              out_channels=c, 
                              sample_size=latent_size, 
                              sample_size_t=num_frames, 
                              activation_fn="gelu-approximate",
                            attention_bias=True,
                            attention_type="default",
                            double_self_attention=False,
                            norm_elementwise_affine=False,
                            norm_eps=1e-06,
                            norm_num_groups=32,
                            num_vector_embeds=None,
                            only_cross_attention=False,
                            upcast_attention=False,
                            use_linear_projection=False,
                            use_additional_conditions=False, 
                            downsampler=None,
                            interpolation_scale_t=args.interpolation_scale_t, 
                            interpolation_scale_h=args.interpolation_scale_h, 
                            interpolation_scale_w=args.interpolation_scale_w, 
                            use_rope=args.use_rope, 
                            ).to(device)
    
    try:
        path = "PixArt-Alpha-XL-2-512.safetensors"
        from safetensors.torch import load_file as safe_load
        ckpt = safe_load(path, device="cpu")
        # import ipdb;ipdb.set_trace()
        if ckpt['pos_embed.proj.weight'].shape != model.pos_embed.proj.weight.shape and ckpt['pos_embed.proj.weight'].ndim == 4:
            repeat = model.pos_embed.proj.weight.shape[2]
            ckpt['pos_embed.proj.weight'] = ckpt['pos_embed.proj.weight'].unsqueeze(2).repeat(1, 1, repeat, 1, 1) / float(repeat)
            del ckpt['proj_out.weight'], ckpt['proj_out.bias']
        msg = model.load_state_dict(ckpt, strict=False)
        print(msg)
    except Exception as e:
        print(e)
    print(model)
    print(f'{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e9} B')
    # import sys;sys.exit()
    x = torch.randn(b, c,  1+(args.num_frames-1)//ae_stride_t+args.use_image_num, args.max_height//ae_stride_h, args.max_width//ae_stride_w).to(device)
    cond = torch.randn(b, 1+args.use_image_num, args.model_max_length, cond_c).to(device)
    attn_mask = torch.randint(0, 2, (b, 1+(args.num_frames-1)//ae_stride_t+args.use_image_num, args.max_height//ae_stride_h, args.max_width//ae_stride_w)).to(device)  # B L or B 1+num_images L
    cond_mask = torch.randint(0, 2, (b, 1+args.use_image_num, args.model_max_length)).to(device)  # B L or B 1+num_images L
    timestep = torch.randint(0, 1000, (b,), device=device)
    model_kwargs = dict(hidden_states=x, encoder_hidden_states=cond, attention_mask=attn_mask, 
                        encoder_attention_mask=cond_mask, use_image_num=args.use_image_num, timestep=timestep)
    with torch.no_grad():
        output = model(**model_kwargs)
    print(output[0].shape)




