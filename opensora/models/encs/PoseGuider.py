import os
import torch
import torch.nn as nn
import torch.nn.init as init
from einops import rearrange
import numpy as np

from opensora.models.causalvideovae.model.modules.conv import Conv2d
from opensora.models.diffusion.opensora.modules import PatchEmbed2D


class RepeatCat(nn.Module):
    def __init__(self, kernel_size):
        super(RepeatCat, self).__init__()
        self.kernel_size = kernel_size
    
    def forward(self, x):
        return torch.cat([x[:, :, : 1].repeat(1, 1, self.kernel_size - 1, 1, 1), x], dim=2)


class PoseGuider(nn.Module):
    def __init__(self, noise_latent_channels=4, latent_pose='aa', final_proj=None):
        super(PoseGuider, self).__init__()

        self.conv_layers = nn.Sequential(
            Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1),
            nn.GroupNorm(3, 3),  #TODO: too few
            nn.ReLU(),

            nn.ZeroPad3d((0, 1, 0, 1, 0, 0)),  # causalvideovae
            Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2),

            nn.GroupNorm(16, 16),
            nn.ReLU(),

            Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
            nn.GroupNorm(16, 16),
            nn.ReLU(),

            nn.ZeroPad3d((0, 1, 0, 1, 0, 0)),
            RepeatCat(3),
            nn.Conv3d(in_channels=16, out_channels=32, kernel_size=3, stride=2),

            nn.GroupNorm(32, 32),
            nn.ReLU(),

            Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.GroupNorm(32, 32),
            nn.ReLU(),

            nn.ZeroPad3d((0, 1, 0, 1, 0, 0)),
            RepeatCat(3),
            nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, stride=2),

            nn.GroupNorm(32, 64),
            nn.ReLU(),

            Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.GroupNorm(32, 64),
            nn.ReLU(),
            Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(32, 128),
            nn.ReLU()
        )

        # Final projection layer
        self.latent_pose = latent_pose
        if self.latent_pose == 'aa':
            self.final_proj = Conv2d(in_channels=128, out_channels=noise_latent_channels, kernel_size=1)
        elif self.latent_pose in ['aa_hack', 'ipi0']:
            self.final_proj = final_proj
        else:
            raise ValueError

        # Initialize layers
        self._initialize_weights()

        self.scale = nn.Parameter(torch.ones(1))

    # def _initialize_weights(self):
    #     # Initialize weights with Gaussian distribution and zero out the final layer
    #     for m in self.conv_layers:
    #         if isinstance(m, Conv2d):
    #             init.normal_(m.weight, mean=0.0, std=0.02)
    #             if m.bias is not None:
    #                 init.zeros_(m.bias)

    #     init.zeros_(self.final_proj.weight)
    #     if self.final_proj.bias is not None:
    #         init.zeros_(self.final_proj.bias)
    
    def _initialize_weights(self):
        # Initialize weights with He initialization and zero out the biases
        for m in self.conv_layers:
            if isinstance(m, Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                init.normal_(m.weight, mean=0.0, std=np.sqrt(2. / n))
                if m.bias is not None:
                    init.zeros_(m.bias)

        # For the final projection layer, initialize weights to zero (or you may choose to use He initialization here as well)
        # init.zeros_(self.final_proj.weight)
        # if self.final_proj.bias is not None:
        #     init.zeros_(self.final_proj.bias)
        for p in self.final_proj.parameters():
            nn.init.zeros_(p)

    def forward(self, x, **kwargs):
        # assert x.dim() == 5
        # video_length = x.shape[2]
        # x = rearrange(x, "b c f h w -> (b f) c h w")

        x = self.conv_layers(x)
        x = self.final_proj(x, **kwargs)

        if self.latent_pose in ['aa_hack', 'ipi0']:
            x = x[0]

        # x = rearrange(x, "(b f) c h w -> b f c h w", f=video_length)

        return x * self.scale

    @classmethod
    def from_pretrained(cls,pretrained_model_path):
        if not os.path.exists(pretrained_model_path):
            print(f"There is no model file in {pretrained_model_path}")
        print(f"loaded PoseGuider's pretrained weights from {pretrained_model_path} ...")

        state_dict = torch.load(pretrained_model_path, map_location="cpu")
        model = PoseGuider(noise_latent_channels=4)
                
        m, u = model.load_state_dict(state_dict, strict=False)
        # print(f"### missing keys: {len(m)}; \n### unexpected keys: {len(u)};")        
        params = [p.numel() for n, p in model.named_parameters()]
        print(f"### PoseGuider's Parameters: {sum(params) / 1e6} M")
        
        return model
