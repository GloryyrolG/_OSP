from argparse import Namespace
import os, random, re
import numpy as np

import torch
from accelerate.logging import get_logger

from opensora.utils.utils import text_preprocessing

from opensora.dataset.t2v_datasets import SingletonMeta, DataSetProg
from opensora.dataset.t2v_datasets import T2V_dataset, get_kpmaps

logger = get_logger(__name__)

dataset_prog = DataSetProg()


class Meta_dataset(T2V_dataset):
    def __init__(self, args, transform, temporal_sample, tokenizer, transform_topcrop):
        super().__init__(args, transform, temporal_sample, tokenizer, transform_topcrop)

        if self.num_frames != 1:
            # inpaint
            # The proportion of executing the i2v task.
            self.i2v_ratio = getattr(args, 'i2v_ratio', 0.5)
            self.transition_ratio = getattr(args, 'transition_ratio', 0.4)
            self.v2v_ratio = getattr(args, 'v2v_ratio', 0.1)
            self.clear_video_ratio = getattr(args, 'clear_video_ratio', 0.0)
            assert self.i2v_ratio + self.transition_ratio + self.v2v_ratio + self.clear_video_ratio <= 1, 'The sum of i2v_ratio, transition_ratio, v2v_ratio and clear video ratio should be less than 1.'
        
        self.default_text_ratio = getattr(args, 'default_text_ratio', 0.1)
        self.default_text = f"The {'video' if self.num_frames != 1 else 'image'} showcases a scene with coherent and clear visuals."
    
    def get_mask_masked_video(self, video):
        # video shape (T, C, H, W)
        # 1 means masked, 0 means not masked
        t, c, h, w = video.shape
        mask = torch.ones_like(video, device=video.device, dtype=video.dtype)
        
        rand_num = random.random()
        # i2v
        if rand_num < self.i2v_ratio:
            mask[0] = 0
        # transition
        elif rand_num < self.i2v_ratio + self.transition_ratio:
            mask[0] = 0
            mask[-1] = 0
        # video continuation
        elif rand_num < self.i2v_ratio + self.transition_ratio + self.v2v_ratio:
            end_idx = random.randint(1, t)
            mask[:end_idx] = 0
        # clear video
        elif rand_num < self.i2v_ratio + self.transition_ratio + self.v2v_ratio + self.clear_video_ratio:
            mask[:] = 0
        # random mask
        else:
            idx_to_select = random.randint(0, t - 1)
            selected_indices = random.sample(range(0, t), idx_to_select)
            mask[selected_indices] = 0
        masked_video = video * (mask < 0.5)

        # save_video(masked_video.permute(0, 2, 3, 1).cpu().numpy(), 'masked_video.mp4')
        return dict(mask=mask, masked_video=masked_video)

    def drop(self, text):
        rand_num = random.random()
        rand_num_text = random.random()

        if rand_num < self.cfg:
            text = self.default_text if rand_num_text < self.default_text_ratio else ''

        return dict(text=text)

    
class Inpaint_dataset(Meta_dataset):
    def get_video(self, idx):
        video_path = dataset_prog.cap_list[idx]['path']
        assert os.path.exists(video_path), f"file {video_path} do not exist!"
        frame_indice = dataset_prog.cap_list[idx]['sample_frame_index']
        # Data filtering should not be in the repo
        video, sample_frame_ids = self.decord_read(video_path, predefine_num_frames=len(frame_indice))
        data = {}
        data['pixel_values'] = video

        # data.update(self.get_sapiens_outs(idx, sample_frame_ids))
        data.update(self.get_dep_outs(idx, sample_frame_ids))

        kp2ds_orig = self.get_kp2ds(idx, sample_frame_ids)

        h, w = video.shape[-2:]
        # assert h / w <= 17 / 16 and h / w >= 8 / 16, f'Only videos with a ratio (h/w) less than 17/16 and more than 8/16 are supported. But video ({video_path}) found ratio is {round(h / w, 2)} with the shape of {video.shape}'
        t = video.shape[0]
        st_ks = ['pixel_values', 'parts', 'deps', 'normals']  # 'feat3s'
        x = torch.cat([data[k] for k in st_ks], dim=1)
        tsfm_kwargs = vars(Namespace(kp2ds=kp2ds_orig))
        x, tsfm_returns = self.transform(x, kwargs=tsfm_kwargs)  # T C H W -> T C H W
        data.update(zip(st_ks, x.chunk(len(st_ks), dim=1)))  # never took it apart after putting it together

        kp2ds = tsfm_returns['kp2ds']
        kp_aug = False
        if kp_aug:
            if torch.rand([]) < 1 / 1000:
                print(f"{'>>>' * 10} Using kp_aug!")
            
            imgres = np.array([self.max_height, self.max_width])
            center = imgres[:: -1] / 2
            scale = imgres / 200  # same as imgres
            self.augm_params()
            kp2ds = self.j2d_processing(kp2ds.numpy(), center, scale)  # dtype
            kp2ds = torch.from_numpy(kp2ds).float()
    
        # Other data
        # h_orig, w_orig = (dataset_prog.cap_list[idx]['resolution']['height'],
        #                   dataset_prog.cap_list[idx]['resolution']['width'])  # orig
        # kpmaps_orig = get_kpmaps(kp2ds_orig, h_orig, w_orig)
        # assert kpmaps_orig.shape == video_orig.shape
        # kpmaps = self.transform(kpmaps_orig, kwargs=tsfm_kwargs).transpose(0, 1)
        kpmaps_crop = get_kpmaps(kp2ds, self.max_height, self.max_width)
        totensor, norm = self.transform.transforms[0], self.transform.transforms[-1]
        data['kpmaps'] = norm(totensor(kpmaps_crop))    

        # Stacked together along the channel and operated as one sample!
        # video = torch.cat([video, kpmaps], dim=1)

        # inpaint
        data.update(self.get_mask_masked_video(data['pixel_values']))
        # mask, masked_video = inpaint_cond_data['mask'], inpaint_cond_data['masked_video']
        # video = torch.cat([video, masked_video, mask], dim=1) # T 3*C H W

        # 1.
        # kpmaps = kpmaps.transpose(0, 1)
        # st_strs = ['video', 'masked_video', 'mask', 'kpmaps']
        # st_vars = {}
        # for st_str in st_strs:
        #     st_vars[st_str] = eval(f'{st_str}.transpose(0, 1)')  # T C H W -> C T H W
        # locals().update(st_vars)
        # 2.
        # x = torch.cat(vars_, dim=1)
        # data.update(zip(st_ks, x.chunk(len(vars_), dim=0)))
        # 3.
        st_ks.extend(['masked_video', 'mask', 'kpmaps'])
        for st_k in st_ks:
            data[st_k] = data[st_k].transpose(0, 1)

        if '/motion' in video_path.lower():
            import warnings; warnings.warn(f"{'>>>' * 10} Act as text!")
            text = ' '.join(video_path.split('/')[-1].split('_')[: -1]).replace(' +', ',').lower()
            text = re.sub(r'\d', '', text).strip()  # rm no
        else:
            text = dataset_prog.cap_list[idx]['cap']
        if not isinstance(text, list):
            text = [text]
        raw_text = [random.choice(text)]

        text = text_preprocessing(raw_text, support_Chinese=self.support_Chinese)

        out_ = self.drop(text)
        text = out_['text']

        text_tokens_and_mask = self.tokenizer(
            text,
            max_length=self.model_max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors='pt'
        )
        input_ids = text_tokens_and_mask['input_ids']
        cond_mask = text_tokens_and_mask['attention_mask']

        return dict(input_ids=input_ids, cond_mask=cond_mask,
                    txt=text, vid_pth=video_path, **data)
    