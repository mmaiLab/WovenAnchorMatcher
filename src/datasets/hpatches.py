import os
import os.path as osp
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from src.utils.dataset import read_hpatches_color, pad_hpatches


class HPatchesDataset(Dataset):
    def __init__(self,
                 root_dir,
                 mode='train',
                 img_resize=None,
                 df=None,
                 padding=True,
                 pad_df=None,
                 augment_fn=None,
                 **kwargs):
        super().__init__()
        self.root_dir = root_dir
        self.mode = mode

        # prepare scene_info and pair_info
        self.scene_folders = sorted(
            name for name in os.listdir(self.root_dir)
            if osp.isdir(osp.join(self.root_dir, name)))
        self.pair_infos = []
        for scene in self.scene_folders:
            if scene in ["i_contruction", "i_crownnight", "i_dc", "i_pencils",
                         "i_whitebuilding", "v_artisans", "v_astronautis", "v_talent"]:
                continue
            img0 = osp.join(scene, "1.ppm")
            if not osp.exists(osp.join(self.root_dir, img0)):
                continue
            i=2
            while True:
                img1 = osp.join(scene, f"{i}.ppm") # image 2.ppm ...
                H = osp.join(scene, f"H_1_{i}") # image H_1_2 ...
                if osp.exists(osp.join(self.root_dir, img1)) and osp.exists(osp.join(self.root_dir, H)):
                    self.pair_infos.append([(img0, img1), H])
                    i += 1
                else:
                    break

        # parameters for image resizing, padding and depthmap padding
        self.img_resize = img_resize
        self.df = df
        self.padding = padding
        self.pad_df = pad_df

        self.coarse_scale = kwargs.get('coarse_scale', 0.125)

    def __len__(self):
        return len(self.pair_infos)

    def __getitem__(self, idx):
        (img_name0, img_name1), H_name = self.pair_infos[idx]
        img_path0 = osp.join(self.root_dir, img_name0)
        img_path1 = osp.join(self.root_dir, img_name1)
        imagec_0, scale0, mask0, prepad_size0 = read_hpatches_color(
            img_path0, self.img_resize, self.df, padding=self.padding, set_max=False)
        imagec_1, scale1, mask1, prepad_size1 = read_hpatches_color(
            img_path1, self.img_resize, self.df, padding=self.padding, set_max=False)
        if self.padding:
            imagec_0, imagec_1, mask0, mask1 = pad_hpatches(imagec_0, imagec_1, prepad_size0, prepad_size1, self.pad_df)

        # read Homography. shape: (3, 3)
        H_path = osp.join(self.root_dir, H_name)
        H_0to1 = torch.from_numpy(np.loadtxt(H_path, dtype=np.float32))
        scene = img_name0.split('/')[0]

        data = {
            'imagec_0': imagec_0.squeeze(0),  # (3, h, w)
            'imagec_1': imagec_1.squeeze(0),  # (3, h, w)
            'prepad_size0': prepad_size0,  # [h, w]
            'prepad_size1': prepad_size1,
            'H_0to1': H_0to1, # (3, 3)
            'scale0': scale0,  # [scale_w, scale_h]
            'scale1': scale1,
            'dataset_name': 'HPatches',
            'scene_id': scene,
            'pair_id': idx,
            'pair_names': (img_name0, img_name1),
        }
        if mask0 is not None:  # img_padding is True
            if self.coarse_scale:
                [mask0, mask1] = F.interpolate(torch.stack([mask0, mask1], dim=0)[None].float(),
                                                       scale_factor=self.coarse_scale,
                                                       mode='nearest',
                                                       recompute_scale_factor=False)[0].bool()
            data.update({'mask0': mask0, 'mask1': mask1})

        return data
