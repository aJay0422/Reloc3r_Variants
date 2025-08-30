### from reloc3r
import os
import os.path as osp
import numpy as np

import src.reloc3r_variants.models.reloc3r_modules.path_to_reloc3r

from third_party.reloc3r.reloc3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from third_party.reloc3r.reloc3r.utils.image import imread_cv2

DATA_ROOT='./data/megadepth1500'
DESCRIPTION_ROOT = "./data/descriptions/megadepth1500"


class MegaDepth_valid(BaseStereoViewDataset):
    def __init__(self, description=False, prompt_id=2, *args, **kwargs):
        self.ROOT = DATA_ROOT
        self.use_description = description
        if self.use_description:
            assert prompt_id is not None, "prompt_id is None. Please specify a prompt_id."
            self.DESCRIPTION_ROOT = osp.join(DESCRIPTION_ROOT, f'prompt{prompt_id}')
        super().__init__(*args, **kwargs)
        self.metadata = dict(np.load(osp.join(self.ROOT, f'megadepth_meta_test.npz'), allow_pickle=True))
        with open(osp.join(self.ROOT, f'megadepth_test_pairs.txt'), 'r') as f:
            self.scenes = f.readlines()

        self.load_depth = False

    def __len__(self):
        return len(self.scenes)
    
    def _get_views(self, idx, resolution,  rng):
        """
        load data for megadepth_validation views
        """
        # load metadata
        views = []
        image_idx1, image_idx2 = self.scenes[idx].strip().split(' ')
        view_idxs = [image_idx1, image_idx2]
        for view_idx in view_idxs:
            input_image_filename = osp.join(self.ROOT, view_idx)
            # load rgb images
            input_rgb_image = imread_cv2(input_image_filename)
            # load metadata
            intrinsics = np.float32(self.metadata[view_idx].item()['intrinsic'])
            camera_pose = np.linalg.inv(np.float32(self.metadata[view_idx].item()['pose']))

            if self.use_description:
                description_path = self.get_description_path(view_idx)
                with open(description_path, 'r') as f:
                    description = f.read()
            else:
                description = ""
                
            image, intrinsics = self._crop_resize_if_necessary(
                input_rgb_image, intrinsics, resolution, rng=rng, info=(self.ROOT, view_idx))
            
            views.append(dict(
                img=image,
                camera_pose=camera_pose,  # cam2world
                camera_intrinsics=intrinsics,
                dataset='MegaDepth',
                label=self.ROOT,
                instance=view_idx,
                description=description,))
        return views
    
    def get_description_path(self, view_idx):
        # Example of view_idx: Undistorted_SfM/0022/images/427154679_de14c315f4_o.jpg
        splitted = view_idx.split(os.sep)
        scene_id = splitted[1]
        img_name = splitted[-1][:-4]  # remove .jpg
        description_path = osp.join(self.DESCRIPTION_ROOT, scene_id, img_name + '.txt')
        return description_path