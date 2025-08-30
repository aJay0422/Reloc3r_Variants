import numpy as np
import src.reloc3r_variants.models.reloc3r_modules.path_to_reloc3r

from third_party.reloc3r.reloc3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from third_party.reloc3r.reloc3r.utils.image import imread_cv2, cv2
from collections import OrderedDict
# from pdb import set_trace as bb

import os.path as osp


DATA_ROOT = './data/scannet1500' 
DESCRIPTION_ROOT = "./data/descriptions/scannet1500"



def label_to_str(label):
    return '_'.join(label)


class ScanNet1500(BaseStereoViewDataset):
    
    def __init__(self, description=False, prompt_id=2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_root = DATA_ROOT
        self.use_description = description
        if self.use_description:
            assert prompt_id is not None, "prompt_id is None. Please specify a prompt_id."
            self.DESCRIPTION_ROOT = osp.join(DESCRIPTION_ROOT, f'prompt{prompt_id}')
        self.pairs_path = '{}/test.npz'.format(self.data_root)
        self.subfolder_mask = 'scannet_test_1500/scene{:04d}_{:02d}'
        with np.load(self.pairs_path) as data:
            self.pair_names = data['name']

    def __len__(self):
        return len(self.pair_names)

    def _get_views(self, idx, resolution, rng):
        scene_name, scene_sub_name, name1, name2 = self.pair_names[idx]

        views = []

        for name in [name1, name2]: 
            
            color_path = '{}/{}/color/{}.jpg'.format(self.data_root, self.subfolder_mask, name).format(scene_name, scene_sub_name)
            color_image = imread_cv2(color_path)  
            color_image = cv2.resize(color_image, (640, 480))

            intrinsics_path = '{}/{}/intrinsic/intrinsic_depth.txt'.format(self.data_root, self.subfolder_mask).format(scene_name, scene_sub_name)
            intrinsics = np.loadtxt(intrinsics_path).astype(np.float32)[0:3,0:3]

            pose_path = '{}/{}/pose/{}.txt'.format(self.data_root, self.subfolder_mask, name).format(scene_name, scene_sub_name)
            camera_pose = np.loadtxt(pose_path).astype(np.float32)

            if self.use_description:
                description_path = osp.join(self.DESCRIPTION_ROOT, "scene{:04d}_{:02d}".format(scene_name, scene_sub_name), "color", f"{name}.txt")
                with open(description_path, 'r') as f:
                    description = f.read()
            else:
                description = ""

            color_image, intrinsics = self._crop_resize_if_necessary(color_image, 
                                                                     intrinsics, 
                                                                     resolution, 
                                                                     rng=rng)

            view_idx_splits = color_path.split('/')

            views.append(dict(
                img = color_image,
                camera_intrinsics = intrinsics,
                camera_pose = camera_pose,
                dataset = 'ScanNet1500',
                label = label_to_str(view_idx_splits[:-1]),
                instance = view_idx_splits[-1],
                description = description,
                ))
        # return OrderedDict([("view1", views[0]), ("view2", views[1])])
        return views
