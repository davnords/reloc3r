import os.path as osp
import numpy as np
import json
import glob

from reloc3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from reloc3r.utils.image import imread_cv2
import random
from pathlib import Path

DATA_ROOT = "/mimer/NOBACKUP/groups/3d-dl/vkitti"  

class VKitti(BaseStereoViewDataset):
    def __init__(self, *args, ROOT=DATA_ROOT, **kwargs):
        self.ROOT = ROOT
        super().__init__(*args, **kwargs)

        txt_path = osp.join(self.ROOT, "sequence_list.txt")
        if osp.exists(txt_path):
            with open(txt_path, 'r') as f:
                sequence_list = [line.strip() for line in f.readlines()]
        else:
            # Generate sequence list and save to txt            
            sequence_list = glob.glob(osp.join(self.ROOT, "*/*/*/rgb/*"))            
            sequence_list = [file_path.split(self.ROOT)[-1].lstrip('/') for file_path in sequence_list]
            sequence_list = sorted(sequence_list)

            # Save to txt file
            with open(txt_path, 'w') as f:
                f.write('\n'.join(sequence_list))
        
        
        self.sequence_list = sequence_list
        self.sequence_list_len = len(self.sequence_list)

        self.depth_max = 80

        if self.split =="train":
            pass
        else:
            print("invalid split, exit!")
            exit()
      
    def __len__(self):
        return self.sequence_list_len * 10000

    def _get_views(self, idx, resolution, rng):

        seq_index = random.randint(0, self.sequence_list_len - 1)
        seq_name = self.sequence_list[seq_index]
        camera_id = int(seq_name[-1])

        try:
            camera_parameters = np.loadtxt(
                osp.join(self.ROOT, "/".join(seq_name.split("/")[:2]), "extrinsic.txt"), 
                delimiter=" ", 
                skiprows=1
            )
            camera_parameters = camera_parameters[camera_parameters[:, 1] == camera_id]

            camera_intrinsic = np.loadtxt(
                osp.join(self.ROOT, "/".join(seq_name.split("/")[:2]), "intrinsic.txt"), 
                delimiter=" ", 
                skiprows=1
            )
            camera_intrinsic = camera_intrinsic[camera_intrinsic[:, 1] == camera_id]
        except Exception as e:
            print(f"Error loading camera parameters for {seq_name}: {e}")
            raise

        num_images = len(camera_parameters)
        max_distance = 100
        # ids = np.random.choice(num_images, 2, replace=False)
        id1 = np.random.randint(0, num_images)

        # Compute valid range for id2
        low = max(0, id1 - max_distance)
        high = min(num_images - 1, id1 + max_distance)

        # Randomly choose the second id within that range (excluding id1)
        possible_ids = [i for i in range(low, high + 1) if i != id1]
        id2 = int(np.random.choice(possible_ids))
        ids = [id1, id2]
        # ids = self.get_nearby_ids(ids, num_images, expand_ratio=8)

        views = []
        for image_idx in ids:
            label = f"rgb_{image_idx:05d}.jpg"
            image_filepath = osp.join(self.ROOT, seq_name, label)

            # load image
            input_rgb_image = imread_cv2(image_filepath)

            extri_opencv = camera_parameters[image_idx][2:].reshape(4, 4)

            intri_opencv = np.eye(3)
            intri_opencv[0, 0] = camera_intrinsic[image_idx][-4]
            intri_opencv[1, 1] = camera_intrinsic[image_idx][-3]
            intri_opencv[0, 2] = camera_intrinsic[image_idx][-2]
            intri_opencv[1, 2] = camera_intrinsic[image_idx][-1]
            K = intri_opencv
            pose = np.linalg.inv(extri_opencv)

            K = K.astype(np.float32)
            camera_pose = pose.astype(np.float32)
            
            rgb_image, intrinsics = self._crop_resize_if_necessary(
                input_rgb_image, K, resolution, rng=rng, info=image_filepath)

            views.append(dict(
            img=rgb_image,
            camera_pose=camera_pose,  # cam2world
            camera_intrinsics=intrinsics,
            dataset='VKitti',
            label=self.ROOT,
            instance=osp.join(seq_name, label)))


        return views