import os.path as osp
import numpy as np
import json
import itertools
from collections import deque
import os

from reloc3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from reloc3r.utils.image import imread_cv2
import random
from pathlib import Path
import re

DATA_ROOT = "/mimer/NOBACKUP/groups/3d-dl/wildrgbd"  

CATEGORIES = [
    "apple",
    "backpack",
    "ball",
    "banana",
    "boat",
    "book",
    "bottle",
    "bowl",
    "box",
    "bucket",
    "bus",
    "cake",
    "car",
    "carrot",
    "cellphone",
    "chair",
    "plane",
    "TV",
]

def load_cam_poses(path):
    poses = []
    with open(path, "r") as f:
        for line in f:
            tokens = line.strip().split()
            frame_id = int(tokens[0])
            mat = np.array([float(x) for x in tokens[1:]]).reshape(4, 4)
            poses.append((frame_id, mat))
    return poses

class WildRGBD(BaseStereoViewDataset):
    def __init__(self, mask_bg=False, *args, ROOT=DATA_ROOT, **kwargs):
        self.ROOT = ROOT
        super().__init__(*args, **kwargs)
        assert mask_bg in (True, False, 'rand')
        self.mask_bg = mask_bg


        invalid_sequences = [
            "chair/scene_490"
        ]
        scenes = {}
        for category in CATEGORIES: 
            scenes[category] = os.listdir(osp.join(self.ROOT,category, "scenes"))
            scenes[category] = [s for s in scenes[category] if f"{category}/{s}" not in invalid_sequences]
        self.scenes = scenes
        # os.listdir(self.ROOT)

        if self.split =="train":
            pass
        else:
            print("invalid split, exit!")
            exit()
      
    def __len__(self):
        return len(CATEGORIES) * 10000

    def _get_views(self, idx, resolution, rng):
        category = random.choice(CATEGORIES)
        scenes = self.scenes[category]
        scene = random.choice(scenes)
        scene_dir = Path(osp.join(self.ROOT, category, "scenes", scene))

        poses = load_cam_poses(scene_dir / "cam_poses.txt")

        with open(scene_dir / "metadata", "r") as f:
            meta = json.load(f)

        K_flat = meta["K"]  # list of 9 numbers
        K = np.array(K_flat).reshape(3, 3).T

        frames = sorted([p.name for p in (scene_dir / "rgb").iterdir() if p.suffix == ".png"])

        frame1, frame2 = random.sample(frames, 2)
        frame_number1,frame_number2 = int(re.search(r'\d+', frame1).group()), int(re.search(r'\d+', frame2).group())
        
        _, pose1 = poses[frame_number1]
        _, pose2 = poses[frame_number2]

        views = []

        groups = [(scene, frame1, pose1), (scene, frame2, pose2)]
        for group in groups:
            scene, label, pose = group

            impath = osp.join(self.ROOT, category, "scenes", scene, "rgb", label)  

            # load image
            input_rgb_image = imread_cv2(impath)
            intrinsics = K.astype(np.float32)
            camera_pose = pose.astype(np.float32)
        
            
            rgb_image, intrinsics = self._crop_resize_if_necessary(
                input_rgb_image, intrinsics, resolution, rng=rng, info=impath)

            views.append(dict(
            img=rgb_image,
            camera_pose=camera_pose,  # cam2world
            camera_intrinsics=intrinsics,
            dataset='WildRGBD',
            label=self.ROOT,
            instance=osp.join(scene, label)))

        return views