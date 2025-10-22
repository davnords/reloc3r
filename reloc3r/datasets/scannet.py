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
import torch

DATA_ROOT = "/mimer/NOBACKUP/groups/3d-dl/scannet/scans/scans_train"  

def read_scannet_pose(path):
    """ Read ScanNet's Camera2World pose and transform it to World2Camera.

    Returns:
        cam2world (np.ndarray): (4, 4)
    """
    cam2world = np.loadtxt(path, delimiter=' ')

    if not np.isfinite(cam2world).all():
        return None

    # world2cam = np.linalg.inv(cam2world)
    return cam2world

def read_scannet_intrinsic(path):
    """ Read ScanNet's intrinsic matrix and return the 3x3 matrix.
    """
    intrinsic = np.loadtxt(path, delimiter=' ')
    return intrinsic[:-1, :-1]

class ScanNet(BaseStereoViewDataset):
    def __init__(self, *args, ROOT=DATA_ROOT, **kwargs):
        self.ROOT = ROOT
        super().__init__(*args, **kwargs)
        
        
        self.scenes = list(Path(ROOT).iterdir())
        # os.listdir(self.ROOT)

        if self.split =="train":
            pass
        else:
            print("invalid split, exit!")
            exit()
      
    def __len__(self):
        return len(self.scenes) * 10000

    def _get_views(self, idx, resolution, rng):
        scene = random.choice(self.scenes)
        scene_dir = Path(osp.join(self.ROOT, scene))
        K = read_scannet_intrinsic(scene_dir / "intrinsic/intrinsic_color.txt")
        
        frames = sorted(
            [p.name for p in (scene_dir / "color").iterdir() if p.suffix == ".jpg"],
            key=lambda x: int(x.split('.')[0])
        )
        
        # Select first frame randomly
        frame1 = random.choice(frames)
        frame1_number = int(frame1.split('.')[0])
        
        # Define the range for frame2 (within 500 units)
        min_frame2_number = frame1_number + 10  # or frame1_number + 10 if you don't want the same frame
        max_frame2_number = frame1_number + 500
        
        # Filter frames within the valid range
        valid_frames = [
            f for f in frames 
            if min_frame2_number <= int(f.split('.')[0]) <= max_frame2_number
        ]
        
        # If no valid frames (shouldn't happen), fall back to all frames
        if not valid_frames:
            valid_frames = frames
        
        # Select second frame from valid options
        frame2 = random.choice(valid_frames)
        frame2_number = int(frame2.split('.')[0])

        pose1, pose2 = read_scannet_pose(scene_dir / "pose" / (frame1.replace(".jpg", ".txt"))), read_scannet_pose(scene_dir / "pose" / (frame2.replace(".jpg", ".txt")))
        
        if pose1 is None or pose2 is None:
            return self._get_views(idx, resolution, rng)

        views = []

        groups = [(scene, frame1, pose1), (scene, frame2, pose2)]
        for group in groups:
            scene, label, pose = group

            impath = osp.join(self.ROOT, scene, "color", label)  

            # load image
            input_rgb_image = imread_cv2(impath)
            K = K.astype(np.float32)
            camera_pose = pose.astype(np.float32)
            
            rgb_image, intrinsics = self._crop_resize_if_necessary(
                input_rgb_image, K, resolution, rng=rng, info=impath)

            views.append(dict(
            img=rgb_image,
            camera_pose=camera_pose,  # cam2world
            camera_intrinsics=intrinsics,
            dataset='ScanNet',
            label=self.ROOT,
            instance=osp.join(scene, label)))

        return views