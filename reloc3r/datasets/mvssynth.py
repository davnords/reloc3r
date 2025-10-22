import os.path as osp
import numpy as np
import json

from reloc3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from reloc3r.utils.image import imread_cv2
import random
from pathlib import Path

DATA_ROOT = "/mimer/NOBACKUP/groups/3d-dl/MVS-Synth/GTAV_540"  

def read_img_depth_pose(pose_path):
    with open(pose_path) as f:
        r_info = json.load(f)
        c_x = r_info["c_x"]
        c_y = r_info["c_y"]
        f_x = r_info["f_x"]
        f_y = r_info["f_y"]
        extrinsic = np.linalg.inv(np.array(r_info["extrinsic"]))
        # extrinsic = inv(extrinsic)
          
    # This is only for GTA 540
    f_x = f_x * 810 / 1920

    K = np.array([[f_x, 0, c_x], [0, f_y, c_y], [0,0,1]])
    return K, extrinsic

class MVSSynth(BaseStereoViewDataset):
    def __init__(self, *args, ROOT=DATA_ROOT, **kwargs):
        self.ROOT = ROOT
        super().__init__(*args, **kwargs)
        
        
        self.scenes = [p for p in Path(ROOT).iterdir() if p.is_dir()]
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
        
        frames = sorted([p.name for p in (scene_dir / "images").iterdir() if p.suffix == ".png"])
        
        # Select first frame randomly
        frame1 = random.choice(frames)
        frame1_number = int(Path(frame1).stem)

        # Compute valid range for frame2
        max_number = 99  # since the last frame is 0099.png
        upper_limit = min(frame1_number + 15, max_number)

        # Choose a random frame in [frame1_number + 1, upper_limit]
        if frame1_number < max_number:
            frame2_number = random.randint(frame1_number + 1, upper_limit)
        else:
            frame2_number = max_number  # if the first frame is already the last one

        # Convert back to filename with zero padding
        frame2 = f"{frame2_number:04d}.png"

        pose_path1 = scene_dir / "poses" / (frame1.replace(".png", ".json"))
        pose_path2 = scene_dir / "poses" / (frame2.replace(".png", ".json"))

        K1, pose1 = read_img_depth_pose(pose_path1)
        K2, pose2 = read_img_depth_pose(pose_path2)


        views = []

        groups = [(scene, frame1, pose1, K1), (scene, frame2, pose2, K2)]
        for group in groups:
            scene, label, pose, K = group

            impath = osp.join(self.ROOT, scene, "images", label)  

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
            dataset='MVSSynth',
            label=self.ROOT,
            instance=osp.join(scene, label)))

        return views