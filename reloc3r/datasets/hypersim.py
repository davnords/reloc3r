import os.path as osp
import numpy as np

from reloc3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from reloc3r.utils.image import imread_cv2
import random
from pathlib import Path
import torch
import pandas as pd
from glob import glob
import h5py
import cv2

DATA_ROOT = "/mimer/NOBACKUP/groups/3d-dl/ml-hypersim/contrib/99991/downloads"  


class Hypersim(BaseStereoViewDataset):
    def __init__(self, *args, ROOT=DATA_ROOT, **kwargs):
        self.ROOT = ROOT
        super().__init__(*args, **kwargs)

        data_root = Path("/mimer/NOBACKUP/groups/3d-dl/ml-hypersim/contrib/99991/downloads")
        metadata_camera_parameters_csv_file = (
            data_root / "metadata_camera_parameters.csv"
        )
        self.df_camera_parameters = pd.read_csv(
            metadata_camera_parameters_csv_file, index_col="scene_name"
        )
        scene_names = [p.name for p in Path(ROOT).iterdir() if p.is_dir()]

        valid_scene_names = []
        for name in scene_names:
            cam_dir = data_root / name / "_detail" / "cam_00"
            if cam_dir.is_dir():
                valid_scene_names.append(name)
        
        self.scene_names = valid_scene_names

        if self.split =="train":
            pass
        else:
            print("invalid split, exit!")
            exit()
      
    def __len__(self):
        return len(self.scene_names) * 10000

    def _get_views(self, idx, resolution, rng):
        scene_name = random.choice(self.scene_names)

        df_: pd.Series = self.df_camera_parameters.loc[scene_name]  # type: ignore
        width_pixels = int(df_["settings_output_img_width"])
        height_pixels = int(df_["settings_output_img_height"])

        M_proj = [
            [
                df_["M_proj_00"],
                df_["M_proj_01"],
                df_["M_proj_02"],
                df_["M_proj_03"],
            ],
            [
                df_["M_proj_10"],
                df_["M_proj_11"],
                df_["M_proj_12"],
                df_["M_proj_13"],
            ],
            [
                df_["M_proj_20"],
                df_["M_proj_21"],
                df_["M_proj_22"],
                df_["M_proj_23"],
            ],
            [
                df_["M_proj_30"],
                df_["M_proj_31"],
                df_["M_proj_32"],
                df_["M_proj_33"],
            ],
        ]
        M_proj = np.array(M_proj)
        M_screen_from_ndc = np.array(
            [
                [0.5 * (width_pixels), 0, 0, 0.5 * (width_pixels)],
                [0, -0.5 * (height_pixels), 0, 0.5 * (height_pixels)],
                [0, 0, 0.5, 0.5],  # doesn't matter
                [0, 0, 0, 1.0],
            ]
        )
        x = (M_screen_from_ndc @ M_proj)[[0, 1, 3]]
        K, R = cv2.decomposeProjectionMatrix(x)[:2]  # type: ignore
        K = K / K[2, 2]

        scene_root = Path(osp.join(self.ROOT, scene_name))
        camera_name = "cam_00"

        image_paths = sorted(
            glob(
                (
                    scene_root
                    / "images"
                    / f"scene_{camera_name}_final_preview"
                    / "frame.*.color.jpg"
                ).as_posix()
            )
        )
        

        camera_root = scene_root / "_detail" / camera_name
        camera_positions_hdf5_file = camera_root / "camera_keyframe_positions.hdf5"
        camera_orientations_hdf5_file = (
            camera_root / "camera_keyframe_orientations.hdf5"
        )

        with (
            h5py.File(camera_positions_hdf5_file, "r") as h5_pos,
            h5py.File(camera_orientations_hdf5_file, "r") as h5_rots,
        ):  # type: ignore
            camera_positions: np.ndarray = h5_pos["dataset"][:]  # type: ignore
            rots: np.ndarray = h5_rots["dataset"][:]  # type: ignore
            rots = rots.transpose((0, 2, 1))
            translations = -rots @ camera_positions[..., None]
            poses = np.zeros((len(rots), 4, 4))
            poses[:, 3, 3] = 1.0
            poses[:, :3, :3] = R[None] @ rots
            poses[:, :3, 3:] = R[None] @ translations

        K = np.array(K).reshape(3, 3).astype(np.float32)
        cx = K[0, 2]
        cy = K[1, 2]

        if cy > 768 or cx > 1024:
            return self._get_views(idx, resolution, rng)

        image_paths = {int(ip.split(".")[-3]): ip for ip in image_paths}
        image_ids = image_paths.keys()

        image_ids = sorted(image_ids)
        num_images = len(image_ids)
        if num_images < 2:
            raise ValueError(f"Not enough images in scene {scene_name}")

        max_distance = 65
        id1 = random.choice(image_ids)
        low = max(image_ids[0], id1 - max_distance)
        high = min(image_ids[-1], id1 + max_distance)

        # choose a second id within the valid window
        candidates = [i for i in image_ids if low <= i <= high and i != id1]
        if not candidates:
            return []  # no valid neighbor
        id2 = random.choice(candidates)

        selected_ids = [id1, id2]

        views = []
        for img_id in selected_ids:
            T = torch.tensor(poses[img_id]).float()
            # impath = Path(image_paths[img_id])
            impath = image_paths[img_id]
            # osp.join(self.ROOT, scene, "color", label)  
            input_rgb_image = imread_cv2(impath)
            rgb_image, intrinsics = self._crop_resize_if_necessary(
                input_rgb_image, K, resolution, rng=rng, info=impath)

            T_w2c = T.numpy()
            T_c2w = np.linalg.inv(T_w2c)

            views.append(dict(
            img=rgb_image,
            camera_pose=T_c2w,  # cam2world
            camera_intrinsics=intrinsics,
            dataset='Hypersim',
            label=self.ROOT,
            instance=osp.join(scene_name, impath.split('/')[-1])))
        return views