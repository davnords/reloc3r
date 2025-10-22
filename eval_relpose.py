import argparse
import os
import numpy as np
import torch
torch.backends.cuda.matmul.allow_tf32 = True  # for gpu >= Ampere and pytorch >= 1.12

from reloc3r.relpose_transformer import RelposeTransformer
from reloc3r.reloc3r_relpose import Reloc3rRelpose, setup_reloc3r_relpose_model, inference_relpose
from reloc3r.datasets import get_data_loader
from reloc3r.utils.metric import *
from reloc3r.utils.device import to_numpy

from tqdm import tqdm
# from pdb import set_trace as bb


# CUDA_VISIBLE_DEVICES=2 python eval_relpose.py --model "RelposeTransformer(vit='dinov3')" --test_dataset "MegaDepth_valid(resolution=(512,384), seed=777)" --ckpt /mimer/NOBACKUP/groups/snic2022-6-266/davnords/reloc3r/output_dir/dinov3_256/5231156/checkpoint-80.pth --amp 1

# DINOv3:
# * {'auc@5': np.float64(0.023891542132695516), 'auc@10': np.float64(0.11034403356711069), 'auc@20': np.float64(0.27244366201162334)}
# MuM: 
# * {'auc@5': np.float64(0.08492012865543365), 'auc@10': np.float64(0.24579739037752152), 'auc@20': np.float64(0.45101412255565326)}
# CroCov2:
# {'auc@5': np.float64(0.0475976287206014), 'auc@10': np.float64(0.15792263940175374), 'auc@20': np.float64(0.3345313930988311)}

# Finetuning
# MuM:
# * {'auc@5': np.float64(0.16652185324033103), 'auc@10': np.float64(0.3517917633295059), 'auc@20': np.float64(0.5532298429290453)}

# CroCov2:
# * {'auc@5': np.float64(0.07672052969932555), 'auc@10': np.float64(0.20933479200204216), 'auc@20': np.float64(0.39184619931777315)}

def get_args_parser():
    parser = argparse.ArgumentParser(description='evaluation code for relative camera pose estimation')

    # model
    parser.add_argument('--model', type=str, 
        # default='Reloc3rRelpose(img_size=224)')
        default='Reloc3rRelpose(img_size=512)')
    
    # test set
    parser.add_argument('--test_dataset', type=str, 
        # default="ScanNet1500(resolution=(224,224), seed=777)")
        default="ScanNet1500(resolution=(512,384), seed=777)")
    parser.add_argument('--ckpt', type=str, required=True, 
        help='path to the trained model checkpoint')
    parser.add_argument('--batch_size', type=int,
        default=1)
    parser.add_argument('--num_workers', type=int,
        default=10)
    parser.add_argument('--amp', type=int, default=1,
                                choices=[0, 1], help="Use Automatic Mixed Precision for pretraining")

    # parser.add_argument('--output_dir', type=str, 
    #     default='./output', help='path where to save the pose errors')

    return parser


def build_dataset(dataset, batch_size, num_workers, test=False):
    split = ['Train', 'Test'][test]
    print('Building {} data loader for {}'.format(split, dataset))
    loader = get_data_loader(dataset,
                             batch_size=batch_size,
                             num_workers=num_workers,
                             pin_mem=True,
                             shuffle=not (test),
                             drop_last=not (test))
    print('Dataset length: ', len(loader))
    return loader


def test(args):
    
    # if not os.path.exists(args.output_dir):
    #     os.makedirs(args.output_dir)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    print('Loading model: {:s}'.format(args.model))
    reloc3r_relpose = eval(args.model)

    ckpt = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    print('Loading checkpoint from {:s}'.format(args.ckpt))
    print(reloc3r_relpose.load_state_dict(ckpt['model'], strict=True))

    reloc3r_relpose.to(device)
    reloc3r_relpose.eval()
    
    data_loader_test = {dataset.split('(')[0]: build_dataset(dataset, args.batch_size, args.num_workers, test=True)
                        for dataset in args.test_dataset.split('+')}

    # start evaluation
    rerrs, terrs = [], []
    for test_name, testset in data_loader_test.items():
        print('Testing {:s}'.format(test_name))
        with torch.no_grad():
            for batch in tqdm(testset):

                pose = inference_relpose(batch, reloc3r_relpose, device, use_amp=bool(args.amp))

                view1, view2 = batch
                gt_pose2to1 = torch.inverse(view1['camera_pose']) @ view2['camera_pose']
                rerrs_prh = []
                terrs_prh = []

                # rotation angular err
                R_prd = pose[:,0:3,0:3]
                for sid in range(len(R_prd)):
                    rerrs_prh.append(get_rot_err(to_numpy(R_prd[sid]), to_numpy(gt_pose2to1[sid,0:3,0:3])))
                
                # translation direction angular err
                t_prd = pose[:,0:3,3]
                for sid in range(len(t_prd)): 
                    transl = to_numpy(t_prd[sid])
                    gt_transl = to_numpy(gt_pose2to1[sid,0:3,-1])
                    transl_dir = transl / np.linalg.norm(transl)
                    gt_transl_dir = gt_transl / np.linalg.norm(gt_transl)
                    terrs_prh.append(get_transl_ang_err(transl_dir, gt_transl_dir)) 

                rerrs += rerrs_prh
                terrs += terrs_prh

        rerrs = np.array(rerrs)
        terrs = np.array(terrs)
        print('In total {} pairs'.format(len(rerrs)))

        # auc
        print(error_auc(rerrs, terrs, thresholds=[5, 10, 20]))

        # # save err list to file
        # err_list = np.concatenate((rerrs[:,None], terrs[:,None]), axis=-1)
        # output_file = '{}/pose_error_list.txt'.format(args.output_dir)
        # np.savetxt(output_file, err_list)
        # print('Pose errors saved to {}'.format(output_file))


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    test(args)

