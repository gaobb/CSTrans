"""
Modified from:
'''
Test code written by Viresh Ranjan

Last modified by: Minh Hoai Nguyen (minhhoai@cs.stonybrook.edu)
Date: 2021/04/19
'''
"""

import copy
import os
import json
import torch
import argparse
import time
import torch.optim as optim
from tqdm import tqdm
from tools.utils import extract_features, return_cur_time_str
from tools.utils import MincountLoss, PerturbationLoss
from tools.utils import (
    create_folder,
    save_args,
    get_count_from_density_map,
    dist_init,
    load_train_args_from_json,
    str2bool,
    set_requires_grad,
)
from tools.vis import save_vis_test_results
from tools.train_eval_tools import (
    set_model,
    return_dataloader,
    update_count_dict,
    save_count_dict,
    cal_metrics_using_count_dict,
)
from tools.constants import PROJECT_DIR


def set_test_args():
    parser = argparse.ArgumentParser(description="Few Shot Counting Evaluation code")
    parser.add_argument(
        "--test_split",
        type=str,
        default="test",
        choices=["test", "val", "all", "test_coco", "val_coco"],
        help="what data split to evaluate on",
    )
    parser.add_argument(
        "--testing_gpu_mode",
        type=str,
        default="multiple_train_multiple_test",
        choices=[
            "multiple_train_multiple_test",
            "single_train_multiple_test",
            "multiple_train_single_test",
        ],
    )
    parser.add_argument(
        "--adapt",
        type=str2bool,
        default=False,
        help="whether to perform test time adaptation",
    )
    parser.add_argument(
        "--gradient_steps",
        type=int,
        default=100,
        help="number of gradient steps for the adaptation",
    )
    parser.add_argument(
        "--adaptation_learning_rate",
        type=float,
        default=1e-7,
        help="learning rate for adaptation",
    )
    parser.add_argument(
        "--weight_mincount",
        type=float,
        default=1e-9,
        help="weight multiplier for Mincount Loss",
    )
    parser.add_argument(
        "--weight_perturbation",
        type=float,
        default=1e-4,
        help="weight multiplier for Perturbation Loss",
    )
    parser.add_argument("--local_rank", default=-1, type=int, help="node rank for distributed training")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=f"{PROJECT_DIR}/outputs/xxx",
        help="output dir",
    )
    parser.add_argument(
        "--tebgn",
        type=int,
        default=None,
        help="gt bboxes number when testing, set an int less than 1 (e.g. -1) to do NOT limit gt bboxes number",
    )
    parser.add_argument("--box_group_index", type=int, default=0, help="box group index")
    args = parser.parse_args()

    args = load_train_args_from_json(args)
    args = dist_init(args)
    args.cur_time_str = return_cur_time_str()
    args.data_shuffle = False
    args.adapt_setting_str = "adapt" if args.adapt else "noadapt"
    args.testing_gt_bboxes_num = args.training_gt_bboxes_num
    args.random_shuffle_testing_gt_bboxes = False

    args.img_vis_dir = os.path.join(args.data_path, "img_vis")
    args.anno_file = os.path.join(args.data_path, "annotation_FSC-147.json")
    args.gt_den_map_dir = None
    args.gt_den_map_vis_dir = None

    if args.vis_corr_feat:
        # if args.fuse_init_feat, it is not proper to vis the corr feat since there are too much channels
        assert not args.fuse_init_feat
    assert os.path.exists(args.output_dir)
    print("args.regressor_ckpt_path: ", args.regressor_ckpt_path)
    assert os.path.exists(args.regressor_ckpt_path)
    assert os.path.exists(args.anno_file)
    assert os.path.exists(args.im_dir)

    if args.tebgn is not None:
        args.testing_gt_bboxes_num = args.tebgn
    args.test_result_file = os.path.join(
        args.output_dir,
        f"results_{args.test_split}_{args.adapt_setting_str}_tebgn={args.testing_gt_bboxes_num}_bgi={args.box_group_index}.txt",
    )

    args.save_vis_res = False
    args.vis_corr_feat = False
    args.save_fname_err_prefix = True
    args.save_fname_err_postfix = False

    assert not (args.save_fname_err_prefix and args.save_fname_err_postfix)

    return args


def test(args, test_dataloader, resnet50_conv, regressor):
    test_st_time = time.time()
    cnt = 0
    SAE = 0  # sum of absolute errors
    SSE = 0  # sum of square errors
    POS_ERR = 0
    NEG_ERR = 0
    count_dict = {}

    print("Evaluation on {} data".format(args.test_split))
    for image, dots, boxes, im_id, scale_factor in tqdm(test_dataloader):
        assert args.batch_size == 1
        im_id = im_id[0]
        scale_factor = scale_factor[0]

        with torch.no_grad():
            features = extract_features(args, resnet50_conv, image, boxes)
            if args.vis_corr_feat:
                corr_feat_for_vis = features.copy()
            else:
                corr_feat_for_vis = None

        if not args.adapt:
            with torch.no_grad():
                output = regressor(features)
        else:
            features = set_requires_grad(args, features, True)
            adapted_regressor = copy.deepcopy(regressor)
            adapted_regressor.train()
            optimizer = optim.Adam(adapted_regressor.parameters(), lr=args.adaptation_learning_rate)
            for _ in range(0, args.gradient_steps):
                optimizer.zero_grad()
                output = adapted_regressor(features)
                lCount = args.weight_mincount * MincountLoss(args, output, boxes.clone())
                lPerturbation = args.weight_perturbation * PerturbationLoss(args, output, boxes.clone(), sigma=8)
                Loss = lCount + lPerturbation
                # loss can become zero in some cases, where loss is a 0 valued scalar and not a tensor
                # So Perform gradient descent only for non zero cases
                if torch.is_tensor(Loss):
                    Loss.backward()
                    optimizer.step()
            features = set_requires_grad(args, features, False)
            output = adapted_regressor(features)

        gt_cnt = dots.shape[1]
        pred_cnt = get_count_from_density_map(args, output)
        cnt = cnt + 1
        err = abs(gt_cnt - pred_cnt)
        SAE += err
        SSE += err ** 2

        init_err = pred_cnt - gt_cnt
        if init_err > 0:
            POS_ERR += init_err
        elif init_err < 0:
            NEG_ERR += init_err

        update_count_dict(count_dict, im_id, gt_cnt, pred_cnt, with_loss=False)

        if args.save_vis_res:
            save_vis_test_results(
                args,
                im_id,
                output,
                gt_cnt,
                pred_cnt,
                run_mode="testing",
                vis_with_init_img=True,
                dots=dots,
                boxes=boxes,
                scale_factor=scale_factor,
                corr_feat_for_vis=corr_feat_for_vis,
            )

    save_json_file_dir = save_count_dict(args, count_dict, run_mode="testing", split=args.test_split)
    # aggregate and analyze the results
    if args.rank == 0:
        (
            acc_MAE,
            acc_RMSE,
            acc_POS_ERR,
            acc_NEG_ERR,
            all_sample_num,
        ) = cal_metrics_using_count_dict(save_json_file_dir, with_loss=False)
        test_fps = all_sample_num / (time.time() - test_st_time)
        with open(os.path.join(args.output_dir, "stats.json"), "r") as f:
            stats = json.load(f)
        print_str = "[best_epoch: {}] [cur_epoch: {}] [bv_mae: {:.2f}] [bv_rmse: {:.2f}] [{}] num_imgs: {}, MAE: {:.2f}, RMSE: {:.2f}, POS_ERR: {:.2f}, NEG_ERR: {:.2f}, test_fps: {:.2f}".format(
            stats["best_epoch"],
            stats["cur_epoch"],
            stats["best_mae"],
            stats["best_rmse"],
            args.test_split,
            int(all_sample_num),
            acc_MAE,
            acc_RMSE,
            acc_POS_ERR,
            acc_NEG_ERR,
            test_fps,
        )
        print(print_str)
        print("args.test_result_file: ", args.test_result_file)
        with open(args.test_result_file, "a") as f:
            f.write(print_str + "\n")


def main():
    args = set_test_args()
    if args.rank == 0:
        create_folder(args.output_dir)
        save_args(args, mode=args.test_split)
    resnet50_conv, regressor = set_model(args)
    test_dataloader, _ = return_dataloader(args, run_mode="testing")
    test(args, test_dataloader, resnet50_conv, regressor)


if __name__ == "__main__":
    main()
