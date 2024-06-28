"""
Modified from:
'''
Training Code for Learning To Count Everything, CVPR 2021
Authors: Viresh Ranjan,Udbhav, Thu Nguyen, Minh Hoai

Last modified by: Minh Hoai Nguyen (minhhoai@cs.stonybrook.edu)
Date: 2021/04/19
'''
"""

import os
import torch
import argparse
import time
import json
from tqdm import tqdm
from tools.utils import (
    extract_features,
    return_cur_time_str,
    create_folder,
    save_args,
    get_count_from_density_map,
    dist_init,
    str2bool,
    int_or_None,
    load_train_args_from_json,
    set_requires_grad,
    set_random_seed,
    write_nvidia_info,
)
from tools.train_eval_tools import (
    set_criterion,
    set_model_and_optim,
    return_dataloader,
    return_loss,
    init_stats,
    update_stats,
    plot_stats,
    save_stats,
    save_log,
    save_model,
    update_count_dict,
    save_count_dict,
    cal_metrics_using_count_dict,
)
from tools.vis import save_vis_test_results
from tools.constants import PROJECT_DIR


def set_train_args():
    parser = argparse.ArgumentParser(description="Few Shot Counting Training code")
    parser.add_argument("--pretrained", type=str2bool, default=True, help="use pretrianed model")
    parser.add_argument("--fast_train", type=str2bool, default=True, help="fast train")

    parser.add_argument(
        "--save_vis_res",
        type=str2bool,
        default=False,
        help="whether to save vis test results",
    )
    parser.add_argument("--vis_corr_feat", type=str2bool, default=False, help="whether to vis corr feat")
    parser.add_argument(
        "--save_fname_err_prefix",
        type=str2bool,
        default=False,
        help="whether to add err info as prefix when saving vis results",
    )
    parser.add_argument(
        "--save_fname_err_postfix",
        type=str2bool,
        default=True,
        help="whether to add err info as postfix when saving vis results",
    )

    parser.add_argument("--val_loss", type=str2bool, default=False, help="calculate validation loss")
    parser.add_argument(
        "--metric_for_best",
        type=str,
        default="mae",
        choices=["mae", "rmse", "val_loss"],
        help="choose which metric to decide the best epoch",
    )
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--local_rank", default=-1, type=int, help="node rank for distributed training")
    parser.add_argument("--backbone", type=str, default="resnet50", help="backbone")
    parser.add_argument(
        "--torch_home",
        type=str,
        default=f"{PROJECT_DIR}/torchvision_pretrained_models",
        help="torch home",
    )
    parser.add_argument(
        "--resume",
        type=str2bool,
        default=True,
        help="whether to resume if the running had ever been stopped",
    )
    parser.add_argument("--inplace_flag", type=str2bool, default=True, help="inplace")
    parser.add_argument(
        "--multi_scale_exemplar_feature",
        type=str2bool,
        default=True,
        help="whether to use multi-scale exemplar feature",
    )
    parser.add_argument(
        "--support_feat_resize",
        type=str2bool,
        default=True,
        help="whether to resize the examplar bbox feature",
    )
    parser.add_argument(
        "--corr_feat_keep_method",
        type=str,
        default="query_feat_pad",
        choices=["query_feat_pad", "corr_feat_resize"],
        help="the method to keep the spatial dimension of the corr feat",
    )
    parser.add_argument(
        "--random_shuffle_training_gt_bboxes",
        type=str2bool,
        default=False,
        help="whether to random shuffle the training exemplar bboxes",
    )
    parser.add_argument(
        "--trbgn",
        type=int,
        default=None,
        help="gt bboxes number when training, set an int less than 1 (e.g. -1) to do NOT limit gt bboxes number",
    )
    parser.add_argument(
        "--fuse_init_feat",
        type=str2bool,
        default=True,
        help="whether to fuse the init feat with the corr_feat",
    )
    parser.add_argument(
        "--fuse_type",
        type=str,
        default="mul",
        choices=["mul", "add", "None"],
        help="fuse type",
    )

    parser.add_argument(
        "--regressor_in_prepro_type",
        type=str,
        default="conv1x1",
        choices=["conv1x1", "none"],
        help="regressor input prep processing type",
    )
    parser.add_argument(
        "--regressor_in_prepro_out_chans",
        type=int_or_None,
        default=32,
        help="regressor input prep processing out chans",
    )
    parser.add_argument(
        "--regressor_type",
        type=str,
        default="swt",
        choices=["conv", "swt"],
        help="regressor type",
    )

    # Arguments for Swin Transformer Decoder
    parser.add_argument(
        "--swint_regressor_patch_size",
        type=int_or_None,
        default=2,
        help="swint regressor patch size",
    )
    parser.add_argument(
        "--swint_regressor_embed_dim",
        type=int_or_None,
        default=128,
        help="swint regressor embed dim",
    )
    parser.add_argument(
        "--swint_regressor_depths",
        type=int_or_None,
        default=[2, 2, 2, 2, 2],
        nargs="+",
        help="swint regressor depths",
    )
    parser.add_argument(
        "--swint_regressor_num_heads",
        type=int_or_None,
        default=[8, 8, 4, 4, 2],
        nargs="+",
        help="swint regressor num heads",
    )
    parser.add_argument(
        "--swint_regressor_window_size",
        type=int_or_None,
        default=7,
        help="swint regressor window size",
    )
    parser.add_argument(
        "--use_swmsa_in_encoder",
        type=str2bool,
        default=True,
        help="whether to use sw-msa in encoder",
    )
    parser.add_argument(
        "--use_swmsa_in_decoder",
        type=str2bool,
        default=False,
        help="whether to use sw-msa in decoder",
    )
    parser.add_argument(
        "--use_attn_mask_in_encoder_swmsa",
        type=str2bool,
        default=True,
        help="whether to use attn mask in basiclayer in sw-msa",
    )
    parser.add_argument(
        "--use_attn_mask_in_decoder_swmsa",
        type=str2bool,
        default=False,
        help="whether to use attn mask in basiclayer_up in sw-msa",
    )
    parser.add_argument(
        "--norm_for_each_out_feat",
        type=str2bool,
        default=False,
        help="whether to use norm layer for each out feat of each layer",
    )

    args = parser.parse_args()

    args = dist_init(args)
    args.learning_rate = 1e-5
    if args.num_nodes != 1:
        args.learning_rate *= args.num_nodes

    if not args.use_swmsa_in_encoder:
        args.use_attn_mask_in_encoder_swmsa = False
    if not args.use_swmsa_in_decoder:
        args.use_attn_mask_in_decoder_swmsa = False

    if args.metric_for_best == "val_loss":
        args.val_loss = True
    args.density_map_scale = 100.0
    args.regressor_output_channels = 1
    args.data_path = os.path.join(PROJECT_DIR, "datasets/FSC-147")
    args.data_split_file = os.path.join(args.data_path, "Train_Test_Val_FSC-147.json")
    args.im_dir = os.path.join(args.data_path, "img")
    args.training_gt_bboxes_num = 1
    args.density_map_type = "adaptive"
    args.val_split = "val"
    args.gt_den_map_dir = os.path.join(args.data_path, f"gt_density_map_{args.density_map_type}")
    args.gt_den_map_vis_dir = os.path.join(args.data_path, f"gt_density_map_{args.density_map_type}_vis")
    args.MAPS = ["map3", "map4"]  # notice: from bottom to high

    if "resnet" in args.backbone:
        args.map2nchannels = {"map2": 256, "map3": 512, "map4": 1024}
    else:
        raise  # currently not supported

    if args.multi_scale_exemplar_feature:
        args.Scales = [0.9, 1.1]
        args.num_scales = len(args.Scales) + 1
    else:
        args.Scales = None
        args.num_scales = 1
    if args.fuse_init_feat:
        args.epochs = 1500
        args.regressor_input_channels = 0
        for cur_map in args.MAPS:
            args.regressor_input_channels += args.map2nchannels[cur_map] * args.num_scales
    else:
        args.epochs = 1500
        args.regressor_input_channels = len(args.MAPS) * args.num_scales

    if args.regressor_type == "conv":
        regressor_config_str = "rt=con"
    elif args.regressor_type == "swt":
        regressor_config_str = "rt={}_p{}_e{}_d{}_n{}_w{}_{}{}{}{}".format(
            args.regressor_type[0:3],
            args.swint_regressor_patch_size,
            args.swint_regressor_embed_dim,
            "".join([str(name) for name in args.swint_regressor_depths]),
            "".join([str(name) for name in args.swint_regressor_num_heads]),
            args.swint_regressor_window_size,
            str(args.use_swmsa_in_encoder)[0],
            str(args.use_swmsa_in_decoder)[0],
            str(args.use_attn_mask_in_encoder_swmsa)[0],
            str(args.use_attn_mask_in_decoder_swmsa)[0],
        )

    if args.density_map_type == "adaptive":
        density_map_type_str = "a"
    elif "fixsig" in args.density_map_type:
        density_map_type_str = args.density_map_type.lstrip("fixsig")
    else:
        raise  # currently not supported

    if args.regressor_in_prepro_type == "none":
        ds_str = "noprep"
    elif args.regressor_in_prepro_type == "conv1x1":
        ds_str = f"c1_o{args.regressor_in_prepro_out_chans}"
    else:
        raise

    if args.trbgn is not None:
        args.training_gt_bboxes_num = (
            args.trbgn
        )  # if args.trbgn is not None, args.trbgn --> args.training_gt_bboxes_num

    args.output_dir = os.path.join(
        PROJECT_DIR,
        "outputs/d={}_{}_{}_f={}_t={}_n={}".format(
            density_map_type_str,
            ds_str,
            regressor_config_str,
            str(args.fuse_init_feat)[0],
            args.fuse_type[0],
            args.training_gt_bboxes_num,
        ),
    )
    args.anno_file = os.path.join(args.data_path, "annotation_FSC-147.json")

    args.regressor_ckpt_path = os.path.join(args.output_dir, "regressor.pth")
    args.stats_path = os.path.join(args.output_dir, "stats.json")
    args.data_shuffle = True
    args.pin_memory = False  # True
    args.non_blocking = False  # True
    assert args.pin_memory == args.non_blocking
    assert args.batch_size == 1
    args.adapt_setting_str = "noadapt"
    args.MIN_HW = 384
    args.MAX_HW = 1584
    args.IM_NORM_MEAN = [0.485, 0.456, 0.406]
    args.IM_NORM_STD = [0.229, 0.224, 0.225]
    cur_time_str = return_cur_time_str()
    args.resume_history_dict = {"{}".format(cur_time_str): "begin (first time)"}
    if args.rank == 0:
        write_nvidia_info(args, cur_time_str)

    assert not (args.save_fname_err_prefix and args.save_fname_err_postfix)

    return args


def forward_and_cal_loss(args, resnet50_conv, image, boxes, im_id, regressor, gt_density, criterion):
    with torch.no_grad():
        features = extract_features(args, resnet50_conv, image, boxes)
    features = set_requires_grad(args, features, True)
    output = regressor(features)
    loss = return_loss(args, output, gt_density, criterion)
    return output, loss


def train(
    args,
    epoch,
    train_dataloader,
    train_sampler,
    resnet50_conv,
    regressor,
    criterion,
    optimizer,
):
    train_st_time = time.time()
    if not args.fast_train:
        train_loss = 0
        count_dict = {}
    print("Training on train set data")
    for image, dots, boxes, im_id, gt_density, _ in train_dataloader:
        train_sampler.set_epoch(epoch)
        assert args.batch_size == 1
        im_id = im_id[0]
        optimizer.zero_grad()
        output, loss = forward_and_cal_loss(
            args, resnet50_conv, image, boxes, im_id, regressor, gt_density, criterion
        )
        loss.backward()
        optimizer.step()

        if not args.fast_train:
            train_loss += loss.item()
            pred_cnt = get_count_from_density_map(args, output)
            gt_cnt = dots.shape[1]
            update_count_dict(count_dict, im_id, gt_cnt, pred_cnt, with_loss=True, loss=train_loss)

    if not args.fast_train:
        save_json_file_dir = save_count_dict(args, count_dict, run_mode="training", split="train")
        acc_MAE, acc_RMSE, _, _, num_imgs, acc_loss = cal_metrics_using_count_dict(save_json_file_dir, with_loss=True)
        if args.rank == 0:
            train_fps = num_imgs / (time.time() - train_st_time)
            print(
                "[train] Epoch: {}, On train data, num_imgs: {}, MAE: {:6.2f}, RMSE: {:6.2f}, loss: {:.10f}, train_fps: {:.2f}".format(
                    epoch, num_imgs, acc_MAE, acc_RMSE, acc_loss, train_fps
                )
            )
        return acc_loss, acc_MAE, acc_RMSE
    else:
        return 0.0, 0.0, 0.0


def eval(
    args,
    epoch,
    val_dataloader,
    val_sampler,
    resnet50_conv,
    regressor,
    criterion,
    with_loss=True,
):
    eval_st_time = time.time()
    if with_loss:
        val_loss = 0
    count_dict = {}
    print("Evaluating on {} data".format(args.val_split))
    for sample in val_dataloader:
        val_sampler.set_epoch(epoch)
        if with_loss:
            image, dots, boxes, im_id, gt_density, scale_factor = sample
        else:
            image, dots, boxes, im_id, scale_factor = sample
        assert args.batch_size == 1
        im_id = im_id[0]
        scale_factor = scale_factor[0]

        with torch.no_grad():
            features = extract_features(args, resnet50_conv, image, boxes)
            if args.vis_corr_feat:
                corr_feat_for_vis = features.copy()
            else:
                corr_feat_for_vis = None
            output = regressor(features)

        if with_loss:
            loss = return_loss(args, output, gt_density, criterion)
            val_loss += loss.item()

        gt_cnt = dots.shape[1]
        pred_cnt = get_count_from_density_map(args, output)

        if with_loss:
            update_count_dict(count_dict, im_id, gt_cnt, pred_cnt, with_loss=True, loss=val_loss)
        else:
            update_count_dict(count_dict, im_id, gt_cnt, pred_cnt, with_loss=False, loss=None)

        if args.save_vis_res:
            save_vis_test_results(
                args,
                im_id,
                output,
                gt_cnt,
                pred_cnt,
                run_mode="training",
                vis_with_init_img=True,
                dots=dots,
                boxes=boxes,
                scale_factor=scale_factor,
                corr_feat_for_vis=corr_feat_for_vis,
                epoch=epoch,
            )

    save_json_file_dir = save_count_dict(args, count_dict, run_mode="training", split=args.val_split)
    if with_loss:
        acc_MAE, acc_RMSE, _, _, num_imgs, acc_loss = cal_metrics_using_count_dict(save_json_file_dir, with_loss=True)
        if args.rank == 0:
            eval_fps = num_imgs / (time.time() - eval_st_time)
            print(
                "[eval] Epoch: {}, On {} data, num_imgs: {}, MAE: {:6.2f}, RMSE: {:6.2f}, loss: {:.10f}, eval_fps: {:.2f}".format(
                    epoch,
                    args.val_split,
                    num_imgs,
                    acc_MAE,
                    acc_RMSE,
                    acc_loss,
                    eval_fps,
                )
            )
        return acc_loss, acc_MAE, acc_RMSE
    else:
        acc_MAE, acc_RMSE, _, _, num_imgs = cal_metrics_using_count_dict(save_json_file_dir, with_loss=False)
        if args.rank == 0:
            eval_fps = num_imgs / (time.time() - eval_st_time)
            print(
                "[eval] Epoch: {}, On {} data, num_imgs: {}, MAE: {:6.2f}, RMSE: {:6.2f}, eval_fps: {:.2f}".format(
                    epoch, args.val_split, num_imgs, acc_MAE, acc_RMSE, eval_fps
                )
            )
        return acc_MAE, acc_RMSE


def main():
    set_random_seed(2022)
    args = set_train_args()
    if args.resume and os.path.exists(args.regressor_ckpt_path) and os.path.exists(args.stats_path):
        args = load_train_args_from_json(args)  # load args.resume_history_dict
        with open(args.stats_path, "r") as f:  # load the whole stats information
            stats = json.load(f)
        cur_time_str = return_cur_time_str()
        args.resume_history_dict["{}".format(cur_time_str)] = "resume from epoch {}".format(stats["cur_epoch"])
        if args.rank == 0:
            write_nvidia_info(args, cur_time_str)
        start_epoch = stats["cur_epoch"] + 1
        if start_epoch == (args.epochs - 1):
            return
        args.make_sure_to_resume = True
    else:
        # only need to init the stats of rank 0
        if args.rank == 0:
            stats = init_stats(args)
        start_epoch = 0
        args.make_sure_to_resume = False

    criterion = set_criterion(args)
    resnet50_conv, regressor, optimizer = set_model_and_optim(
        args, backbone_mode="eval", regressor_mode="train", return_optim=True
    )
    train_dataloader, train_sampler, val_dataloader, val_sampler = return_dataloader(args, run_mode="training")

    if args.rank == 0:
        print("args.output_dir: ", args.output_dir)
        create_folder(args.output_dir)
        save_args(args, mode="train")

    st_time = time.time()
    for epoch in tqdm(range(start_epoch, args.epochs)):
        regressor.train()
        train_loss, train_mae, train_rmse = train(
            args,
            epoch,
            train_dataloader,
            train_sampler,
            resnet50_conv,
            regressor,
            criterion,
            optimizer,
        )
        regressor.eval()
        if args.val_loss:
            val_loss, val_mae, val_rmse = eval(
                args,
                epoch,
                val_dataloader,
                val_sampler,
                resnet50_conv,
                regressor,
                criterion,
                with_loss=True,
            )
        else:
            val_mae, val_rmse = eval(
                args,
                epoch,
                val_dataloader,
                val_sampler,
                resnet50_conv,
                regressor,
                criterion,
                with_loss=False,
            )

        if args.rank == 0:
            if args.val_loss:
                stats = update_stats(
                    args,
                    stats,
                    epoch,
                    train_mae,
                    train_rmse,
                    val_mae,
                    val_rmse,
                    train_loss,
                    val_loss,
                )
            else:
                stats = update_stats(
                    args,
                    stats,
                    epoch,
                    train_mae,
                    train_rmse,
                    val_mae,
                    val_rmse,
                    train_loss,
                )

            # determine whether it is the best
            if args.metric_for_best == "mae":
                best_flag = val_mae <= stats["best_mae"]
            elif args.metric_for_best == "rmse":
                best_flag = val_rmse <= stats["best_rmse"]
            elif args.metric_for_best == "val_loss":
                best_flag = val_loss <= stats["best_val_loss"]
            if best_flag:
                stats["best_mae"] = val_mae
                stats["best_rmse"] = val_rmse
                if args.val_loss:
                    stats["best_val_loss"] = val_loss
                stats["best_epoch"] = epoch
                save_model(args, regressor)

            plot_stats(args, stats)
            save_stats(args, stats)
            print_log = "[{}] tr_mae={:.2f} tr_rmse={:.2f} val_mae={:.2f} val_rmse={:.2f} best_val_mae={:.2f} best_val_rmse={:.2f}".format(
                stats["epoch_list"][-1],
                stats["train_mae"][-1],
                stats["train_rmse"][-1],
                stats["val_mae"][-1],
                stats["val_rmse"][-1],
                stats["best_mae"],
                stats["best_rmse"],
            )
            print(print_log)
            save_log(args, print_log, st_time, start_epoch, epoch)


if __name__ == "__main__":
    main()
