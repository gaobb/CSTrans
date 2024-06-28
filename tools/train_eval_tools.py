import os
import json
import time
import datetime
import numpy as np
import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from collections import OrderedDict
from datasets.FSC147 import FSC147Dataset
from tools.utils import create_folder
from models.model import Resnet50FPN, CountRegressor, weights_normal_init
from models.swint import SwinTransformerRegressor


def set_criterion(args):
    criterion = nn.MSELoss().cuda()
    return criterion


def dist_model(args, model):
    device = torch.device("cuda")
    model = model.to(device, non_blocking=args.non_blocking)
    if args.word_size > 1:
        print(f"Using {args.word_size} GPUs!")
        model = DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)
    return model


def set_model_and_optim(args, backbone_mode="eval", regressor_mode="train", return_optim=True):
    # ########################################
    # feature extractor
    # ########################################
    if "resnet" in args.backbone:
        resnet50_conv = Resnet50FPN(args, pretrained=args.pretrained)
    else:
        raise  # currently not supported
    if backbone_mode == "train":
        resnet50_conv.train()
    elif backbone_mode == "eval":
        resnet50_conv.eval()
    else:
        raise
    resnet50_conv = dist_model(args, resnet50_conv)

    # ########################################
    # regressor
    # ########################################
    if args.regressor_type == "conv":
        regressor = CountRegressor(args, pool="mean")
    elif args.regressor_type == "swt":
        regressor = SwinTransformerRegressor(
            args,
            in_chans=args.regressor_input_channels,
            patch_size=args.swint_regressor_patch_size,
            embed_dim=args.swint_regressor_embed_dim,
            depths=args.swint_regressor_depths,
            num_heads=args.swint_regressor_num_heads,
            window_size=args.swint_regressor_window_size,
            pool="mean",
        )

    if regressor_mode == "train":
        if not args.make_sure_to_resume:
            if args.regressor_type == "conv":
                weights_normal_init(regressor, dev=0.001)
        regressor.train()
    elif regressor_mode == "eval":
        regressor.eval()
    else:
        raise
    regressor = dist_model(args, regressor)
    if args.make_sure_to_resume:
        # configure map_location properly
        map_location = {"cuda:%d" % 0: "cuda:%d" % args.local_rank}
        regressor.load_state_dict(torch.load(args.regressor_ckpt_path, map_location=map_location))

    # ########################################
    # optim
    # ########################################
    if return_optim:
        optimizer = optim.Adam(regressor.parameters(), lr=args.learning_rate)
        return resnet50_conv, regressor, optimizer
    else:
        return resnet50_conv, regressor


def set_model(args):
    # ########################################
    # feature extractor
    # ########################################
    if "resnet" in args.backbone:
        resnet50_conv = Resnet50FPN(args, pretrained=args.pretrained)
    else:
        raise  # currently not supported
    resnet50_conv.eval()
    resnet50_conv = dist_model(args, resnet50_conv)

    # ########################################
    # regressor
    # ########################################
    if args.regressor_type == "conv":
        regressor = CountRegressor(args, pool="mean")
    elif args.regressor_type == "swt":
        regressor = SwinTransformerRegressor(
            args,
            in_chans=args.regressor_input_channels,
            patch_size=args.swint_regressor_patch_size,
            embed_dim=args.swint_regressor_embed_dim,
            depths=args.swint_regressor_depths,
            num_heads=args.swint_regressor_num_heads,
            window_size=args.swint_regressor_window_size,
            pool="mean",
        )
    regressor.eval()

    if args.testing_gpu_mode == "multiple_train_multiple_test":
        regressor = dist_model(args, regressor)
        # configure map_location properly
        map_location = {"cuda:%d" % 0: "cuda:%d" % args.local_rank}
        regressor.load_state_dict(torch.load(args.regressor_ckpt_path, map_location=map_location))
    elif args.testing_gpu_mode == "single_train_multiple_test":
        # configure map_location properly
        map_location = {"cuda:%d" % 0: "cuda:%d" % args.local_rank}
        regressor.load_state_dict(torch.load(args.regressor_ckpt_path, map_location=map_location))
        regressor = dist_model(args, regressor)
    elif args.testing_gpu_mode == "multiple_train_single_test":
        regressor = dist_model(args, regressor)
        # configure map_location properly
        map_location = {"cuda:%d" % 0: "cuda:%d" % args.local_rank}
        state_dict = torch.load(args.regressor_ckpt_path, map_location=map_location)
        keys = state_dict.keys()
        values = state_dict.values()
        new_keys = []
        for key in keys:
            new_key = key[7:]
            new_keys.append(new_key)
        new_state_dict = OrderedDict(list(zip(new_keys, values)))  # create a new OrderedDict with (key, value) pairs
        regressor.load_state_dict(new_state_dict)
    else:
        raise  # currently not supported

    return resnet50_conv, regressor


def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def return_dataloader(args, run_mode):
    with open(args.anno_file) as f:
        annotations = json.load(f)
    with open(args.data_split_file) as f:
        data_split = json.load(f)

    if run_mode == "training":
        data_split_type = args.val_split
    elif run_mode == "testing":
        data_split_type = args.test_split
    else:
        raise
    if run_mode == "training" and args.val_loss:
        val_data = FSC147Dataset(
            args,
            annotations,
            run_mode=run_mode,
            data_split=data_split,
            data_split_type=data_split_type,
            return_gt_density=True,
        )
    else:
        val_data = FSC147Dataset(
            args,
            annotations,
            run_mode=run_mode,
            data_split=data_split,
            data_split_type=data_split_type,
            return_gt_density=False,
        )
    val_sampler = DistributedSampler(val_data, shuffle=args.data_shuffle)
    val_dataloader = DataLoader(
        val_data,
        batch_size=args.batch_size,
        pin_memory=args.pin_memory,
        sampler=val_sampler,
        worker_init_fn=worker_init_fn,
    )

    if run_mode == "training":
        training_data = FSC147Dataset(
            args,
            annotations,
            run_mode="training",
            data_split=data_split,
            data_split_type="train",
            return_gt_density=True,
        )
        train_sampler = DistributedSampler(training_data, shuffle=args.data_shuffle)
        train_dataloader = DataLoader(
            training_data,
            batch_size=args.batch_size,
            pin_memory=args.pin_memory,
            sampler=train_sampler,
            worker_init_fn=worker_init_fn,
        )
        return train_dataloader, train_sampler, val_dataloader, val_sampler
    else:
        return val_dataloader, val_sampler


def return_loss(args, output, gt_density, criterion):
    # if image size isn't divisible by 8, gt size is slightly different from output size
    if output.shape[-2] != gt_density.shape[-2] or output.shape[-1] != gt_density.shape[-1]:
        orig_count = gt_density.sum().detach().item()
        gt_density = F.interpolate(
            gt_density,
            size=(output.shape[-2], output.shape[-1]),
            mode="bilinear",
            align_corners=True,
        )
        new_count = gt_density.sum().detach().item()
        if new_count > 0:
            gt_density = gt_density * (orig_count / new_count)
    if args.density_map_scale != 1:
        gt_density *= args.density_map_scale
    loss = criterion(output, gt_density)
    return loss


def init_stats(args):
    stats = {
        "epoch_list": [],
        "train_mae": [],
        "train_rmse": [],
        "val_mae": [],
        "val_rmse": [],
        "train_loss": [],
        "best_mae": 1e7,
        "best_rmse": 1e7,
        "best_epoch": 0,
        "cur_epoch": 0,
    }
    if args.val_loss:
        stats["val_loss"] = []
        stats["best_val_loss"] = 1e7
    return stats


def update_stats(
    args,
    stats,
    epoch,
    train_mae,
    train_rmse,
    val_mae,
    val_rmse,
    train_loss,
    val_loss=None,
):
    stats["epoch_list"].append(epoch)
    stats["train_mae"].append(train_mae)
    stats["train_rmse"].append(train_rmse)
    stats["val_mae"].append(val_mae)
    stats["val_rmse"].append(val_rmse)
    stats["train_loss"].append(train_loss)
    stats["cur_epoch"] = epoch
    if args.val_loss and val_loss is not None:
        stats["val_loss"].append(val_loss)
    return stats


def save_stats(args, stats):
    with open(args.stats_path, "w") as f:
        json.dump(stats, f, indent=4)


def save_log(args, print_log, st_time, start_epoch, epoch):
    stats_file = os.path.join(args.output_dir, "log.txt")
    with open(stats_file, "a") as f:
        f.write(print_log)
    avg_epoch_time_present = (time.time() - st_time) / (epoch - start_epoch + 1)
    estimate_leftover_time = (args.epochs - epoch) * avg_epoch_time_present
    estimate_leftover_time_str = datetime.timedelta(seconds=estimate_leftover_time)
    with open(stats_file, "a") as f:
        f.write(" avg_etp={:.2f}s el_time={}\n".format(avg_epoch_time_present, estimate_leftover_time_str))


def rm_if_exists(path):
    if os.path.exists(path):
        os.system(f"rm {path}")


def plot_stats(args, stats):
    if (not args.fast_train) or args.val_loss:
        max_exclude_epoch_num = 3
        if len(stats["epoch_list"]) <= max_exclude_epoch_num:
            exclude_epoch_num_list = list(range(len(stats["epoch_list"])))
        else:
            exclude_epoch_num_list = list(range(max_exclude_epoch_num))
        # for analysis
        for exclude_epoch_num in exclude_epoch_num_list:
            if not args.fast_train:
                plt.plot(
                    stats["epoch_list"][exclude_epoch_num:],
                    stats["train_loss"][exclude_epoch_num:],
                    label="train_loss",
                )
            if args.val_loss:
                plt.plot(
                    stats["epoch_list"][exclude_epoch_num:],
                    stats["val_loss"][exclude_epoch_num:],
                    label="val_loss",
                )
                # notice
                best_val_loss = min(stats["val_loss"][exclude_epoch_num:])
                best_val_loss_epoch = stats["val_loss"][exclude_epoch_num:].index(best_val_loss)
                plt.title(
                    "e={}_bvloss={:.10f}_tloss={:.10f}".format(
                        best_val_loss_epoch + exclude_epoch_num,
                        best_val_loss,
                        stats["train_loss"][exclude_epoch_num:][best_val_loss_epoch],
                    )
                )
            plt.legend()
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            loss_fig_path = os.path.join(args.output_dir, f"loss_exclude_initial_{exclude_epoch_num}_epoch.png")
            rm_if_exists(loss_fig_path)
            plt.savefig(loss_fig_path)
            plt.close()

    for metric in ["train_mae", "train_rmse", "val_mae", "val_rmse"]:
        plt.plot(stats["epoch_list"], stats[metric], label=metric)
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    best_val_mae = min(stats["val_mae"])
    best_val_mae_epoch = stats["val_mae"].index(best_val_mae)
    best_val_rmse = min(stats["val_rmse"])
    best_val_rmse_epoch = stats["val_rmse"].index(best_val_rmse)
    plt.title(
        "e={}_bvmae={:.2f}_e={}_bvrmse={:.2f}".format(
            best_val_mae_epoch, best_val_mae, best_val_rmse_epoch, best_val_rmse
        )
    )
    metric_fig_path = os.path.join(args.output_dir, "metric.png")
    rm_if_exists(metric_fig_path)
    plt.savefig(os.path.join(args.output_dir, "metric.png"))
    plt.close()


def save_model(args, regressor):
    torch.save(regressor.state_dict(), args.regressor_ckpt_path)


def update_count_dict(count_dict, im_id, gt_cnt, pred_cnt, with_loss=False, loss=None):
    count_dict[im_id] = {}
    count_dict[im_id]["gt_cnt"] = gt_cnt
    count_dict[im_id]["pred_cnt"] = pred_cnt
    if with_loss and loss is not None:
        count_dict[im_id]["loss"] = loss


def save_count_dict(args, count_dict, run_mode, split):
    save_json_file_dir = os.path.join(
        args.output_dir,
        f"metrics_json_results_rm={run_mode}_s={split}_{args.adapt_setting_str}",
    )
    create_folder(save_json_file_dir)
    save_json_file_path = os.path.join(save_json_file_dir, f"{args.rank}.json")
    with open(save_json_file_path, "w") as f:
        json.dump(count_dict, f, indent=4)
    torch.distributed.barrier()
    return save_json_file_dir


def cal_metrics_using_count_dict(save_json_file_dir, with_loss=False):
    all_count_dict = {}
    json_filename_list = os.listdir(save_json_file_dir)
    for json_filename in json_filename_list:
        json_file_path = os.path.join(save_json_file_dir, json_filename)
        with open(json_file_path, "r") as f:
            count_dict = json.load(f)
            all_count_dict.update(count_dict)

    all_sample_num = len(all_count_dict.keys())

    SAE = 0  # sum of absolute errors
    SSE = 0  # sum of square errors
    POS_ERR = 0
    NEG_ERR = 0
    if with_loss:
        loss_sum = 0
    for im_id in all_count_dict.keys():
        gt_cnt = all_count_dict[im_id]["gt_cnt"]
        pred_cnt = all_count_dict[im_id]["pred_cnt"]

        err = abs(gt_cnt - pred_cnt)
        SAE += err
        SSE += err ** 2

        init_err = pred_cnt - gt_cnt
        if init_err > 0:
            POS_ERR += init_err
        elif init_err < 0:
            NEG_ERR += init_err

        if with_loss:
            loss_sum += all_count_dict[im_id]["loss"]

    MAE = SAE / all_sample_num
    RMSE = (SSE / all_sample_num) ** 0.5
    if with_loss:
        return MAE, RMSE, POS_ERR, NEG_ERR, all_sample_num, loss_sum / all_sample_num
    else:
        return MAE, RMSE, POS_ERR, NEG_ERR, all_sample_num


if __name__ == "__main__":
    pass
