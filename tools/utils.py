import argparse
import numpy as np
import math
import torch
import torch.distributed as dist
import torch.nn.functional as F
import os
import random
import json
import time


def matlab_style_gauss2D(shape=(3, 3), sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m, n = [(ss - 1.0) / 2.0 for ss in shape]
    y, x = np.ogrid[-m: m + 1, -n: n + 1]
    h = np.exp(-(x * x + y * y) / (2.0 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def PerturbationLoss(args, output, boxes, sigma=8, use_gpu=True):
    Loss = 0.0
    if boxes.shape[2] > 1:  # if #examplar boxes is larger than 1
        boxes = boxes.squeeze()
        for tempBoxes in boxes.squeeze():
            y1 = int(tempBoxes[1])
            y2 = int(tempBoxes[3])
            x1 = int(tempBoxes[2])
            x2 = int(tempBoxes[4])
            out = output[:, :, y1:y2, x1:x2]
            GaussKernel = matlab_style_gauss2D(shape=(out.shape[2], out.shape[3]), sigma=sigma)
            GaussKernel = torch.from_numpy(GaussKernel).float()
            if use_gpu:
                GaussKernel = GaussKernel.cuda(non_blocking=args.non_blocking)
            Loss += F.mse_loss(out.squeeze(), GaussKernel)
    else:
        boxes = boxes.squeeze()
        y1 = int(boxes[1])
        y2 = int(boxes[3])
        x1 = int(boxes[2])
        x2 = int(boxes[4])
        out = output[:, :, y1:y2, x1:x2]
        Gauss = matlab_style_gauss2D(shape=(out.shape[2], out.shape[3]), sigma=sigma)
        GaussKernel = torch.from_numpy(Gauss).float()
        if use_gpu:
            GaussKernel = GaussKernel.cuda(non_blocking=args.non_blocking)
        Loss += F.mse_loss(out.squeeze(), GaussKernel)
    return Loss


def MincountLoss(args, output, boxes, use_gpu=True):
    ones = torch.ones(1)
    if use_gpu:
        ones = ones.cuda(non_blocking=args.non_blocking)
    Loss = 0.0
    if boxes.shape[2] > 1:  # if #examplar boxes is larger than 1
        boxes = boxes.squeeze()
        for tempBoxes in boxes.squeeze():
            y1 = int(tempBoxes[1])
            y2 = int(tempBoxes[3])
            x1 = int(tempBoxes[2])
            x2 = int(tempBoxes[4])
            X = output[:, :, y1:y2, x1:x2].sum().unsqueeze(0)
            if X.item() <= 1:
                Loss += F.mse_loss(X, ones)
    else:
        boxes = boxes.squeeze()
        y1 = int(boxes[1])
        y2 = int(boxes[3])
        x1 = int(boxes[2])
        x2 = int(boxes[4])
        X = output[:, :, y1:y2, x1:x2].sum().unsqueeze(0)
        if X.item() <= 1:
            Loss += F.mse_loss(X, ones)
    return Loss


def cal_corr_feat(args, image_features, examples_features):
    if args.corr_feat_keep_method == "query_feat_pad":
        supp_h, supp_w = examples_features.shape[2], examples_features.shape[3]
        features = F.conv2d(
            F.pad(
                image_features,
                (
                    (int(supp_w / 2)),
                    int((supp_w - 1) / 2),
                    int(supp_h / 2),
                    int((supp_h - 1) / 2),
                ),
            ),
            examples_features,
        )
    elif args.corr_feat_keep_method == "corr_feat_resize":
        _, _, query_h, query_w = image_features.shape
        features = F.conv2d(image_features, examples_features)
        features = F.interpolate(features, size=(query_h, query_w), mode="bilinear", align_corners=True)
    else:
        raise
    return features


def resize_support_feat_and_conv(args, M, image_features, boxes_scaled):
    box_hs = boxes_scaled[:, 3] - boxes_scaled[:, 1]
    box_ws = boxes_scaled[:, 4] - boxes_scaled[:, 2]
    max_h = math.ceil(max(box_hs))
    max_w = math.ceil(max(box_ws))
    for j in range(0, M):  # for each examplar box
        y1, x1 = int(boxes_scaled[j, 1]), int(boxes_scaled[j, 2])
        y2, x2 = int(boxes_scaled[j, 3]), int(boxes_scaled[j, 4])
        if j == 0:
            examples_features = image_features[:, :, y1:y2, x1:x2]
            if examples_features.shape[2] != max_h or examples_features.shape[3] != max_w:
                examples_features = F.interpolate(
                    examples_features,
                    size=(max_h, max_w),
                    mode="bilinear",
                    align_corners=True,
                )
        else:
            feat = image_features[:, :, y1:y2, x1:x2]
            if feat.shape[2] != max_h or feat.shape[3] != max_w:
                feat = F.interpolate(feat, size=(max_h, max_w), mode="bilinear", align_corners=True)
            examples_features = torch.cat((examples_features, feat), dim=0)  # cat in the batch dim
    """
    Convolving example features over image features
    """
    corr_features = cal_corr_feat(args, image_features, examples_features)
    return examples_features, corr_features


def conv_respectively(args, M, image_features, boxes_scaled):
    supp_feat_list = []
    corr_feat_list = []
    for j in range(0, M):  # for each examplar box
        y1, x1 = int(boxes_scaled[j, 1]), int(boxes_scaled[j, 2])
        y2, x2 = int(boxes_scaled[j, 3]), int(boxes_scaled[j, 4])
        examples_features = image_features[:, :, y1:y2, x1:x2]
        supp_feat_list.append(examples_features)
        corr_feat_list.append(cal_corr_feat(args, image_features, examples_features))
    if M == 1:
        return supp_feat_list, corr_feat_list[0]
    else:
        return supp_feat_list, torch.cat(corr_feat_list, dim=1)


def scale_h_w(args, h, w, scale, image_features):
    h1 = math.ceil(h * scale)
    w1 = math.ceil(w * scale)
    if h1 < 1:  # use original size if scaled size is too small
        h1 = h
    if w1 < 1:
        w1 = w
    if args.corr_feat_keep_method == "corr_feat_resize":
        # notice
        _, _, query_h, query_w = image_features.shape
        if h1 > query_h:
            h1 = query_h
        if w1 > query_w:
            w1 = query_w
    return h1, w1


def scale_supp_feat(args, image_features, examples_features, scale):
    _, _, h, w = examples_features.shape
    h1, w1 = scale_h_w(args, h, w, scale, image_features)
    return F.interpolate(examples_features, size=(h1, w1), mode="bilinear", align_corners=True)


def resize_support_feat_and_conv_scaled(args, image_features, examples_features, scale):
    examples_features_scaled = scale_supp_feat(args, image_features, examples_features, scale)
    return cal_corr_feat(args, image_features, examples_features_scaled)


def conv_respectively_scaled(args, M, image_features, examples_features_list, scale):
    corr_feat_scaled_list = []
    for examples_features in examples_features_list:  # for each examplar box
        examples_features_scaled = scale_supp_feat(args, image_features, examples_features, scale)
        corr_feat_scaled_list.append(cal_corr_feat(args, image_features, examples_features_scaled))
    if M == 1:
        return corr_feat_scaled_list[0]
    else:
        return torch.cat(corr_feat_scaled_list, dim=1)


def fuse_init_feat_with_corr_feat(args, image_features, combined):
    num_bboxes, num_scales, corr_feat_h, corr_feat_w = combined.shape
    _, _, init_feat_h, init_feat_w = image_features.shape
    if init_feat_h != corr_feat_h or init_feat_w != corr_feat_w:
        image_features = F.interpolate(
            image_features,
            size=(corr_feat_h, corr_feat_w),
            mode="bilinear",
            align_corners=True,
        )
    combined -= torch.min(combined)
    fuse_feat_of_diff_bboxes = []
    for box_i in range(num_bboxes):
        fuse_feat_of_diff_scales = []
        for scale_i in range(num_scales):
            if args.fuse_type == "mul":
                fuse_feat_of_diff_scales.append(
                    image_features * (combined[box_i, scale_i, :, :].unsqueeze(0).unsqueeze(0))
                )
            elif args.fuse_type == "add":
                fuse_feat_of_diff_scales.append(
                    image_features + (combined[box_i, scale_i, :, :].unsqueeze(0).unsqueeze(0))
                )
            else:
                raise  # currently not supported
        fuse_feat_of_diff_bboxes.append(torch.cat(fuse_feat_of_diff_scales, dim=1))
    return torch.cat(fuse_feat_of_diff_bboxes, dim=0)


def extract_features(args, feature_model, image, boxes):
    N, M = image.shape[0], boxes.shape[2]
    # N: batch size
    # M: the number of the exemplar boxes of each image
    """
    Getting features for the image N * C * H * W
    """
    Image_features = feature_model(image)
    """
    Getting features for the examples (N*M) * C * h * w
    """
    all_feat_dict = {}
    for ix in range(0, N):
        all_feat_dict[f"sample_{ix}"] = {}
    for ix in range(0, N):
        cur_boxes = boxes[ix][0]
        for keys in args.MAPS:
            image_features = Image_features[keys][ix].unsqueeze(0)
            if keys == "map1" or keys == "map2":
                Scaling = 4.0
            elif keys == "map3":
                Scaling = 8.0
            elif keys == "map4":
                Scaling = 16.0
            else:
                Scaling = 32.0
            boxes_scaled = cur_boxes / Scaling
            boxes_scaled[:, 1:3] = torch.floor(boxes_scaled[:, 1:3])
            boxes_scaled[:, 3:5] = torch.ceil(boxes_scaled[:, 3:5])
            boxes_scaled[:, 3:5] = boxes_scaled[:, 3:5] + 1  # make the end indices exclusive
            feat_h, feat_w = image_features.shape[-2], image_features.shape[-1]
            # make sure exemplars don't go out of bound
            boxes_scaled[:, 1:3] = torch.clamp_min(boxes_scaled[:, 1:3], 0)
            boxes_scaled[:, 3] = torch.clamp_max(boxes_scaled[:, 3], feat_h)
            boxes_scaled[:, 4] = torch.clamp_max(boxes_scaled[:, 4], feat_w)
            if args.support_feat_resize:
                examples_features, corr_features = resize_support_feat_and_conv(args, M, image_features, boxes_scaled)
            else:
                examples_features_list, corr_features = conv_respectively(args, M, image_features, boxes_scaled)

            combined = corr_features.permute([1, 0, 2, 3])
            # computing features for scales 0.9 and 1.1
            if args.Scales is not None:
                for scale in args.Scales:
                    if args.support_feat_resize:
                        corr_features_scaled = resize_support_feat_and_conv_scaled(
                            args, image_features, examples_features, scale
                        )
                    else:
                        corr_features_scaled = conv_respectively_scaled(
                            args, M, image_features, examples_features_list, scale
                        )
                    corr_features_scaled = corr_features_scaled.permute([1, 0, 2, 3])
                    combined = torch.cat((combined, corr_features_scaled), dim=1)  # cat in the channel dim

            if args.fuse_init_feat:
                combined = fuse_init_feat_with_corr_feat(args, image_features, combined)

            all_feat_dict[f"sample_{ix}"][f"{keys}"] = combined
    return all_feat_dict


def create_folder(root_path):
    if not os.path.exists(root_path):
        os.makedirs(root_path, mode=0o777, exist_ok=True)


def set_random_seed(seed=2022):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def save_args(args, mode):
    args_dict = args.__dict__
    with open(os.path.join(args.output_dir, f"args_{mode}.txt"), "w") as f:
        for key in sorted(args_dict.keys()):
            f.write(key + ": " + str(args_dict[key]) + "\n")
    with open(os.path.join(args.output_dir, f"args_{mode}.json"), "w") as f:
        json.dump(args.__dict__, f, indent=4)


def get_count_from_density_map(args, density_map):
    assert density_map.shape[0] == 1
    cnt = density_map.sum().item() / args.density_map_scale
    return cnt


def dist_init(args):
    torch.distributed.init_process_group(backend="nccl")
    args.word_size = dist.get_world_size()
    torch.cuda.set_device(args.local_rank)
    args.rank = torch.distributed.get_rank()
    args.device_count = torch.cuda.device_count()
    args.num_nodes = args.word_size // args.device_count

    return args


def load_train_args_from_json(args):
    folder_name = args.output_dir.split("/")[-1]
    if "debug_" in folder_name:
        folder_name = folder_name[6:]
    with open(os.path.join(args.output_dir, "args_train.json"), "r") as f:
        train_args_dict = json.load(f)
    for key in train_args_dict:
        if key == "output_dir":
            print("args.__dict__[key]: ", args.__dict__[key])
            print("train_args_dict[key]: ", train_args_dict[key])
        elif key == "local_rank" or key == "rank":
            pass
        else:
            args.__dict__[key] = train_args_dict[key]
    return args


def str2bool(variable):
    assert isinstance(variable, str)
    if variable.lower() in ("yes", "y", "true", "t", "1"):
        return True
    elif variable.lower() in ("no", "n", "false", "f", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value is required.")


def int_or_None(variable):
    assert isinstance(variable, str)
    if variable == "None":
        return None
    else:
        return int(variable)


def return_cur_time_str():
    return time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime(time.time()))


def set_requires_grad(args, features, requires_grad_flag):
    for sample_key in features.keys():
        for map_key in args.MAPS:
            features[sample_key][map_key].requires_grad = requires_grad_flag
    return features


def write_nvidia_info(args, cur_time_str):
    assert args.rank == 0
    if not os.path.exists(args.output_dir):
        create_folder(args.output_dir)
    with open(os.path.join(args.output_dir, "nvidia_info.txt"), "a") as f:
        f.write("=" * 100 + "\n")
        f.write(f"cur_time_str: {cur_time_str}\n")
        for line in os.popen("nvidia-smi").readlines():
            f.write(line + "\n")
        f.write("\n")


if __name__ == "__main__":
    pass
