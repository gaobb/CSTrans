import numpy as np
import cv2
import os
import json
import torch
import torch.nn.functional as F

try:
    from utils import create_folder
    from constants import PROJECT_DIR
except:  # noqa: E722
    from tools.utils import create_folder
    from tools.constants import PROJECT_DIR


def save_vis_prob_map(save_fname=None, prob_map=None, return_prob_map=False):
    assert prob_map is not None
    prob_map_min = np.min(prob_map)
    prob_map -= prob_map_min
    prob_map_max = np.max(prob_map)
    if prob_map_max > 0:
        prob_map = (255 * prob_map / prob_map_max).astype(np.uint8)
    elif prob_map_max == 0:
        prob_map = prob_map.astype(np.uint8)
    else:
        raise
    vis_prob_map = cv2.applyColorMap(prob_map, cv2.COLORMAP_JET)
    if return_prob_map:
        return vis_prob_map
    else:
        assert save_fname is not None
        cv2.imwrite(save_fname, vis_prob_map)


def float2int(num):
    return int(num + 0.5)


def vis_dataset():
    imgs_root = f"{PROJECT_DIR}/datasets/FSC-147/images_384_VarV2"
    gt_density_map_root = f"{PROJECT_DIR}/datasets/FSC-147/gt_density_map_adaptive_384_VarV2"
    gt_vis_density_map_root = f"{PROJECT_DIR}/datasets/FSC-147/gt_density_map_adaptive_384_VarV2_vis"
    imgs_with_dot_vis_root = f"{PROJECT_DIR}/datasets/FSC-147/images_384_VarV2_vis"
    anno_file = f"{PROJECT_DIR}/datasets/FSC-147/annotation_FSC147_384.json"
    imgs_name_list = os.listdir(imgs_root)
    imgs_name_list.sort()
    create_folder(gt_vis_density_map_root)
    create_folder(imgs_with_dot_vis_root)

    with open(anno_file) as f:
        annotations = json.load(f)

    for img_name_i, img_name in enumerate(imgs_name_list):
        print("Processing {}/{}. img_name is {}".format(img_name_i + 1, len(imgs_name_list), img_name))
        density_map = np.load(os.path.join(gt_density_map_root, img_name.replace(".jpg", ".npy")))
        save_vis_prob_map(
            save_fname=os.path.join(gt_vis_density_map_root, img_name.replace(".jpg", ".jpg")),
            prob_map=density_map,
        )

        anno = annotations[img_name]
        bboxes = anno["box_examples_coordinates"]
        dots = np.array(anno["points"])
        init_img = cv2.imread(os.path.join(imgs_root, img_name))
        for center_point in dots:
            cv2.circle(
                init_img,
                (float2int(center_point[0]), float2int(center_point[1])),
                4,
                (0, 255, 0),
                thickness=-1,
            )
        for bbox in bboxes:
            cv2.rectangle(
                init_img,
                (float2int(bbox[0][0]), float2int(bbox[0][1])),
                (float2int(bbox[2][0]), float2int(bbox[2][1])),
                (0, 0, 255),
                2,
            )
        cv2.imwrite(
            os.path.join(imgs_with_dot_vis_root, img_name.replace(".jpg", ".png")),
            init_img,
        )


def save_vis_test_results(
    args,
    im_id,
    output,
    gt_cnt,
    pred_cnt,
    run_mode,
    vis_with_init_img=False,
    dots=None,
    boxes=None,
    scale_factor=None,
    corr_feat_for_vis=None,
    epoch=None,
):
    init_img = cv2.imread(os.path.join(args.im_dir, im_id))
    img_name = im_id.split(".")[0]
    vis_gt_density_map = cv2.imread(os.path.join(args.gt_den_map_vis_dir, im_id))
    assert vis_gt_density_map is not None

    # vis on the init image
    img_with_anno = init_img.copy()
    assert dots is not None
    assert dots.shape[0] == 1
    assert boxes is not None
    assert boxes.shape[0] == 1
    for center_point in dots[0].cpu().numpy():
        cv2.circle(
            img_with_anno,
            (float2int(center_point[0]), float2int(center_point[1])),
            4,
            (0, 255, 0),
            thickness=-1,
        )
    for bbox in boxes[0][0].cpu().numpy():
        # notice
        cv2.rectangle(
            img_with_anno,
            (float2int(bbox[2] / scale_factor), float2int(bbox[1] / scale_factor)),
            (float2int(bbox[4] / scale_factor), float2int(bbox[3] / scale_factor)),
            (0, 0, 255),
            2,
        )

    assert img_with_anno is not None
    assert output.shape[0] == 1
    # if image size isn't divisible by 8, gt size is slightly different from output size
    if output.shape[2] != vis_gt_density_map.shape[0] or output.shape[3] != vis_gt_density_map.shape[1]:
        output = F.interpolate(
            output,
            size=(vis_gt_density_map.shape[0], vis_gt_density_map.shape[1]),
            mode="bilinear",
            align_corners=True,
        )
    pred_density_map = output.detach().cpu().numpy()[0, 0, :, :]
    vis_pred_density_map = save_vis_prob_map(prob_map=pred_density_map, return_prob_map=True)

    # save dir of vis results
    if run_mode == "training":
        split_str = args.val_split
        epoch_str = f"e={epoch}"
    elif run_mode == "testing":
        split_str = args.test_split
        with open(os.path.join(args.output_dir, "stats.json"), "r") as f:
            stats = json.load(f)
        epoch_str = "bste={}_cure={}".format(stats["best_epoch"], stats["cur_epoch"])
    results_vis_dir = os.path.join(
        args.output_dir,
        f"vis_results_rm={run_mode}_s={split_str}_{args.adapt_setting_str}_{epoch_str}",
    )
    create_folder(results_vis_dir)
    if args.save_fname_err_prefix:
        save_file_path = os.path.join(
            results_vis_dir,
            "error={:+.2f}_gtcnt={}_predcnt={:.2f}_id={}.jpg".format(pred_cnt - gt_cnt, gt_cnt, pred_cnt, img_name),
        )
    elif args.save_fname_err_postfix:
        save_file_path = os.path.join(
            results_vis_dir,
            "id={}_error={:+.2f}_gtcnt={}_predcnt={:.2f}.jpg".format(img_name, pred_cnt - gt_cnt, gt_cnt, pred_cnt),
        )
    else:
        save_file_path = os.path.join(results_vis_dir, "id={}.jpg".format(img_name))

    # merge the vis results to save
    space_width = 20
    vertical_space = np.ones((img_with_anno.shape[0], space_width, 3), np.uint8) * 255
    if vis_with_init_img:
        horizontal_space = np.ones((space_width, img_with_anno.shape[1] * 2 + space_width, 3), np.uint8) * 255
        merge_vis_result_1 = np.concatenate((init_img, vertical_space, img_with_anno), axis=1)
        merge_vis_result_2 = np.concatenate((vis_gt_density_map, vertical_space, vis_pred_density_map), axis=1)
        merge_vis_results = np.concatenate((merge_vis_result_1, horizontal_space, merge_vis_result_2), axis=0)
    else:
        merge_vis_results = np.concatenate(
            (
                img_with_anno,
                vertical_space,
                vis_gt_density_map,
                vertical_space,
                vis_pred_density_map,
            ),
            axis=1,
        )

    # vis_corr_feat
    if args.vis_corr_feat:
        assert corr_feat_for_vis is not None

        num_sample = len(corr_feat_for_vis.keys())
        for ix in range(num_sample):
            _, _, tar_h, tar_w = corr_feat_for_vis[f"sample_{ix}"][args.MAPS[0]].shape
            for map_key in args.MAPS[1:]:
                if (
                    corr_feat_for_vis[f"sample_{ix}"][map_key].shape[2] != tar_h
                    or corr_feat_for_vis[f"sample_{ix}"][map_key].shape[3] != tar_w
                ):
                    corr_feat_for_vis[f"sample_{ix}"][map_key] = F.interpolate(
                        corr_feat_for_vis[f"sample_{ix}"][map_key],
                        size=(tar_h, tar_w),
                        mode="bilinear",
                        align_corners=True,
                    )
            Combined = torch.cat(
                [corr_feat_for_vis[f"sample_{ix}"][map_key] for map_key in args.MAPS],
                dim=1,
            )
            if ix == 0:
                all_feat = 1.0 * Combined.unsqueeze(0)
            else:
                all_feat = torch.cat((all_feat, Combined.unsqueeze(0)), dim=0)

        all_feat = all_feat.detach().cpu().numpy()
        all_feat -= np.min(all_feat)
        _, bbox_num, scale_mul_layer_num, corr_feat_h, corr_feat_w = all_feat.shape
        _, w, c = merge_vis_results.shape
        vis_mask = (
            np.ones(
                ((corr_feat_h + space_width) * bbox_num + space_width, w, c),
                dtype=np.uint8,
            )
            * 255
        )
        for i in range(bbox_num):
            for j in range(scale_mul_layer_num):
                cur_corr_feat = save_vis_prob_map(prob_map=all_feat[0, i, j, :, :], return_prob_map=True)
                vis_mask[
                    (corr_feat_h + space_width) * i
                    + space_width: (corr_feat_h + space_width) * i
                    + corr_feat_h
                    + space_width,
                    (corr_feat_w + space_width) * j: (corr_feat_w + space_width) * j + corr_feat_w,
                    :,
                ] = cur_corr_feat
        merge_vis_results = np.concatenate((merge_vis_results, vis_mask), axis=0)

    cv2.imwrite(save_file_path, merge_vis_results)


if __name__ == "__main__":
    vis_dataset()
