import numpy as np
import cv2
import random
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class resizeImage(object):
    """
    If either the width or height of an image exceed a specified value, resize the image so that:
        1. The maximum of the new height and new width does not exceed a specified value
        2. The new height and new width are divisible by 8
        3. The aspect ratio is preserved
    No resizing is done if both height and width are smaller than the specified value
    By: Minh Hoai Nguyen (minhhoai@gmail.com)
    """

    def __init__(self, args):
        self.args = args
        self.Normalize = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=self.args.IM_NORM_MEAN, std=self.args.IM_NORM_STD),
            ]
        )

    def __call__(self, sample):
        image, lines_boxes = sample["image"], sample["lines_boxes"]

        W, H = image.size
        if W > self.args.MAX_HW or H > self.args.MAX_HW:
            scale_factor = float(self.args.MAX_HW) / max(H, W)
            new_H = 8 * int(H * scale_factor / 8)
            new_W = 8 * int(W * scale_factor / 8)
            image = transforms.Resize((new_H, new_W))(image)
        else:
            scale_factor = 1

        boxes = list()
        for box in lines_boxes:
            box2 = [int(k * scale_factor) for k in box]
            y1, x1, y2, x2 = box2[0], box2[1], box2[2], box2[3]
            boxes.append([0, y1, x1, y2, x2])

        boxes = torch.Tensor(boxes).unsqueeze(0)
        image = self.Normalize(image)
        sample = {"image": image, "boxes": boxes, "scale_factor": scale_factor}
        return sample


class resizeImageWithGT(object):
    """
    If either the width or height of an image exceed a specified value, resize the image so that:
        1. The maximum of the new height and new width does not exceed a specified value
        2. The new height and new width are divisible by 8
        3. The aspect ratio is preserved
    No resizing is done if both height and width are smaller than the specified value
    By: Minh Hoai Nguyen (minhhoai@gmail.com)
    Modified by: Viresh
    """

    def __init__(self, args):
        self.args = args
        self.Normalize = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=self.args.IM_NORM_MEAN, std=self.args.IM_NORM_STD),
            ]
        )

    def __call__(self, sample):
        image, lines_boxes, density = (
            sample["image"],
            sample["lines_boxes"],
            sample["gt_density"],
        )

        W, H = image.size
        if W > self.args.MAX_HW or H > self.args.MAX_HW:
            scale_factor = float(self.args.MAX_HW) / max(H, W)
        else:
            scale_factor = 1

        assert 0 < scale_factor <= 1
        if scale_factor != 1:
            new_H = 8 * int(H * scale_factor / 8)
            new_W = 8 * int(W * scale_factor / 8)
            image = transforms.Resize((new_H, new_W))(image)
            orig_count = np.sum(density)
            density = cv2.resize(density, (new_W, new_H))
            new_count = np.sum(density)
            if new_count > 0:
                density = density * (orig_count / new_count)

        boxes = list()
        for box in lines_boxes:
            box2 = [int(k * scale_factor) for k in box]
            y1, x1, y2, x2 = box2[0], box2[1], box2[2], box2[3]
            boxes.append([0, y1, x1, y2, x2])

        boxes = torch.Tensor(boxes).unsqueeze(0)
        image = self.Normalize(image)
        density = torch.from_numpy(density).unsqueeze(0)
        sample = {
            "image": image,
            "boxes": boxes,
            "gt_density": density,
            "scale_factor": scale_factor,
        }
        return sample


class FSC147Dataset(Dataset):
    def __init__(
        self,
        args,
        annotations,
        run_mode,
        data_split,
        data_split_type,
        return_gt_density=True,
    ):
        self.args = args
        self.annotations = annotations
        self.run_mode = run_mode
        self.data_split = data_split
        self.data_split_type = data_split_type
        self.return_gt_density = return_gt_density
        if self.data_split_type == "all":
            self.im_ids = self.data_split["train"] + self.data_split["val"] + self.data_split["test"]
        elif self.data_split_type in ["train", "val", "test", "val_coco", "test_coco"]:
            self.im_ids = self.data_split[data_split_type]
        else:
            raise
        if self.return_gt_density:
            transform_train_list = []
            transform_train_list.append(resizeImageWithGT(self.args))
            self.TransformTrain = transforms.Compose(transform_train_list)
        else:
            self.Transform = transforms.Compose([resizeImage(self.args)])

    def __len__(self):
        return len(self.im_ids)

    def __getitem__(self, idx):
        im_id = self.im_ids[idx]
        anno = self.annotations[im_id]
        bboxes = anno["box_examples_coordinates"]
        dots = np.array(anno["points"])

        rects = list()
        if self.run_mode == "training":
            if self.args.random_shuffle_training_gt_bboxes:
                random.shuffle(bboxes)
            if self.args.training_gt_bboxes_num >= 1:
                bboxes = bboxes[: self.args.training_gt_bboxes_num]
        elif self.run_mode == "testing":
            if self.args.random_shuffle_testing_gt_bboxes:
                random.shuffle(bboxes)
            # if self.args.testing_gt_bboxes_num >= 1:
            #     bboxes = bboxes[: self.args.testing_gt_bboxes_num]
            if self.args.testing_gt_bboxes_num >= 1:
                bboxes = bboxes[
                    self.args.box_group_index: self.args.box_group_index + self.args.testing_gt_bboxes_num
                ]
        for bbox in bboxes:
            x1 = bbox[0][0]
            y1 = bbox[0][1]
            x2 = bbox[2][0]
            y2 = bbox[2][1]
            rects.append([y1, x1, y2, x2])

        image = Image.open("{}/{}".format(self.args.im_dir, im_id))
        image.load()

        if self.return_gt_density:
            density_path = self.args.gt_den_map_dir + "/" + im_id.split(".")[0] + ".npy"
            density = np.load(density_path).astype("float32")
            sample = {"image": image, "lines_boxes": rects, "gt_density": density}
            sample = self.TransformTrain(sample)
            image, boxes, gt_density, scale_factor = (
                sample["image"].cuda(non_blocking=self.args.non_blocking),
                sample["boxes"].cuda(non_blocking=self.args.non_blocking),
                sample["gt_density"].cuda(non_blocking=self.args.non_blocking),
                sample["scale_factor"],
            )
            return image, dots, boxes, im_id, gt_density, scale_factor
        else:
            sample = {"image": image, "lines_boxes": rects}
            sample = self.Transform(sample)
            image, boxes, scale_factor = (
                sample["image"].cuda(non_blocking=self.args.non_blocking),
                sample["boxes"].cuda(non_blocking=self.args.non_blocking),
                sample["scale_factor"],
            )
            return image, dots, boxes, im_id, scale_factor
