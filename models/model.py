import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from collections import OrderedDict


class Resnet50FPN(nn.Module):
    def __init__(self, args, pretrained=True):
        super(Resnet50FPN, self).__init__()
        self.args = args
        self.pretrained = pretrained
        os.environ["TORCH_HOME"] = self.args.torch_home
        if self.args.backbone == "resnet50":
            backbone_net = torchvision.models.resnet50(pretrained=self.pretrained)
        else:
            raise
        children = list(backbone_net.children())
        self.conv1 = nn.Sequential(*children[:4])
        self.conv2 = children[4]
        self.conv3 = children[5]
        self.conv4 = children[6]

    def forward(self, x):
        feat = OrderedDict()
        x = self.conv1(x)
        x = self.conv2(x)
        x_map3 = self.conv3(x)
        x_map4 = self.conv4(x_map3)
        feat["map3"] = x_map3
        feat["map4"] = x_map4
        return feat


class CountRegressor(nn.Module):
    def __init__(self, args, pool="mean"):
        super(CountRegressor, self).__init__()
        self.pool = pool
        self.args = args

        if self.args.regressor_in_prepro_type == "none":
            # split image into non-overlapping patches
            regressor_in_chans = self.args.regressor_input_channels
        elif self.args.regressor_in_prepro_type == "conv1x1":
            self.regressor_in_prepro_conv1x1 = nn.Conv2d(
                self.args.regressor_input_channels,
                self.args.regressor_in_prepro_out_chans,
                kernel_size=1,
            )
            regressor_in_chans = self.args.regressor_in_prepro_out_chans
        else:
            raise

        self.regressor = nn.Sequential(
            nn.Conv2d(regressor_in_chans, 196, 7, padding=3),
            nn.ReLU(inplace=self.args.inplace_flag),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(196, 128, 5, padding=2),
            nn.ReLU(inplace=self.args.inplace_flag),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=self.args.inplace_flag),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(64, 32, 1),
            nn.ReLU(inplace=self.args.inplace_flag),
            nn.Conv2d(32, self.args.regressor_output_channels, 1),
            nn.ReLU(inplace=self.args.inplace_flag),
        )

    def forward(self, all_feat_dict):
        num_sample = len(all_feat_dict.keys())  # args.batch_size
        for ix in range(num_sample):
            _, _, tar_h, tar_w = all_feat_dict[f"sample_{ix}"][self.args.MAPS[0]].shape
            for map_key in self.args.MAPS[1:]:
                if (
                    all_feat_dict[f"sample_{ix}"][map_key].shape[2] != tar_h
                    or all_feat_dict[f"sample_{ix}"][map_key].shape[3] != tar_w
                ):
                    all_feat_dict[f"sample_{ix}"][map_key] = F.interpolate(
                        all_feat_dict[f"sample_{ix}"][map_key],
                        size=(tar_h, tar_w),
                        mode="bilinear",
                        align_corners=True,
                    )
            Combined = torch.cat(
                [all_feat_dict[f"sample_{ix}"][map_key] for map_key in self.args.MAPS],
                dim=1,
            )
            if ix == 0:
                all_feat = 1.0 * Combined.unsqueeze(0)
            else:
                all_feat = torch.cat((all_feat, Combined.unsqueeze(0)), dim=0)
        if num_sample == 1:
            if self.args.regressor_in_prepro_type == "none":
                output = self.regressor(all_feat.squeeze(0))
            elif self.args.regressor_in_prepro_type == "conv1x1":
                output = self.regressor(self.regressor_in_prepro_conv1x1(all_feat.squeeze(0)))
            else:
                raise
            if self.pool == "mean":
                output = torch.mean(output, dim=(0), keepdim=True)
                return output
            elif self.pool == "max":
                output, _ = torch.max(output, 0, keepdim=True)
                return output
        else:
            assert self.args.regressor_in_prepro_type == "none"
            for i in range(0, num_sample):
                output = self.regressor(all_feat[i])
                if self.pool == "mean":
                    output = torch.mean(output, dim=(0), keepdim=True)
                elif self.pool == "max":
                    output, _ = torch.max(output, 0, keepdim=True)
                if i == 0:
                    Output = output
                else:
                    Output = torch.cat((Output, output), dim=0)
            return Output


def weights_normal_init(model, dev=0.01):
    if isinstance(model, list):
        for m in model:
            weights_normal_init(m, dev)
    else:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, dev)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, dev)


def weights_xavier_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain("relu"))
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
