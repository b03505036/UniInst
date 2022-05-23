import torch
from torch.nn import functional as F
from torch import nn

from adet.utils.comm import compute_locations, aligned_bilinear
from detectron2.utils.events import get_event_storage
from detectron2.structures.masks import BitMasks
from torchvision.ops.boxes import box_iou
from detectron2.structures.boxes import matched_boxlist_iou, pairwise_iou
from detectron2.modeling.meta_arch.retinanet import permute_to_N_HWA_K

def dice_coefficient(x, target):
    eps = 1e-5
    n_inst = x.size(0)
    x = x.reshape(n_inst, -1)
    target = target.reshape(n_inst, -1)
    intersection = (x * target).sum(dim=1)
    union = (x ** 2.0).sum(dim=1) + (target ** 2.0).sum(dim=1) + eps
    loss = 1. - (2 * intersection / union)
    return loss

def qulity_dice_thr(x, target):
    eps = 1e-5
    n_inst = x.size(0)
    x = x.reshape(n_inst, -1)
    x_ = torch.zeros(x.shape).to(x.device)
    x_[x >= 0.5] = 1.0
    target = target.reshape(n_inst, -1)
    intersection = (x_ * target).sum(dim=1)
    union = (x_ ** 2.0).sum(dim=1) + (target ** 2.0).sum(dim=1) + eps
    qulity = (2 * intersection / union)
    return qulity



def parse_dynamic_params(params, channels, weight_nums, bias_nums):
    assert params.dim() == 2
    assert len(weight_nums) == len(bias_nums)
    assert params.size(1) == sum(weight_nums) + sum(bias_nums)

    num_insts = params.size(0)
    num_layers = len(weight_nums)

    params_splits = list(torch.split_with_sizes(
        params, weight_nums + bias_nums, dim=1
    ))

    weight_splits = params_splits[:num_layers]
    bias_splits = params_splits[num_layers:]

    for l in range(num_layers):
        if l < num_layers - 1:
            weight_splits[l] = weight_splits[l].reshape(num_insts * channels, -1, 1, 1)
            bias_splits[l] = bias_splits[l].reshape(num_insts * channels)
        else:
            weight_splits[l] = weight_splits[l].reshape(num_insts * 1, -1, 1, 1)
            bias_splits[l] = bias_splits[l].reshape(num_insts)

    return weight_splits, bias_splits


def build_dynamic_mask_head(cfg):
    return DynamicMaskHead(cfg)


class DynamicMaskHead(nn.Module):
    def __init__(self, cfg):
        super(DynamicMaskHead, self).__init__()
        self.num_layers = cfg.MODEL.CONDINST.MASK_HEAD.NUM_LAYERS
        self.channels = cfg.MODEL.CONDINST.MASK_HEAD.CHANNELS
        self.in_channels = cfg.MODEL.CONDINST.MASK_BRANCH.OUT_CHANNELS
        self.mask_out_stride = cfg.MODEL.CONDINST.MASK_OUT_STRIDE
        self.disable_rel_coords = cfg.MODEL.OYOR.DISABLE_REL_COORDS

        soi = cfg.MODEL.FCOS.SIZES_OF_INTEREST
        self.register_buffer("sizes_of_interest", torch.tensor(soi + [soi[-1] * 2]))

        weight_nums, bias_nums = [], []
        for l in range(self.num_layers):
            if l == 0:
                if not self.disable_rel_coords:
                    weight_nums.append((self.in_channels + 2) * self.channels)
                else:
                    weight_nums.append(self.in_channels * self.channels)
                bias_nums.append(self.channels)
            elif l == self.num_layers - 1:
                weight_nums.append(self.channels * 1)
                bias_nums.append(1)
            else:
                weight_nums.append(self.channels * self.channels)
                bias_nums.append(self.channels)

        self.weight_nums = weight_nums
        self.bias_nums = bias_nums
        self.num_gen_params = sum(weight_nums) + sum(bias_nums)

    def mask_heads_forward(self, features, weights, biases, num_insts):
        '''
        :param features
        :param weights: [w0, w1, ...]
        :param bias: [b0, b1, ...]
        :return:
        '''
        assert features.dim() == 4
        n_layers = len(weights)
        x = features
        for i, (w, b) in enumerate(zip(weights, biases)):
            x = F.conv2d(
                x, w, bias=b,
                stride=1, padding=0,
                groups=num_insts
            )
            if i < n_layers - 1:
                x = F.relu(x)
        return x

    def mask_heads_forward_with_coords(
            self, mask_feats, mask_feat_stride, instances
    ):
        locations = compute_locations(
            mask_feats.size(2), mask_feats.size(3),
            stride=mask_feat_stride, device=mask_feats.device
        )
        n_inst = len(instances)

        im_inds = instances.im_inds
        mask_head_params = instances.mask_head_params

        N, _, H, W = mask_feats.size()

        if not self.disable_rel_coords:
            instance_locations = instances.locations
            relative_coords = instance_locations.reshape(-1, 1, 2) - locations.reshape(1, -1, 2)
            relative_coords = relative_coords.permute(0, 2, 1).float()
            self.sizes_of_interest = relative_coords.new_tensor([8, 16, 32, 64, 128])
            soi = self.sizes_of_interest.float()[instances.fpn_levels]
            relative_coords = relative_coords / soi.reshape(-1, 1, 1)
            relative_coords = relative_coords.to(dtype=mask_feats.dtype)

            mask_head_inputs = torch.cat([
                relative_coords, mask_feats[im_inds].reshape(n_inst, self.in_channels, H * W)
            ], dim=1)
        else:
            mask_head_inputs = mask_feats[im_inds].reshape(n_inst, self.in_channels, H * W)

        mask_head_inputs = mask_head_inputs.reshape(1, -1, H, W)

        weights, biases = parse_dynamic_params(
            mask_head_params, self.channels,
            self.weight_nums, self.bias_nums
        )

        mask_logits = self.mask_heads_forward(mask_head_inputs, weights, biases, n_inst)

        mask_logits = mask_logits.reshape(-1, 1, H, W)

        assert mask_feat_stride >= self.mask_out_stride
        assert mask_feat_stride % self.mask_out_stride == 0
        mask_logits = aligned_bilinear(mask_logits, int(mask_feat_stride / self.mask_out_stride))

        return mask_logits.sigmoid()

    def __call__(self, mask_feats, mask_feat_stride, pred_instances, gt_instances=None):
        if self.training:

            gt_bitmasks = pred_instances.gt_bitmasks

            if len(pred_instances) == 0:
                loss_mask = mask_feats.sum() * 0 + pred_instances.mask_head_params.sum() * 0
            else:
                mask_scores = self.mask_heads_forward_with_coords(
                    mask_feats, mask_feat_stride, pred_instances
                )
                mask_losses = dice_coefficient(mask_scores, gt_bitmasks)
                loss_mask = mask_losses.mean()

            return loss_mask.float()
        else:
            if len(pred_instances) > 0:
                mask_scores = self.mask_heads_forward_with_coords(
                    mask_feats, mask_feat_stride, pred_instances
                )
                pred_instances.pred_global_masks = mask_scores.float()

            return pred_instances

    @torch.no_grad()
    def cost(self, mask_feats, pred_instances, mask_feat_stride=8):

        gt_bitmasks = pred_instances.gt_bitmasks.unsqueeze(dim=1)

        if len(pred_instances) == 0:
            qulity = mask_feats.sum() * 0 + pred_instances.mask_head_params.sum() * 0 - 1.0
            return qulity, torch.zeros_like(qulity)
        else:
            mask_scores = self.single_img_forward(
                mask_feats, mask_feat_stride, pred_instances
            )

            qulity = qulity_dice_thr(mask_scores, gt_bitmasks)
            iou = self.compute_mask_iou_matrix(mask_scores, gt_bitmasks.squeeze(1))
        return qulity.float(), iou

    @torch.no_grad()
    def single_img_forward(
            self, mask_feats, mask_feat_stride, instances
    ):
        _, H, W = mask_feats.size()
        locations = compute_locations(
            H, W,
            stride=mask_feat_stride, device=mask_feats.device
        )
        n_inst = len(instances)

        mask_head_params = instances.mask_head_params

        if not self.disable_rel_coords:
            # TODO realize this part
            # 計算相對座標
            instance_locations = instances.locations
            relative_coords = instance_locations.reshape(-1, 1, 2) - locations.reshape(1, -1, 2)
            relative_coords = relative_coords.permute(0, 2, 1).float()
            # TODO why sizes_of_interest
            self.sizes_of_interest = relative_coords.new_tensor([8, 16, 32, 64, 128])
            soi = self.sizes_of_interest.float()[instances.fpn_levels]
            relative_coords = relative_coords / soi.reshape(-1, 1, 1)
            relative_coords = relative_coords.to(dtype=mask_feats.dtype)

            mask_feats = mask_feats.repeat(n_inst, 1, 1, 1)
            mask_head_inputs = torch.cat([
                relative_coords, mask_feats.reshape(n_inst, self.in_channels, H * W)
            ], dim=1)
        else:
            mask_feats = mask_feats.repeat(n_inst, 1, 1, 1)
            mask_head_inputs = mask_feats.reshape(n_inst, self.in_channels, H * W)
        mask_head_inputs = mask_head_inputs.reshape(1, -1, H, W)
        weights, biases = parse_dynamic_params(
            mask_head_params, self.channels,
            self.weight_nums, self.bias_nums
        )

        mask_logits = self.mask_heads_forward(mask_head_inputs, weights, biases, n_inst)

        mask_logits = mask_logits.reshape(-1, 1, H, W)

        assert mask_feat_stride >= self.mask_out_stride
        assert mask_feat_stride % self.mask_out_stride == 0
        mask_logits = aligned_bilinear(mask_logits, int(mask_feat_stride / self.mask_out_stride))

        return mask_logits.sigmoid()

    @torch.no_grad()
    def compute_mask_iou_matrix(self, mask_scores, gt_bitmasks):

        pred_mask = (mask_scores > 0.5).float()
        bit_masks = (gt_bitmasks > 0.5).float()
        n = pred_mask.shape[0]
        bit_masks = bit_masks.unsqueeze(1)
        inter = (pred_mask * bit_masks).view(n, -1).sum(dim=1)
        value = (pred_mask.view(n, -1).sum(dim=1) + bit_masks.view(n, -1).sum(dim=1) - inter + 1e-5)
        overlaps = inter / value
        return overlaps