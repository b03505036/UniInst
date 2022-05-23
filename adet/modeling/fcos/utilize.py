import torch
from torch import nn
from torchvision.ops import boxes as box_ops
from torchvision.ops import nms  # BC-compat
from detectron2.modeling.meta_arch.retinanet import permute_to_N_HWA_K
from typing import List, Dict
from detectron2.layers import ShapeSpec, cat
import math
import copy
from detectron2.structures.boxes import Boxes

def batched_nms(boxes, scores, idxs, iou_threshold):
    """
    Same as torchvision.ops.boxes.batched_nms, but safer.
    """
    assert boxes.shape[-1] == 4
    # TODO may need better strategy.
    # Investigate after having a fully-cuda NMS op.
    if len(boxes) < 40000:
        return box_ops.batched_nms(boxes, scores, idxs, iou_threshold)

    result_mask = scores.new_zeros(scores.size(), dtype=torch.bool)
    for id in torch.unique(idxs).cpu().tolist():
        mask = (idxs == id).nonzero(as_tuple=False).view(-1)
        keep = nms(boxes[mask], scores[mask], iou_threshold)
        result_mask[mask[keep]] = True
    keep = result_mask.nonzero(as_tuple=False).view(-1)
    keep = keep[scores[keep].argsort(descending=True)]
    return keep

def generalized_batched_nms(boxes, scores, idxs, iou_threshold,
                            score_threshold=0.025, nms_type="normal"):
    assert boxes.shape[-1] == 4

    if nms_type == "normal":
        keep = batched_nms(boxes, scores, idxs, iou_threshold)
    elif nms_type.startswith("softnms"):
        keep = batched_softnms(boxes, scores, idxs, iou_threshold,
                               score_threshold=score_threshold)
    elif nms_type == "cluster":
        keep = batched_clusternms(boxes, scores, idxs, iou_threshold)
    else:
        raise NotImplementedError("NMS type not implemented: \"{}\"".format(nms_type))

    return keep

def batched_clusternms(boxes, scores, idxs, iou_threshold):
    assert boxes.shape[-1] == 4

    result_mask = scores.new_zeros(scores.size(), dtype=torch.bool)
    for id in torch.unique(idxs).cpu().tolist():
        mask = (idxs == id).nonzero(as_tuple=False).view(-1)
        keep = cluster_nms(boxes[mask], scores[mask], iou_threshold)
        result_mask[mask[keep]] = True
    keep = result_mask.nonzero(as_tuple=False).view(-1)
    keep = keep[scores[keep].argsort(descending=True)]
    return keep

def softnms(boxes, scores, sigma, score_threshold, soft_mode="gaussian"):
    assert soft_mode in ["linear", "gaussian"]

    undone_mask = scores >= score_threshold
    while undone_mask.sum() > 1:
        idx = scores[undone_mask].argmax()
        idx = undone_mask.nonzero(as_tuple=False)[idx].item()
        top_box = boxes[idx]
        undone_mask[idx] = False
        _boxes = boxes[undone_mask]

        ious = iou(_boxes, top_box)
        scales = scale_by_iou(ious, sigma, soft_mode)

        scores[undone_mask] *= scales
        undone_mask[scores < score_threshold] = False
    return scores

def iou(boxes, top_box):
    x1 = boxes[:, 0].clamp(min=top_box[0])
    y1 = boxes[:, 1].clamp(min=top_box[1])
    x2 = boxes[:, 2].clamp(max=top_box[2])
    y2 = boxes[:, 3].clamp(max=top_box[3])

    inters = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    unions = (top_box[2] - top_box[0]) * (top_box[3] - top_box[1]) + areas - inters

    return inters / unions

def scale_by_iou(ious, sigma, soft_mode="gaussian"):
    if soft_mode == "linear":
        scale = ious.new_ones(ious.size())
        scale[ious >= sigma] = 1 - ious[ious >= sigma]
    else:
        scale = torch.exp(-ious ** 2 / sigma)

    return scale

def batched_softnms(boxes, scores, idxs, iou_threshold,
                    score_threshold=0.001, soft_mode="gaussian"):
    assert soft_mode in ["linear", "gaussian"]
    assert boxes.shape[-1] == 4

    # change scores inplace
    # no need to return changed scores
    for id in torch.unique(idxs).cpu().tolist():
        mask = (idxs == id).nonzero(as_tuple=False).view(-1)
        scores[mask] = softnms(boxes[mask], scores[mask], iou_threshold,
                               score_threshold, soft_mode)

    keep = (scores > score_threshold).nonzero(as_tuple=False).view(-1)
    keep = keep[scores[keep].argsort(descending=True)]
    return keep

def cluster_nms(boxes, scores, iou_threshold):
    last_keep = torch.ones(*scores.shape).to(boxes.device)

    scores, idx = scores.sort(descending=True)
    boxes = boxes[idx]
    origin_iou_matrix = box_ops.box_iou(boxes, boxes).tril(diagonal=-1).transpose(1, 0)

    while True:
        iou_matrix = torch.mm(torch.diag(last_keep.float()), origin_iou_matrix)
        keep = (iou_matrix.max(dim=0)[0] <= iou_threshold)

        if (keep == last_keep).all():
            return idx[keep.nonzero(as_tuple=False)]

        last_keep = keep

class Shift2BoxTransform(object):
    def __init__(self, weights):
        """
        Args:
            weights (4-element tuple): Scaling factors that are applied to the
                (dl, dt, dr, db) deltas.
        """
        self.weights = weights

    def get_deltas(self, shifts, boxes):
        """
        Get box regression transformation deltas (dl, dt, dr, db) that can be used
        to transform the `shifts` into the `boxes`. That is, the relation
        ``boxes == self.apply_deltas(deltas, shifts)`` is true.
        Args:
            shifts (Tensor): shifts, e.g., feature map coordinates
            boxes (Tensor): target of the transformation, e.g., ground-truth
                boxes.
        """
        assert isinstance(shifts, torch.Tensor), type(shifts)
        assert isinstance(boxes, torch.Tensor), type(boxes)

        deltas = torch.cat((shifts - boxes[..., :2], boxes[..., 2:] - shifts),
                           dim=-1) * shifts.new_tensor(self.weights)
        return deltas

    def apply_deltas(self, deltas, shifts):
        """
        Apply transformation `deltas` (dl, dt, dr, db) to `shifts`.
        Args:
            deltas (Tensor): transformation deltas of shape (N, k*4), where k >= 1.
                deltas[i] represents k potentially different class-specific
                box transformations for the single shift shifts[i].
            shifts (Tensor): shifts to transform, of shape (N, 2)
        """
        assert torch.isfinite(deltas).all().item()
        shifts = shifts.to(deltas.dtype)

        if deltas.numel() == 0:
            return torch.empty_like(deltas)

        deltas = deltas.view(deltas.size()[:-1] + (-1, 4)) / shifts.new_tensor(self.weights)
        boxes = torch.cat((shifts.unsqueeze(-2) - deltas[..., :2],
                           shifts.unsqueeze(-2) + deltas[..., 2:]),
                          dim=-1).view(deltas.size()[:-2] + (-1, ))
        return boxes


def permute_all_cls_and_box_to_N_HWA_K_and_concat(
    box_cls, box_delta, box_center, num_classes=80
):
    """
    Rearrange the tensor layout from the network output, i.e.:
    list[Tensor]: #lvl tensors of shape (N, A x K, Hi, Wi)
    to per-image predictions, i.e.:
    Tensor: of shape (N x sum(Hi x Wi x A), K)
    """
    # for each feature level, permute the outputs to make them be in the
    # same format as the labels. Note that the labels are computed for
    # all feature levels concatenated, so we keep the same representation
    # for the objectness, the box_delta and the centerness
    box_cls_flattened = [permute_to_N_HWA_K(x, num_classes) for x in box_cls]
    box_delta_flattened = [permute_to_N_HWA_K(x, 4) for x in box_delta]
    box_center_flattened = [permute_to_N_HWA_K(x, 1) for x in box_center]
    # concatenate on the first dimension (representing the feature levels), to
    # take into account the way the labels were generated (with all feature maps
    # being concatenated as well)
    box_cls = cat(box_cls_flattened, dim=1).view(-1, num_classes)
    box_delta = cat(box_delta_flattened, dim=1).view(-1, 4)
    box_center = cat(box_center_flattened, dim=1).view(-1, 1)
    return box_cls, box_delta, box_center

def permute_all_cls_and_filter_to_N_HWA_K_and_concat(
    box_cls, box_center, num_classes=80
):
    """
    Rearrange the tensor layout from the network output, i.e.:
    list[Tensor]: #lvl tensors of shape (N, A x K, Hi, Wi)
    to per-image predictions, i.e.:
    Tensor: of shape (N x sum(Hi x Wi x A), K)
    """
    # for each feature level, permute the outputs to make them be in the
    # same format as the labels. Note that the labels are computed for
    # all feature levels concatenated, so we keep the same representation
    # for the objectness, the box_delta and the centerness
    box_cls_flattened = [permute_to_N_HWA_K(x, num_classes) for x in box_cls]
    # box_delta_flattened = [permute_to_N_HWA_K(x, 4) for x in box_delta]
    box_center_flattened = [permute_to_N_HWA_K(x, 1) for x in box_center]
    # concatenate on the first dimension (representing the feature levels), to
    # take into account the way the labels were generated (with all feature maps
    # being concatenated as well)
    box_cls = cat(box_cls_flattened, dim=1).view(-1, num_classes)
    # box_delta = cat(box_delta_flattened, dim=1).view(-1, 4)
    box_center = cat(box_center_flattened, dim=1).view(-1, 1)
    return box_cls, box_center

def permute_all_cls_and_filter_and_iou_to_N_HWA_K_and_concat(
    box_cls, box_center, pred_ious, num_classes=80
):
    """
    Rearrange the tensor layout from the network output, i.e.:
    list[Tensor]: #lvl tensors of shape (N, A x K, Hi, Wi)
    to per-image predictions, i.e.:
    Tensor: of shape (N x sum(Hi x Wi x A), K)
    """
    # for each feature level, permute the outputs to make them be in the
    # same format as the labels. Note that the labels are computed for
    # all feature levels concatenated, so we keep the same representation
    # for the objectness, the box_delta and the centerness
    box_cls_flattened = [permute_to_N_HWA_K(x, num_classes) for x in box_cls]
    # box_delta_flattened = [permute_to_N_HWA_K(x, 4) for x in box_delta]
    box_center_flattened = [permute_to_N_HWA_K(x, 1) for x in box_center]
    iou_pred_flattened = [permute_to_N_HWA_K(x, 1) for x in pred_ious]
    # concatenate on the first dimension (representing the feature levels), to
    # take into account the way the labels were generated (with all feature maps
    # being concatenated as well)
    box_cls = cat(box_cls_flattened, dim=1).view(-1, num_classes)
    # box_delta = cat(box_delta_flattened, dim=1).view(-1, 4)
    box_center = cat(box_center_flattened, dim=1).view(-1, 1)
    pred_ious = cat(iou_pred_flattened, dim=1).view(-1, 1)
    return box_cls, box_center, pred_ious

def permute_all_cls_to_N_HWA_K_and_concat(
    box_cls, num_classes=80
):
    """
    Rearrange the tensor layout from the network output, i.e.:
    list[Tensor]: #lvl tensors of shape (N, A x K, Hi, Wi)
    to per-image predictions, i.e.:
    Tensor: of shape (N x sum(Hi x Wi x A), K)
    """
    # for each feature level, permute the outputs to make them be in the
    # same format as the labels. Note that the labels are computed for
    # all feature levels concatenated, so we keep the same representation
    # for the objectness, the box_delta and the centerness
    box_cls_flattened = [permute_to_N_HWA_K(x, num_classes) for x in box_cls]
    # box_delta_flattened = [permute_to_N_HWA_K(x, 4) for x in box_delta]
    # box_center_flattened = [permute_to_N_HWA_K(x, 1) for x in box_center]
    # concatenate on the first dimension (representing the feature levels), to
    # take into account the way the labels were generated (with all feature maps
    # being concatenated as well)
    box_cls = cat(box_cls_flattened, dim=1).view(-1, num_classes)
    # box_delta = cat(box_delta_flattened, dim=1).view(-1, 4)
    # box_center = cat(box_center_flattened, dim=1).view(-1, 1)
    return box_cls


class Scale(nn.Module):
    def __init__(self, init_value=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale


def iou_loss(
    inputs,
    targets,
    weight=None,
    box_mode="xyxy",
    loss_type="iou",
    smooth=False,
    reduction="none"
):
    """
    Compute iou loss of type ['iou', 'giou', 'linear_iou']
    Args:
        inputs (tensor): pred values
        targets (tensor): target values
        weight (tensor): loss weight
        box_mode (str): 'xyxy' or 'ltrb', 'ltrb' is currently supported.
        loss_type (str): 'giou' or 'iou' or 'linear_iou'
        reduction (str): reduction manner
    Returns:
        loss (tensor): computed iou loss.
    """
    if box_mode == "ltrb":
        inputs = torch.cat((-inputs[..., :2], inputs[..., 2:]), dim=-1)
        targets = torch.cat((-targets[..., :2], targets[..., 2:]), dim=-1)
    elif box_mode != "xyxy":
        raise NotImplementedError

    eps = torch.finfo(torch.float32).eps

    inputs_area = (inputs[..., 2] - inputs[..., 0]).clamp_(min=0) \
        * (inputs[..., 3] - inputs[..., 1]).clamp_(min=0)
    targets_area = (targets[..., 2] - targets[..., 0]).clamp_(min=0) \
        * (targets[..., 3] - targets[..., 1]).clamp_(min=0)

    w_intersect = (torch.min(inputs[..., 2], targets[..., 2])
                   - torch.max(inputs[..., 0], targets[..., 0])).clamp_(min=0)
    h_intersect = (torch.min(inputs[..., 3], targets[..., 3])
                   - torch.max(inputs[..., 1], targets[..., 1])).clamp_(min=0)

    area_intersect = w_intersect * h_intersect
    area_union = targets_area + inputs_area - area_intersect
    if smooth:
        ious = (area_intersect + 1) / (area_union + 1)
    else:
        ious = area_intersect / area_union.clamp(min=eps)

    if loss_type == "iou":
        loss = -ious.clamp(min=eps).log()
    elif loss_type == "linear_iou":
        loss = 1 - ious
    elif loss_type == "giou":
        g_w_intersect = torch.max(inputs[..., 2], targets[..., 2]) \
            - torch.min(inputs[..., 0], targets[..., 0])
        g_h_intersect = torch.max(inputs[..., 3], targets[..., 3]) \
            - torch.min(inputs[..., 1], targets[..., 1])
        ac_uion = g_w_intersect * g_h_intersect
        gious = ious - (ac_uion - area_union) / ac_uion.clamp(min=eps)
        loss = 1 - gious
    else:
        raise NotImplementedError
    if weight is not None:
        loss = loss * weight.view(loss.size())
        if reduction == "mean":
            loss = loss.sum() / max(weight.sum().item(), eps)
    else:
        if reduction == "mean":
            loss = loss.mean()
    if reduction == "sum":
        loss = loss.sum()

    return loss

def build_shift_generator(cfg, input_shape):

    return ShiftGenerator(cfg, input_shape)

def _create_grid_offsets(size, stride, offset, device):
    grid_height, grid_width = size
    shifts_start = offset * stride
    # 共走 grid_width步
    # grid_width 可能是取整的，映設回原圖會超出特徵圖
    # 每步 stride 大小

    shifts_x = torch.arange(
        shifts_start, grid_width * stride + shifts_start, step=stride,
        dtype=torch.float32, device=device
    )
    shifts_y = torch.arange(
        shifts_start, grid_height * stride + shifts_start, step=stride,
        dtype=torch.float32, device=device
    )
    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    return shift_x, shift_y

class ShiftGenerator(nn.Module):
    """
    For a set of image sizes and feature maps, computes a set of shifts.
    """
    # TODO: unused_arguments: cfg
    def __init__(self, cfg, input_shape: List[ShapeSpec]):
        super().__init__()
        # fmt: off
        self.num_shifts = cfg.MODEL.SHIFT_GENERATOR.NUM_SHIFTS
        self.strides    = [x.stride for x in input_shape]
        self.offset     = cfg.MODEL.SHIFT_GENERATOR.OFFSET
        # fmt: on

        self.num_features = len(self.strides)

    @property
    def num_cell_shifts(self):
        return [self.num_shifts for _ in self.strides]

    def grid_shifts(self, grid_sizes, device):
        shifts_over_all = []
        for size, stride in zip(grid_sizes, self.strides):
            shift_x, shift_y = _create_grid_offsets(size, stride, self.offset, device)
            shifts = torch.stack((shift_x, shift_y), dim=1)

            shifts_over_all.append(shifts.repeat_interleave(self.num_shifts, dim=0))

        return shifts_over_all

    def forward(self, features):
        """
        Args:
            features (list[Tensor]): list of backbone feature maps on which to generate shifts.
        Returns:
            list[list[Tensor]]: a list of #image elements. Each is a list of #feature level tensors.
                The tensors contains shifts of this image on the specific feature level.
        """
        num_images = len(features[0])
        # grid_size = 特徵圖大小
        grid_sizes = [feature_map.shape[-2:] for feature_map in features]
        shifts_over_all = self.grid_shifts(grid_sizes, features[0].device)

        shifts = [copy.deepcopy(shifts_over_all) for _ in range(num_images)]
        return shifts


    def get_bounding_boxes(self, device):
        """
        Returns:
            Boxes: tight bounding boxes around bitmasks.
            If a mask is empty, it's bounding box will be all zero.
        """
        boxes = torch.zeros(self.tensor.shape[0], 4, dtype=torch.float32, device=device)
        x_any = torch.any(self.tensor, dim=1)
        y_any = torch.any(self.tensor, dim=2)
        for idx in range(self.tensor.shape[0]):
            x = torch.where(x_any[idx, :])[0]
            y = torch.where(y_any[idx, :])[0]
            if len(x) > 0 and len(y) > 0:
                boxes[idx, :] = torch.as_tensor(
                    [x[0], y[0], x[-1] + 1, y[-1] + 1], dtype=torch.float32
                )
        return Boxes(boxes)