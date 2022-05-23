import logging
import torch
from torch import nn

from torch.nn import functional as F
from .utilize import iou_loss
from .utilize import permute_all_cls_and_filter_to_N_HWA_K_and_concat
from . import comm
import math
from typing import List, Dict
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn
from .utilize import build_shift_generator
from adet.utils.comm import compute_locations

from detectron2.modeling.proposal_generator.build import PROPOSAL_GENERATOR_REGISTRY
from .utilize import generalized_batched_nms, Shift2BoxTransform
from .utilize import Scale
from detectron2.layers import ShapeSpec, cat

from detectron2.modeling.meta_arch.retinanet import permute_to_N_HWA_K
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
import pdb

logger = logging.getLogger(__name__)

INF = 100000000

def focal_loss(
    probs,
    targets,
    alpha: float = -1,
    gamma: float = 2,
    reduction: str = "none",
):
    ce_loss = F.binary_cross_entropy(
        probs, targets, reduction="none"
    )
    p_t = probs * targets + (1 - probs) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss

focal_loss_jit = torch.jit.script(focal_loss)
"""
Shape shorthand in this module:

    N: number of images in the minibatch
    L: number of feature maps per image on which RPN is run
    Hi, Wi: height and width of the i-th feature map
    4: size of the box parameterization

Naming convention:

    labels: refers to the ground-truth class of an position.

    reg_targets: refers to the 4-d (left, top, right, bottom) distances that parameterize the ground-truth box.

    logits_pred: predicted classification scores in [-inf, +inf];

    reg_pred: the predicted (left, top, right, bottom), corresponding to reg_targets 

    ctrness_pred: predicted centerness scores

"""


class Uni_fcos_outputs(nn.Module):
    def __init__(self, cfg):
        super(Uni_fcos_outputs, self).__init__()
        # fmt: off
        self.num_classes = cfg.MODEL.FCOS.NUM_CLASSES
        self.in_features = cfg.MODEL.FCOS.IN_FEATURES
        self.fpn_strides = cfg.MODEL.FCOS.FPN_STRIDES
        # Loss parameters:
        self.focal_loss_alpha = cfg.MODEL.FCOS.FOCAL_LOSS_ALPHA
        self.focal_loss_gamma = cfg.MODEL.FCOS.FOCAL_LOSS_GAMMA
        self.oyor_aux_topk = cfg.MODEL.OYOR.AUX_TOPK
        # Inference parameters:
        self.nms_type = cfg.MODEL.NMS_TYPE
        self.max_detections_per_image = cfg.TEST.DETECTIONS_PER_IMAGE

        # Matching and loss
        self.shift2box_transform = Shift2BoxTransform(
            weights=cfg.MODEL.FCOS.BBOX_REG_WEIGHTS)
        self.center_sampling_radius = cfg.MODEL.OYOR.CENTER_SAMPLING_RADIUS
        self.instance_assign = cfg.MODEL.OYOR.ASSIGN
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.beta = cfg.MODEL.OYOR.MASK_BETA

        # generate sizes of interest
        soi = []
        prev_size = -1
        for s in cfg.MODEL.FCOS.SIZES_OF_INTEREST:
            soi.append([prev_size, s])
            prev_size = s
        soi.append([prev_size, INF])
        self.sizes_of_interest = soi

    def losses(self, gt_classes, pred_class_logits, pred_filtering):
        """

        Args:
            For `gt_classes` and `gt_shifts_deltas` parameters, see
                :meth:`FCOS.get_ground_truth`.
            Their shapes are (N, R) and (N, R, 4), respectively, where R is
            the total number of shifts across levels, i.e. sum(Hi x Wi)
            For `pred_class_logits`, `pred_shift_deltas` and `pred_fitering`, see
                :meth:`FCOSHead.forward`.

        Returns:
            dict[str: Tensor]:
                mapping from a named loss to a scalar tensor
                storing the loss. Used during training only. The dict keys are:
                "loss_cls" and "loss_box_reg"
        """
        pred_class_logits, pred_filtering = \
            permute_all_cls_and_filter_to_N_HWA_K_and_concat(
                pred_class_logits, pred_filtering,
                self.num_classes
            )  # Shapes: (N x R, K) and (N x R, 4), respectively.

        gt_classes = gt_classes.flatten()
        valid_idxs = gt_classes >= 0
        foreground_idxs = (gt_classes >= 0) & (gt_classes != self.num_classes)
        num_foreground = foreground_idxs.sum()

        gt_classes_target = torch.zeros_like(pred_class_logits)
        gt_classes_target[foreground_idxs, gt_classes[foreground_idxs]] = 1

        num_foreground = comm.all_reduce(num_foreground) / float(comm.get_world_size())

        pred_class_logits = pred_class_logits.sigmoid() * pred_filtering.sigmoid()

        # logits loss
        loss_cls = focal_loss_jit(
            pred_class_logits[valid_idxs],
            gt_classes_target[valid_idxs],
            alpha=self.focal_loss_alpha,
            gamma=self.focal_loss_gamma,
            reduction="sum",
        ) / max(1.0, num_foreground)

        return {
            "loss_cls": loss_cls
        }

    def get_ground_truth(self, shifts, targets, box_cls, iou_preds, box_filter, top_feats, locations, mask_cost_func=None, mask_feats=None):
        """
        Args:
            shifts (list[list[Tensor]]): a list of N=#image elements. Each is a
                list of #feature level tensors. The tensors contains shifts of
                this image on the specific feature level.
            targets (list[Instances]): a list of N `Instances`s. The i-th
                `Instances` contains the ground-truth per-instance annotations
                for the i-th input image.  Specify `targets` during training only.

        Returns:
            gt_classes (Tensor):
                An integer tensor of shape (N, R) storing ground-truth
                labels for each shift.
                R is the total number of shifts, i.e. the sum of Hi x Wi for all levels.
                Shifts in the valid boxes are assigned their corresponding label in the
                [0, K-1] range. Shifts in the background are assigned the label "K".
                Shifts in the ignore areas are assigned a label "-1", i.e. ignore.
            gt_shifts_deltas (Tensor):
                Shape (N, R, 4).
                The last dimension represents ground-truth shift2box transform
                targets (dl, dt, dr, db) that map each shift to its matched ground-truth box.
                The values in the tensor are meaningful only when the corresponding
                shift is labeled as foreground.
        """
        gt_classes = []
        isinstance_list = []
        IoU_results_list = []
        fpn_level = [
            loc.new_ones(len(loc), dtype=torch.long) * level
            for level, loc in enumerate(locations)
        ]

        box_cls = torch.cat([permute_to_N_HWA_K(x, self.num_classes) for x in box_cls], dim=1).detach()
        iou_preds = torch.cat([permute_to_N_HWA_K(x, 1) for x in iou_preds], dim=1)
        box_filter = torch.cat([permute_to_N_HWA_K(x, 1) for x in box_filter], dim=1)
        top_feats = torch.cat([permute_to_N_HWA_K(x, self.num_gen_params) for x in top_feats], dim=1)
        box_cls = box_cls.sigmoid_() * box_filter.sigmoid_()
        fpn_level = torch.cat(fpn_level, dim=0).detach()
        locations_fcos = torch.cat(locations, dim=0).detach()

        fpn_level = [fpn_level.clone() for _ in range(len(box_cls))]
        locations = [locations_fcos.clone() for _ in range(len(box_cls))]
        fpn_level = torch.stack(fpn_level).detach()
        locations = torch.stack(locations).detach()

        for im_id, (shifts_per_image, targets_per_image, box_cls_per_image, iou_preds_per_image, top_feats_per_image, locations_per_image, fpn_level_per_image) in enumerate(zip(
                shifts, targets, box_cls,iou_preds, top_feats, locations, fpn_level)):
            shifts_over_all_feature_maps = torch.cat(shifts_per_image, dim=0).detach()

            gt_boxes = targets_per_image.gt_boxes
            gt_bitmasks_full = targets_per_image.gt_bitmasks_full
            gt_bitmasks = targets_per_image.gt_bitmasks


            prob = box_cls_per_image[:, targets_per_image.gt_classes].t()
            if self.instance_assign == 'OYOR' :
                quality = prob ** (1 - self.beta)
                costs = torch.ones_like(quality)
                IoUs = torch.ones_like(quality)

            if self.center_sampling_radius > 0 and self.instance_assign == 'OYOR':
                centers = self.center(gt_bitmasks_full)
                is_in_boxes = []
                for stride, shifts_i in zip(self.fpn_strides, shifts_per_image):
                    radius = stride * self.center_sampling_radius # 放大幾倍
                    center_boxes = torch.cat((
                        torch.max(centers - radius, gt_boxes.tensor[:, :2]),
                        torch.min(centers + radius, gt_boxes.tensor[:, 2:]),
                    ), dim=-1)

                    center_deltas = self.shift2box_transform.get_deltas(
                        shifts_i, center_boxes.unsqueeze(1))
                    is_in_boxes.append(center_deltas.min(dim=-1).values > 0)
                is_in_boxes = torch.cat(is_in_boxes, dim=1)

            for i in range(len(targets_per_image)):
                index_cand = (is_in_boxes[i] == True).nonzero().squeeze(1)
                cand = Instances((0, 0))
                indexs_one = torch.zeros(index_cand.shape, dtype=torch.long).to('cuda')
                cand.gt_inds = indexs_one.to('cuda')  # add pre
                cand.mask_head_params = top_feats_per_image[index_cand]
                cand.locations = locations_per_image[index_cand]
                cand.fpn_levels = fpn_level_per_image[index_cand]
                cand.im_inds = index_cand.new_full(index_cand.shape, im_id).to('cuda')
                cand.gt_bitmasks = targets_per_image[i].gt_bitmasks.repeat(len(index_cand), 1, 1)
                cost, iou = mask_cost_func(mask_feats[im_id], cand)
                costs[i, index_cand] = cost
                IoUs[i, index_cand] = iou


            # instance, feature points
            if self.instance_assign == 'OYOR' :
                quality = quality * (costs ** self.beta)

            quality[~is_in_boxes] = -1
            need = []
            for i, row in enumerate(quality):
                if torch.any(row != -1):
                    need.append(i)

            quality = quality[need]
            targets_per_image = targets_per_image[need]
            gt_bitmasks_full = gt_bitmasks_full[need]
            gt_bitmasks = gt_bitmasks[need]
            IoUs = IoUs[need]
            is_in_boxes = is_in_boxes[need]

            gt_idxs, shift_idxs = linear_sum_assignment(quality.detach().cpu().numpy(), maximize=True)

            # TODO 抓極度重疊
            # 思路： 萬一找到隨機剔除一個, 非直接解法
            if len((quality[gt_idxs, shift_idxs] == -1).nonzero()) > 0:
                index = (quality[gt_idxs, shift_idxs] != -1).nonzero().squeeze(1)
                quality = quality[index]
                targets_per_image = targets_per_image[index]
                gt_bitmasks_full = gt_bitmasks_full[index]
                gt_bitmasks = gt_bitmasks[index]
                IoUs = IoUs[index]
                is_in_boxes = is_in_boxes [index]
                print('too small and duplicate')
                gt_idxs, shift_idxs = linear_sum_assignment(quality.detach().cpu().numpy(), maximize=True)

            gt_classes_i = shifts_over_all_feature_maps.new_full(
                (len(shifts_over_all_feature_maps),), self.num_classes, dtype=torch.long
            )
            gt_im_index_i = shifts_over_all_feature_maps.new_full(
                (len(shifts_over_all_feature_maps), ), im_id, dtype=torch.long
            )

            # instance 保存的，predict_logit, predict_bbox_delta, gt_ind, im_ind
            instances = None
            if len(targets_per_image) > 0:
                # ground truth classes
                gt_classes_i[shift_idxs] = targets_per_image.gt_classes[gt_idxs]
                instances = Instances((0, 0))
                instances.labels = targets_per_image.gt_classes[gt_idxs]
                instances.gt_inds = torch.from_numpy(gt_idxs).to('cuda') # add pre
                instances.im_inds = gt_im_index_i[gt_idxs]
                instances.top_feat = top_feats_per_image[shift_idxs]
                instances.locations = locations_per_image[shift_idxs]
                instances.fpn_levels = fpn_level_per_image[shift_idxs]
                instances.gt_bitmasks = gt_bitmasks[gt_idxs]
                instances.gt_bitmasks_full = gt_bitmasks_full[gt_idxs]

                instances.pred_ious = iou_preds_per_image[shift_idxs]
                instances.gt_ious = IoUs[gt_idxs, shift_idxs]
                # len = instance
                # 裡面裝著IoU的預測學習
                IoU_container = Instances((0, 0))
                other_predict = []
                other_gt = []
                for i in range(len(is_in_boxes)):
                    ious_cand = (is_in_boxes[i] == True).nonzero().squeeze(1)
                    other_predict.append(iou_preds_per_image[ious_cand])
                    other_gt.append(IoUs[i][ious_cand])
                other_predict = torch.cat(other_predict)
                other_gt = torch.cat(other_gt)
                IoU_container.other_predict = other_predict
                IoU_container.other_gt = other_gt

            gt_classes.append(gt_classes_i)

            if instances:
                isinstance_list.append(instances)
                IoU_results_list.append(IoU_container)
            else:
                print('no target train')

        results = dict()
        results['instances'] = Instances.cat(isinstance_list)
        IoU_results = Instances.cat(IoU_results_list)

        return torch.stack(gt_classes), results, IoU_results


    @torch.no_grad()
    def get_aux_ground_truth(self, shifts, targets, box_cls, top_feats, locations, mask_cost_func=None, mask_feats=None):
        """
        Args:
            shifts (list[list[Tensor]]): a list of N=#image elements. Each is a
                list of #feature level tensors. The tensors contains shifts of
                this image on the specific feature level.
            targets (list[Instances]): a list of N `Instances`s. The i-th
                `Instances` contains the ground-truth per-instance annotations
                for the i-th input image.  Specify `targets` during training only.

        Returns:
            gt_classes (Tensor):
                An integer tensor of shape (N, R) storing ground-truth
                labels for each shift.
                R is the total number of shifts, i.e. the sum of Hi x Wi for all levels.
                Shifts in the valid boxes are assigned their corresponding label in the
                [0, K-1] range. Shifts in the background are assigned the label "K".
                Shifts in the ignore areas are assigned a label "-1", i.e. ignore.
        """
        gt_classes = []

        fpn_level = [
            loc.new_ones(len(loc), dtype=torch.long) * level
            for level, loc in enumerate(locations)
        ]

        box_cls = torch.cat([permute_to_N_HWA_K(x, self.num_classes) for x in box_cls], dim=1)
        top_feats = torch.cat([permute_to_N_HWA_K(x, self.num_gen_params) for x in top_feats], dim=1)
        box_cls = box_cls.sigmoid_()
        fpn_level = torch.cat(fpn_level, dim=0)
        locations_fcos = torch.cat(locations, dim=0)# all_level,2

        fpn_level = [fpn_level.clone() for _ in range(len(box_cls))]# 16, all_level,2
        locations = [locations_fcos.clone() for _ in range(len(box_cls))]# 16, all_level,2
        fpn_level = torch.stack(fpn_level)
        locations = torch.stack(locations)



        for im_id, (shifts_per_image, targets_per_image, box_cls_per_image, top_feats_per_image, locations_per_image, fpn_level_per_image) in enumerate(zip(
                shifts, targets, box_cls, top_feats, locations, fpn_level)):
            gt_boxes = targets_per_image.gt_boxes
            gt_bitmasks_full = targets_per_image.gt_bitmasks_full

            prob = box_cls_per_image[:, targets_per_image.gt_classes].t()

            quality = prob ** (1 - self.beta)
            centers = self.center(gt_bitmasks_full)
            costs = torch.ones_like(quality)
            is_in_boxes = []
            is_in_masks = []
            for stride, shifts_i in zip(self.fpn_strides, shifts_per_image):
                radius = stride * self.center_sampling_radius  # 放大幾倍
                center_boxes = torch.cat((
                    torch.max(centers - radius, gt_boxes.tensor[:, :2]),
                    torch.min(centers + radius, gt_boxes.tensor[:, 2:]),
                ), dim=-1)

                tmp = []
                for i, each in enumerate(gt_bitmasks_full):
                    height, width = each.shape
                    width_x = torch.clamp(shifts_i.type(torch.LongTensor)[:, 0], max=width)
                    height_y = torch.clamp(shifts_i.type(torch.LongTensor)[:, 1], max=height)
                    is_in_mask = each[height_y - 1, width_x - 1] > 0
                    tmp.append(is_in_mask)

                center_deltas = self.shift2box_transform.get_deltas(
                    shifts_i, center_boxes.unsqueeze(1))
                is_in_boxes.append(center_deltas.min(dim=-1).values > 0)
                is_in_masks.append(torch.stack(tmp))

            is_in_boxes = torch.cat(is_in_boxes, dim=1)
            is_in_masks = torch.cat(is_in_masks, dim=1)

            is_in_boxes = is_in_boxes & is_in_masks

            for i in range(len(targets_per_image)):
                index_cand = (is_in_boxes[i] == True).nonzero().squeeze(1)
                # assert len(index_cand) > 0
                cand = Instances((0, 0))
                indexs_one = torch.zeros(index_cand.shape, dtype=torch.long).to('cuda')
                gt = targets_per_image[i]
                gt.in_mask = [locations_per_image[index_cand]]
                # 所有的In mask的點
                cand.gt_inds = indexs_one.to('cuda')  # add pre
                cand.mask_head_params = top_feats_per_image[index_cand]
                cand.locations = locations_per_image[index_cand]
                cand.fpn_levels = fpn_level_per_image[index_cand]
                cand.im_inds = index_cand.new_full(index_cand.shape, im_id).to('cuda')
                cand.gt_bitmasks = targets_per_image[i].gt_bitmasks.repeat(len(index_cand), 1, 1)
                cost, iou = mask_cost_func(mask_feats[im_id], cand)
                costs[i, index_cand] = cost

            quality = quality * (costs ** self.beta)
            quality[~is_in_boxes] = -1

            need = []
            for i, row in enumerate(quality):
                if torch.any(row != -1):
                    need.append(i)

            quality = quality[need]
            targets_per_image = targets_per_image[need]

            candidate_idxs = []
            st, ed = 0, 0
            for shifts_i in shifts_per_image:
                ed += len(shifts_i)
                _, topk_idxs = quality[:, st:ed].topk(self.oyor_aux_topk, dim=1)
                candidate_idxs.append(st + topk_idxs)
                st = ed

            candidate_idxs = torch.cat(candidate_idxs, dim=1)

            candidate_qualities = quality.gather(1, candidate_idxs)

            quality_thr = candidate_qualities.mean(dim=1, keepdim=True)
                          # + candidate_qualities.std(dim=1, keepdim=True)


            is_foreground = torch.zeros_like(quality, dtype=torch.bool).scatter_(1, candidate_idxs, True).to(is_in_boxes.device)
            is_foreground &= quality >= quality_thr

            quality[~is_foreground] = -1

            need = []
            for i, row in enumerate(quality):
                if torch.any(row != -1):
                    need.append(i)

            quality = quality[need]
            targets_per_image = targets_per_image[need]


            # if there are still more than one objects for a position,
            # we choose the one with maximum quality
            if len(quality) != 0 :
                positions_max_quality, gt_matched_idxs = quality.max(dim=0)
                # TODO 用這個思路修正小目標

            # ground truth classes
            has_gt = len(targets_per_image) > 0
            if has_gt:
                gt_classes_i = targets_per_image.gt_classes[gt_matched_idxs]
                # Shifts with quality -1 are treated as background.
                gt_classes_i[positions_max_quality == -1] = self.num_classes
            else:
                gt_classes_i = torch.zeros(
                    quality.size(1), dtype=torch.bool).to(is_in_boxes.device) + self.num_classes

            gt_classes.append(gt_classes_i)

        return torch.stack(gt_classes)


    def _transpose(self, training_targets, num_loc_list):
        '''
        This function is used to transpose image first training targets to level first ones
        :return: level first training targets
        '''
        for im_i in range(len(training_targets)):
            training_targets[im_i] = torch.split(
                training_targets[im_i], num_loc_list, dim=0
            )

        targets_level_first = []
        for targets_per_level in zip(*training_targets):
            targets_level_first.append(
                torch.cat(targets_per_level, dim=0)
            )
        return targets_level_first

    def center(self, bitmasks) :

        _, h, w = bitmasks.size()
        ys = torch.arange(0, h, dtype=torch.float32, device=bitmasks.device)
        xs = torch.arange(0, w, dtype=torch.float32, device=bitmasks.device)

        m00 = bitmasks.sum(dim=-1).sum(dim=-1).clamp(min=1e-6)
        m10 = (bitmasks * xs).sum(dim=-1).sum(dim=-1)
        m01 = (bitmasks * ys[:, None]).sum(dim=-1).sum(dim=-1)
        center_x = m10 / m00
        center_y = m01 / m00
        return torch.stack([center_x, center_y], dim=1)