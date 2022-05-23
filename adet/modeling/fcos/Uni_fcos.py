import math
from typing import List, Dict
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.nn import L1Loss
from .utilize import build_shift_generator
from adet.utils.comm import compute_locations
from . import comm
from .cvpods import sigmoid_focal_loss_jit
from detectron2.modeling.proposal_generator.build import PROPOSAL_GENERATOR_REGISTRY
from .utilize import generalized_batched_nms, Shift2BoxTransform
from .utilize import Scale
from detectron2.layers import ShapeSpec, cat

from detectron2.modeling.meta_arch.retinanet import permute_to_N_HWA_K
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from .Uni_fcos_outputs import Uni_fcos_outputs

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


focal_loss_jit = torch.jit.script(focal_loss)  # type: torch.jit.ScriptModule

@PROPOSAL_GENERATOR_REGISTRY.register()
class Uni_fcos(nn.Module):
    """
    Implement FCOS (https://arxiv.org/abs/1708.02002).
    """
    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super().__init__()

        # target
        self.in_features = cfg.MODEL.FCOS.IN_FEATURES
        self.fpn_strides = cfg.MODEL.FCOS.FPN_STRIDES
        self.yield_proposal = cfg.MODEL.FCOS.YIELD_PROPOSAL
        self.oym_head = Uni_fcos_head(cfg, [input_shape[f] for f in self.in_features])
        self.in_channels_to_top_module = self.oym_head.in_channels_to_top_module
        self.oym_outputs = Uni_fcos_outputs(cfg)
        self.device = torch.device(cfg.MODEL.DEVICE)

        # fmt: off
        self.num_classes = cfg.MODEL.FCOS.NUM_CLASSES
        self.in_features = cfg.MODEL.FCOS.IN_FEATURES
        self.fpn_strides = cfg.MODEL.FCOS.FPN_STRIDES
        # Loss parameters:
        self.focal_loss_alpha = cfg.MODEL.FCOS.FOCAL_LOSS_ALPHA
        self.focal_loss_gamma = cfg.MODEL.FCOS.FOCAL_LOSS_GAMMA
        self.iou_loss_type = cfg.MODEL.FCOS.IOU_LOSS_TYPE
        self.reg_weight = cfg.MODEL.FCOS.REG_WEIGHT
        self.aux_on = cfg.MODEL.OYOR.AUX_ON
        # Inference parameters:
        self.score_threshold = cfg.MODEL.OYOR.SCORE_THRESH_TEST
        self.topk_candidates = cfg.MODEL.FCOS.TOPK_CANDIDATES_TEST
        self.nms_threshold = cfg.MODEL.FCOS.NMS_THRESH_TEST
        self.nms_type = cfg.MODEL.OYOR.NMS_TYPE
        self.L1_loss = L1Loss()
        self.prediction_reranking = False
        # None
        self.max_detections_per_image = cfg.TEST.DETECTIONS_PER_IMAGE
        self.shift2box_transform = Shift2BoxTransform(
            weights=cfg.MODEL.FCOS.BBOX_REG_WEIGHTS)

        self.shift_generator = build_shift_generator(cfg, [input_shape[f] for f in self.in_features])

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(
            3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(
            3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)

    def forward(self, images, features, gt_instances=None, top_module=None, mask_cost_func=None, mask_feats=None):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances: Instances

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                    See :meth:`postprocess` for details.
        Returns:
            dict[str: Tensor]:
                mapping from a named loss to a tensor storing the loss. Used during training only.
        """


        features = [features[f] for f in self.in_features]
        self.oym_outputs.num_gen_params = self.num_gen_params
        locations = self.compute_locations(features)
        shifts = self.shift_generator(features)

        box_cls, box_filter, top_feats, bbox_towers, iou_preds = self.oym_head(features, top_module)

        if self.training:

            loss = dict()
            if self.aux_on:
                aux_class = self.oym_outputs.get_aux_ground_truth(
                    shifts, gt_instances, box_cls, top_feats, locations, mask_cost_func=mask_cost_func, mask_feats=mask_feats)

                loss.update(self.aux_losses(aux_class, box_cls))

            gt_classes, instances, iou_result = self.oym_outputs.get_ground_truth(
                shifts, gt_instances, box_cls, iou_preds, box_filter, top_feats, locations, mask_cost_func, mask_feats=mask_feats)

            iou_loss = self.iou_losses(instances)
            iou_loss_negative = self.iou_losses_negative(iou_result)

            loss.update(iou_loss)
            loss.update(iou_loss_negative)

            loss.update(self.oym_outputs.losses(gt_classes, box_cls, box_filter))

            return instances, loss
        else:
            results = self.inference(box_cls, box_filter, iou_preds, shifts,
                                     images, top_feats, locations)
            return results, {}



    def inference(self, box_cls, box_filter, iou_preds, shifts, images, top_feats, locations=False):
        """
        Arguments:
            box_cls, box_delta, box_filter: Same as the output of :meth:`FCOSHead.forward`
            shifts (list[list[Tensor]): a list of #images elements. Each is a
                list of #feature level tensor. The tensor contain shifts of this
                image on the specific feature level.
            images (ImageList): the input images

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        results = []
        append = results.append
        fpn_level = [
            loc.new_ones(len(loc), dtype=torch.long) * level
            for level, loc in enumerate(locations)
        ]
        box_cls = [permute_to_N_HWA_K(x, self.num_classes) for x in box_cls]
        box_filter = [permute_to_N_HWA_K(x, 1) for x in box_filter]
        iou_preds = [permute_to_N_HWA_K(x, 1) for x in iou_preds]
        top_feats = [permute_to_N_HWA_K(x, self.num_gen_params) for x in top_feats]

        image_size = images.image_sizes[0]
        for img_idx, shifts_per_image in enumerate(shifts):

            box_cls_per_image = [
                box_cls_per_level[img_idx] for box_cls_per_level in box_cls
            ]
            box_filter_per_image = [
                box_filter_per_level[img_idx] for box_filter_per_level in box_filter
            ]
            top_feats_per_image = [
                top_feats_per_level[img_idx] for top_feats_per_level in top_feats
            ]
            iou_preds_per_image = [
                iou_preds_per_level[img_idx] for iou_preds_per_level in iou_preds
            ]

            results_per_image = self.inference_single_image(
                box_cls_per_image, box_filter_per_image,
                shifts_per_image, tuple(image_size), top_feats_per_image, iou_preds_per_image,  locations, fpn_level)
            append(results_per_image)
        return results

    def inference_single_image(self, box_cls, box_filter, shifts, image_size, top_feats, iou_preds,  locations, fpn_levels):
        """
        Single-image inference. Return bounding-box detection results by thresholding
        on scores and applying non-maximum suppression (NMS).

        Arguments:
            box_cls (list[Tensor]): list of #feature levels. Each entry contains
                tensor of size (H x W, K)
            box_delta (list[Tensor]): Same shape as 'box_cls' except that K becomes 4.
            box_filter (list[Tensor]): Same shape as 'box_cls' except that K becomes 1.
            shifts (list[Tensor]): list of #feature levels. Each entry contains
                a tensor, which contains all the shifts for that
                image in that feature level.
            image_size (tuple(H, W)): a tuple of the image height and width.

        Returns:
            Same as `inference`, but for only one image.
        """
        scores_all = []
        top_feats_all = []
        locations_all = []
        class_idxs_all = []
        fpn_level_all = []

        # Iterate over every feature level
        for box_cls_i, box_filter_i, shifts_i, top_feat_i, iou_preds_i, locations_i, fpn_level_i in zip(
                box_cls, box_filter, shifts, top_feats, iou_preds, locations, fpn_levels):

            box_cls_i = (box_cls_i.sigmoid_() * box_filter_i.sigmoid_() * iou_preds_i).flatten()

            # Keep top k top scoring indices only.
            num_topk = min(self.topk_candidates, box_cls_i.size(0))
            predicted_prob, topk_idxs = box_cls_i.sort(descending=True)
            predicted_prob = predicted_prob[:num_topk]
            topk_idxs = topk_idxs[:num_topk]
            #TODO
            # 相當於0.0025
            # predicted_prob = torch.sqrt(predicted_prob)

            # filter out the proposals with low confidence score
            if self.prediction_reranking:
                self.score_threshold = self.score_threshold/2

            keep_idxs = predicted_prob > self.score_threshold
            predicted_prob = predicted_prob[keep_idxs]
            topk_idxs = topk_idxs[keep_idxs]
            shift_idxs = topk_idxs // self.num_classes
            classes_idxs = topk_idxs % self.num_classes

            top_feat_i = top_feat_i[shift_idxs]
            locations_i = locations_i[shift_idxs]
            fpn_level_i = fpn_level_i[shift_idxs]

            top_feats_all.append(top_feat_i)
            locations_all.append(locations_i)
            scores_all.append(predicted_prob)
            class_idxs_all.append(classes_idxs)
            fpn_level_all.append(fpn_level_i)

        scores_all, class_idxs_all, top_feats_all, locations_all, fpn_level_all = [
            cat(x) for x in [scores_all, class_idxs_all, top_feats_all, locations_all, fpn_level_all]
        ]

        if self.nms_type is None:
            # strategies above (e.g. topk_candidates and score_threshold) are
            # useless for OYM, just keep them for debug and analysis
            keep = scores_all.argsort(descending=True)
            keep = keep[:self.max_detections_per_image]

        result = Instances(image_size)
        result.scores = scores_all[keep]
        result.pred_classes = class_idxs_all[keep]
        result.top_feat = top_feats_all[keep]
        result.locations = locations_all[keep]
        result.fpn_levels = fpn_level_all[keep]

        return result

    def compute_locations(self, features):
        locations = []
        for level, feature in enumerate(features):
            h, w = feature.size()[-2:]
            locations_per_level = compute_locations(
                h, w, self.fpn_strides[level],
                feature.device
            )
            locations.append(locations_per_level)

        return locations

    def iou_losses(self, instances):
        instances = instances['instances']
        pred_ious = instances.pred_ious.squeeze(1)
        gt_ious = instances.gt_ious
        loss = self.L1_loss(pred_ious, gt_ious)

        return {'iou_L1_loss': loss}

    def iou_losses_negative(self, instances):

        pred_ious = instances.other_predict.squeeze(1)
        gt_ious = instances.other_gt
        loss = self.L1_loss(pred_ious, gt_ious)

        return {'negative_iou_L1_loss': loss}



    def aux_losses(self, gt_classes, pred_class_logits):
        pred_class_logits = cat([
            permute_to_N_HWA_K(x, self.num_classes) for x in pred_class_logits
        ], dim=1).view(-1, self.num_classes)

        gt_classes = gt_classes.flatten()

        valid_idxs = gt_classes >= 0
        foreground_idxs = (gt_classes >= 0) & (gt_classes != self.num_classes)
        num_foreground = foreground_idxs.sum()

        gt_classes_target = torch.zeros_like(pred_class_logits)
        gt_classes_target[foreground_idxs, gt_classes[foreground_idxs]] = 1

        num_foreground = comm.all_reduce(num_foreground) / float(comm.get_world_size())

        # logits loss
        loss_cls_aux = sigmoid_focal_loss_jit(
            pred_class_logits[valid_idxs],
            gt_classes_target[valid_idxs],
            alpha=self.focal_loss_alpha,
            gamma=self.focal_loss_gamma,
            reduction="sum",
        ) / max(1.0, num_foreground)

        return {"loss_cls_aux": loss_cls_aux}



class Uni_fcos_head(nn.Module):
    """
    The head used in FCOS for object classification and box regression.
    It has two subnets for the two tasks, with a common structure but separate parameters.
    """
    def __init__(self, cfg, input_shape: List[ShapeSpec]):
        super().__init__()
        # fmt: off
        in_channels = input_shape[0].channels
        num_classes = cfg.MODEL.FCOS.NUM_CLASSES
        num_convs = cfg.MODEL.FCOS.NUM_CONVS
        prior_prob = cfg.MODEL.FCOS.PRIOR_PROB
        num_shifts = build_shift_generator(cfg, input_shape).num_cell_shifts
        self.fpn_strides = cfg.MODEL.FCOS.FPN_STRIDES
        self.norm_reg_targets = cfg.MODEL.FCOS.NORM_REG_TARGETS

        # fmt: on
        assert len(set(num_shifts)) == 1, "using differenct num_shifts value is not supported"
        num_shifts = num_shifts[0]

        # top
        in_channel = [s.channels for s in input_shape]
        in_channel = in_channel[0]
        self.in_channels_to_top_module = in_channel

        cls_subnet = []
        instance_subnet = []
        for _ in range(num_convs):
            cls_subnet.append(
                nn.Conv2d(in_channels,
                          in_channels,
                          kernel_size=3,
                          stride=1,
                          padding=1))
            cls_subnet.append(nn.GroupNorm(32, in_channels))
            cls_subnet.append(nn.ReLU())
            instance_subnet.append(
                nn.Conv2d(in_channels,
                          in_channels,
                          kernel_size=3,
                          stride=1,
                          padding=1))
            instance_subnet.append(nn.GroupNorm(32, in_channels))
            instance_subnet.append(nn.ReLU())

        self.cls_subnet = nn.Sequential(*cls_subnet)
        self.instance_subnet = nn.Sequential(*instance_subnet)
        self.cls_score = nn.Conv2d(in_channels,
                                   num_shifts * num_classes,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1)
        self.iou_pred = nn.Conv2d(
            in_channels, 1,
            kernel_size=3, stride=1, padding=1)




        self.max3d = MaxFiltering_gn(in_channels,
                                         kernel_size=cfg.MODEL.OYOR.FILTER_KERNEL_SIZE,
                                         tau=cfg.MODEL.OYOR.FILTER_TAU)


        self.filter = nn.Conv2d(in_channels,
                                num_shifts * 1,
                                kernel_size=3,
                                stride=1,
                                padding=1)

        # Initialization
        for modules in [
            self.cls_subnet, self.cls_score, self.instance_subnet,
            self.max3d, self.filter, self.iou_pred
        ]:
            for layer in modules.modules():
                if isinstance(layer, nn.Conv2d):
                    torch.nn.init.normal_(layer.weight, mean=0, std=0.01)
                    torch.nn.init.constant_(layer.bias, 0)
                if isinstance(layer, nn.GroupNorm):
                    torch.nn.init.constant_(layer.weight, 1)
                    torch.nn.init.constant_(layer.bias, 0)

        # Use prior in model initialization to improve stability
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_score.bias, bias_value)

        self.scales = nn.ModuleList(
            [Scale(init_value=1.0) for _ in range(len(self.fpn_strides))])

    def forward(self, features, top_module=None):
        """
        Arguments:
            features (list[Tensor]): FPN feature map tensors in high to low resolution.
                Each tensor in the list correspond to different feature levels.

        Returns:
            logits (list[Tensor]): #lvl tensors, each has shape (N, K, Hi, Wi).
                The tensor predicts the classification probability
                at each spatial position for each of the K object classes.
            bbox_reg (list[Tensor]): #lvl tensors, each has shape (N, 4, Hi, Wi).
                The tensor predicts 4-vector (dl,dt,dr,db) box
                regression values for every shift. These values are the
                relative offset between the shift and the ground truth box.
            filter (list[Tensor]): #lvl tensors, each has shape (N, 1, Hi, Wi).
                The tensor predicts the centerness at each spatial position.
        """
        logits = []
        filter_subnet = []
        top_feats = []
        bbox_towers = []
        iou_preds = []
        for level, feature in enumerate(features):
            cls_subnet = self.cls_subnet(feature)
            instance_subnet = self.instance_subnet(feature)
            logits.append(self.cls_score(cls_subnet))
            iou_preds.append(self.iou_pred(instance_subnet))
            top_feats.append(top_module(instance_subnet))
            filter_subnet.append(instance_subnet)
        filters = [self.filter(x) for x in self.max3d(filter_subnet)]
        return logits, filters, top_feats, bbox_towers, iou_preds

class MaxFiltering_gn(nn.Module):
    def __init__(self, in_channels: int, kernel_size: int = 3, tau: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.norm = nn.GroupNorm(32, in_channels)
        self.nonlinear = nn.ReLU()
        self.max_pool = nn.MaxPool3d(
            kernel_size=(tau + 1, kernel_size, kernel_size),
            padding=(tau // 2, kernel_size // 2, kernel_size // 2),
            stride=1
        )
        self.margin = tau // 2

    def forward(self, inputs):
        features = []
        for l, x in enumerate(inputs):
            features.append(self.conv(x))

        outputs = []
        for l, x in enumerate(features):
            func = lambda f: F.interpolate(f, size=x.shape[2:], mode="bilinear")
            feature_3d = []
            for k in range(max(0, l - self.margin), min(len(features), l + self.margin + 1)):
                feature_3d.append(func(features[k]) if k != l else features[k])
            feature_3d = torch.stack(feature_3d, dim=2)
            max_pool = self.max_pool(feature_3d)[:, :, min(l, self.margin)]
            output = max_pool + inputs[l]
            outputs.append(self.nonlinear(self.norm(output)))
        return outputs