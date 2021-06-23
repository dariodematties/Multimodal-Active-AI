#TODO PLEASE CHECK USAGE RIGHTS BEFORE PUBLISHING SINCE THIS CODE HAS BEEN COPY-PASTED FROM
# https://github.com/facebookresearch/detr/blob/master/models/detr.py



"""
DETR model and criterion classes.
"""
import sys

import torch
import torch.nn.functional as F
from torch import nn

sys.path.append('detr_CLA/util')
from misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size,
                       is_dist_avail_and_initialized)

from backbone import build_backbone
from transformer import build_transformer


class DETR(nn.Module):
    """ This is the DETR module which is adapted from the original implementation to perform image classification """
    def __init__(self, backbone, transformer, num_classes, num_queries):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of class queries.
                         This is the number of votes we want to be emitted by DETR in order to classify the image
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.input_proj = nn.Conv1d(backbone.num_channels*4*4, hidden_dim, kernel_size=1)
        # self.input_proj = nn.Conv1d(backbone.num_channels*2*2, hidden_dim, kernel_size=1)
        self.backbone = backbone

    def forward(self, samples: NestedTensor, saccades: NestedTensor):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x C x S] (C and S stand for channels and saccades respectively)
               - samples.mask: a binary mask of shape [batch_size x S], containing 1 on padded saccades
            And another NestedTensor, which consists of:
               - saccades.tensor: batched saccades, of shape [batch_size x 2 x S] (2 is for the 2 coordinates of each saccade in the image, S is for the saccade)
               - saccades.mask: a binary mask of shape [batch_size x S], containing 1 on padded saccades
            It returns a dict with the following elements:
               - "pred_logits": the classification logits for all queries.
                                Shape= [batch_size x num_queries x num_classes]
        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)

        if isinstance(saccades, (list, torch.Tensor)):
            saccades = nested_tensor_from_tensor_list(saccades)

        features, pos = self.backbone(samples, saccades)

        src, mask = features[-1].decompose()
        assert mask is not None
        hs = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1])[0]

        outputs_class = self.class_embed(hs)
        out = {'pred_logits': outputs_class[-1]}
        return out


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    """
    def __init__(self, num_queries, num_classes, losses):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories
        """
        super().__init__()
        self.num_queries = num_queries
        self.num_classes = num_classes
        self.losses = losses

    def loss_labels(self, outputs, targets, indices, num_preds, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_predictions]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o.long()

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes)
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def get_loss(self, loss, outputs, targets, indices, num_preds, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_preds, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        #indices = self.matcher(outputs_without_aux, targets)
        batch_size=len(targets)
        idx1=[i for i in range(self.num_queries)]
        idx2=[i for i in range(self.num_queries)]
        indices=[(torch.as_tensor(idx1, dtype=torch.int64),torch.as_tensor(idx2, dtype=torch.int64)) for i in range(batch_size)]

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_preds = sum(len(t["labels"]) for t in targets)
        num_preds = torch.as_tensor([num_preds], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_preds)
        num_preds = torch.clamp(num_preds / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_preds))

        return losses


def build(args, device):
    # the `num_classes` naming here is somewhat misleading.
    # it indeed corresponds to `max_obj_id`, where max_obj_id
    # is the maximum id for a class in your dataset. For example,
    # COCO has a max_obj_id of 90, so we pass `num_classes` to be 90.
    # As another example, for a dataset that has a single class with id 1,
    # you should pass `num_classes` to be 1.
    num_classes = 1000 if args.dataset == 'imagenet' else 90
    device = torch.device(device)

    backbone = build_backbone(args, device)

    transformer = build_transformer(args)

    model = DETR(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
    )
    # TODO this is a hack

    losses = ['labels']
    num_queries=args.num_queries
    criterion = SetCriterion(num_queries, num_classes, losses=losses)
    criterion.to(device)

    return model, criterion
