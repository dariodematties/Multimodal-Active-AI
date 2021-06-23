#TODO PLEASE CHECK USAGE RIGHTS BEFORE PUBLISHING SINCE THIS CODE HAS BEEN COPY-PASTED FROM
# https://github.com/facebookresearch/detr/blob/master/models/backbone.py



"""
Backbone modules.
"""
import os
import sys

from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

sys.path.append('SimCLR/MLP')
import multilayerPerceptron as mlp
sys.path.append('SimCLR/ResNet')
import resnet as rn
sys.path.append('SimCLR')
import SimCLR
import Model_Util
sys.path.append('detr_CLA/util')
from misc import NestedTensor, is_main_process

from position_encoding import build_position_encoding


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.
    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_layers: bool):
        super().__init__()
        self.return_interm_layers=return_interm_layers
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {'layer4': "0"}
        # self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.body = backbone
        self.num_channels = num_channels

    def forward(self, tensor_list: NestedTensor):
        bs, c, s = tensor_list.tensors.shape
        AUX = []
        for saccade in range(s):
            saccade_tensor=tensor_list.tensors[:,:,saccade].view(bs,12,30,30).float()
            AUX.append(self.body(saccade_tensor))
            # AUX.append(self.body(saccade_tensor).view(bs,self.num_channels*2*2))

        xs = {}
        layer1=[]
        layer2=[]
        layer3=[]
        layer4=[]
        for saccade in AUX:
            if self.return_interm_layers:
                layer1.append(saccade['0'].view(bs,self.num_channels*2*2))
                layer2.append(saccade['1'].view(bs,self.num_channels*2*2))
                layer3.append(saccade['2'].view(bs,self.num_channels*2*2))
                layer4.append(saccade['3'].view(bs,self.num_channels*2*2))
            else:
                # layer1.append(saccade['0'].view(bs,self.num_channels*2*2))
                layer1.append(saccade.view(bs,self.num_channels*4*4))

        if self.return_interm_layers:
            layer1 = torch.stack(layer1,2)
            layer2 = torch.stack(layer2,2)
            layer3 = torch.stack(layer3,2)
            layer4 = torch.stack(layer4,2)
            xs['0'] = layer1
            xs['1'] = layer2
            xs['2'] = layer3
            xs['3'] = layer4
        else:
            layer1 = torch.stack(layer1,2)
            xs['0'] = layer1

        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-1:]).to(torch.bool)[0]
            # mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, device: str, name: str, path: str, gpu: int,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool,
                 batch_size: int,
                 distributed: bool,
                 dali_cpu):
        # backbone = getattr(torchvision.models, name)(
            # replace_stride_with_dilation=[False, False, dilation],
            # pretrained=is_main_process(), norm_layer=FrozenBatchNorm2d)


        # Set fuction_f
        if name == 'ResNet18':
            function_f = rn.resnet18(norm_layer=FrozenBatchNorm2d, replace_stride_with_dilation=[False, False, dilation])
            # function_f = rn.resnet18(norm_layer=nn.SyncBatchNorm, replace_stride_with_dilation=[False, False, dilation])
            # function_f = rn.resnet18(norm_layer=nn.SyncBatchNorm)
        elif name == 'ResNet34':
            function_f = rn.resnet34(norm_layer=FrozenBatchNorm2d, replace_stride_with_dilation=[False, False, dilation])
            # function_f = rn.resnet34(norm_layer=nn.SyncBatchNorm, replace_stride_with_dilation=[False, False, dilation])
            # function_f = rn.resnet34(norm_layer=nn.SyncBatchNorm)
        elif name == 'ResNet50':
            function_f = rn.resnet50(norm_layer=FrozenBatchNorm2d, replace_stride_with_dilation=[False, False, dilation])
            # function_f = rn.resnet50(norm_layer=nn.SyncBatchNorm, replace_stride_with_dilation=[False, False, dilation])
            # function_f = rn.resnet50(norm_layer=nn.SyncBatchNorm)
        elif name == 'ResNet101':
            function_f = rn.resnet101(norm_layer=FrozenBatchNorm2d, replace_stride_with_dilation=[False, False, dilation])
            # function_f = rn.resnet101(norm_layer=nn.SyncBatchNorm, replace_stride_with_dilation=[False, False, dilation])
            # function_f = rn.resnet101(norm_layer=nn.SyncBatchNorm)
        elif name == 'ResNet152':
            function_f = rn.resnet152(norm_layer=FrozenBatchNorm2d, replace_stride_with_dilation=[False, False, dilation])
            # function_f = rn.resnet152(norm_layer=nn.SyncBatchNorm, replace_stride_with_dilation=[False, False, dilation])
            # function_f = rn.resnet152(norm_layer=nn.SyncBatchNorm)
        else:
            raise Exception("error: Unrecognized {} architecture" .format(name))

        function_f = function_f.to(device)
        
        # Set function_g
        if name == 'ResNet18' or name == 'ResNet34':
            function_g = mlp.MLP(512*4*4, 1024, 128)
        elif name == 'ResNet50' or name == 'ResNet101' or name == 'ResNet152':
            function_g = mlp.MLP(2048*4*4, 1024, 128)
        else:
            raise Exception("error: Unrecognized {} architecture" .format(name))

        function_g = function_g.to(device)

        # Set SimCLR backbone
        img_size = (30,30)
        backbone = SimCLR.SimCLR_Module(function_f, function_g, batch_size, img_size, device)
        backbone = backbone.to(device)

        # For distributed training, wrap the backbone with torch.nn.parallel.DistributedDataParallel.
        if distributed:
            if dali_cpu:
                backbone = DDP(backbone)
            else:
                backbone = DDP(backbone, device_ids=[gpu], output_device=gpu)

            backbone = backbone.module

        backbone = self.load_model(device, backbone, path, gpu)
        backbone = backbone.f
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)


    def load_model(self, device, backbone, backbone_path, gpu):
       if os.path.isfile(backbone_path):
           print("=> loading pretrained backbone '{}'" .format(backbone_path))
           checkpoint = torch.load(backbone_path, map_location = lambda storage, loc: storage.cuda(gpu))
           backbone.load_state_dict(checkpoint['state_dict'])
           print("=> loaded pretrained backbone '{}'"
                           .format(backbone_path))
           backbone.g = Model_Util.Identity().to(device)
           return backbone
       else:
           print("=> no pretrained backbone found at '{}'" .format(backbone_path))



class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor, tensor_list_saccades: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            # pos.append(self[1](x).to(x.tensors.dtype))
            aux = tensor_list_saccades.to(x.tensors.device)
            pos.append(self[1](aux).to(x.tensors.dtype))

        return out, pos


def build_backbone(args, device):
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    return_interm_layers = False #args.masks
    backbone = Backbone(device,
                        args.backbone,
                        args.backbone_path,
                        args.gpu, train_backbone,
                        return_interm_layers,
                        args.dilation,
                        args.batch_size,
                        args.distributed,
                        args.dali_cpu)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model
