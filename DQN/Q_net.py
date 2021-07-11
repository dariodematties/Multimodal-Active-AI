# Q-network
# This model will learn the dynamics behind the transformer architecture whose main aim
# is to achieve good performance in its assigned task

import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append('SimCLR/ResNet')
import resnet as rn
# import ResNet as rn
sys.path.append('SimCLR/MLP')
import multilayerPerceptron as mlp

class DQN(nn.Module):
    def __init__(self, f, g_x, g_y, batch_size, img_size, device):
        super(DQN, self).__init__()
        self.f = f
        self.g_x = g_x
        self.g_y = g_y
        self.batch_size = batch_size
        self.img_size = img_size
        self.device = device




    def forward(self, inputs):
        outputs = inputs.reshape(self.batch_size,-1,self.img_size[0],self.img_size[1]).float()
        # print('from the pipe {}' .format(outputs.shape))
        outputs = self.f(outputs)
        # print('from resnet {}' .format(outputs.shape))
        outputs_x = self.g_x(outputs)
        # print('from regressor x {}' .format(outputs_x.shape))
        outputs_y = self.g_y(outputs)
        # print('from regressor y {}' .format(outputs_y.shape))

        return outputs_x, outputs_y




def build_dqn(args, device):
    # Set fuction_f
    if args.dqn == 'ResNet18':
        function_f = rn.resnet18(norm_layer=nn.SyncBatchNorm)
        # function_f = rn.ResNet.ResNet18()
    elif args.dqn == 'ResNet34':
        function_f = rn.resnet34(norm_layer=nn.SyncBatchNorm)
        # function_f = rn.ResNet.ResNet34()
    elif args.dqn == 'ResNet50':
        function_f = rn.resnet50(norm_layer=nn.SyncBatchNorm)
        # function_f = rn.ResNet.ResNet50()
    elif args.dqn == 'ResNet101':
        function_f = rn.resnet101(norm_layer=nn.SyncBatchNorm)
        # function_f = rn.ResNet.ResNet101()
    elif args.dqn == 'ResNet152':
        function_f = rn.resnet152(norm_layer=nn.SyncBatchNorm)
        # function_f = rn.ResNet.ResNet152()
    else:
        raise Exception("error: Unrecognized {} architecture" .format(args.dqn))

    function_f = function_f.to(device)
    if args.verbose:
           print('function_f of DQN created from global rank number {}, local rank {}' .format(args.global_rank, args.local_rank))
    

    # Set function_g_x
    if args.dqn == 'ResNet18' or args.dqn == 'ResNet34':
        # function_g_x = mlp.MLP(512, 1024, args.num_of_actions)
        function_g_x = mlp.MLP(512*4*4, 1024, args.num_of_actions)
    elif args.dqn == 'ResNet50' or args.dqn == 'ResNet101' or args.dqn == 'ResNet152':
        # function_g_x = mlp.MLP(2048, 4096, args.num_of_actions)
        function_g_x = mlp.MLP(2048*4*4, 1024, args.num_of_actions)
    else:
        raise Exception("error: Unrecognized {} architecture" .format(args.dqn))

    function_g_x = function_g_x.to(device)
    if args.verbose:
           print('function_g_x of DQN created from global rank number {}, local rank {}' .format(args.global_rank, args.local_rank))
    

    # Set function_g_y
    if args.dqn == 'ResNet18' or args.dqn == 'ResNet34':
        # function_g_y = mlp.MLP(512, 1024, args.num_of_actions)
        function_g_y = mlp.MLP(512*4*4, 1024, args.num_of_actions)
    elif args.dqn == 'ResNet50' or args.dqn == 'ResNet101' or args.dqn == 'ResNet152':
        # function_g_y = mlp.MLP(2048, 4096, args.num_of_actions)
        function_g_y = mlp.MLP(2048*4*4, 1024, args.num_of_actions)
    else:
        raise Exception("error: Unrecognized {} architecture" .format(args.dqn))

    function_g_y = function_g_y.to(device)
    if args.verbose:
           print('function_g_y of DQN created from global rank number {}, local rank {}' .format(args.global_rank, args.local_rank))
    
    # Set model
    img_size = (30,30)
    model = DQN(function_f, function_g_x, function_g_y, args.batch_size, img_size, device)
    model = model.to(device)

    return model



