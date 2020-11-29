# This file runs the Contrastive Learning algorithm from the paper entitled
# A Simple Framework for Contrastive Learning of Visual Representations
# https://arxiv.org/abs/2002.05709


# The final idea behind this program is to implement Contrastive Learning
# through an interactive loop with two DALI pipelines pipe1 and pipe2
# First pipe1 brings a batch of images
# Then we have the following closed loop

#    -------------------
#   |                   |
#   |                   |
#   |              N    |
#   |         -----<----
#   |        |          ^
#   V        v          |
# pipe1 -> pipe2 -> NN -

# where pipe2 augments the batch received from pipe1
# After N runs of the closed loop pipe1 is called
# again and brings the following batch in the dataset

# Everything has to run in distributed GPUs on the system

# I used the guidances provided in https://github.com/NVIDIA/DALI/blob/1e9196702d991d3342ad7a5a7d57c2893abad832/docs/examples/use_cases/pytorch/resnet50/main.py
# from NVIDIA DALI library support

import argparse
import sys
import os
import math

import random
import numpy as np
import torch

import torch.distributed as dist
import torch.distributed.autograd as dist_autograd
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.optim import DistributedOptimizer
from torch.distributed.rpc import RRef
from pytorch_lightning.metrics import Accuracy

from time import time

sys.path.append('SimCLR/NVIDIA DALI')
import NVIDIA_DALI_Pipelines as NDP
sys.path.append('SimCLR/ResNet')
import ResNet as rn
sys.path.append('SimCLR/MLP')
import multilayerPerceptron as mlp
sys.path.append('SimCLR')
import SimCLR
import Objective
import Model_Util
import Utilities

def parse():

        datasets = ['mscoco', 'imagenet']

        optimizers = ['sgd', 'adam', 'lars']

        parser = argparse.ArgumentParser(prog='Contrastive_Learning',
                                         description='This program executes the Contrastive Learning Algorithm using foveated saccades')
        parser.add_argument('data', metavar='DIR', type=str,
                            help='path to MSCOCO or IMAGENET dataset')
        parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                            help='number of data loading workers (default: 4)')
        parser.add_argument('--epochs', default=90, type=int, metavar='N',
                            help='number of total epochs to run')
        parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                            help='manual epoch number (useful on restarts)')
        parser.add_argument('-b', '--batch-size', default=256, type=int,
                            metavar='N', help='mini-batch size per process (default: 256)')
        parser.add_argument('-f', '--num-fixations', default=10, type=int,
                            metavar='F', help='Number of fixations per image (default: 10)')
        parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                            metavar='LR', help='Initial learning rate.  By default, it will be scaled by <global batch size>/256: args.lr = args.lr*float(args.batch_size*args.world_size)/256. A warmup schedule will also be applied over the first --warmup_epochs epochs.')
        parser.add_argument('--lrs', '--learning-rate-scaling', default='linear', type=str,
                            metavar='LRS', help='Function to scale the learning rate value (default: \'linear\').')
        parser.add_argument('--warmup_epochs', default=10, type=int, metavar='W',
                            help='Number of warmup epochs (default: 10)')
        parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                            help='momentum')
        parser.add_argument('--temperature', default=0.05, type=float, metavar='T',
                            help='SimCLR temperature')
        parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                            metavar='W', help='weight decay (default: 1e-4)')
        parser.add_argument('--print-freq', '-p', default=10, type=int,
                            metavar='N', help='print frequency (default: 10)')
        parser.add_argument('--resume', default='', type=str, metavar='PATH',
                            help='path to latest checkpoint (default: none)')
        parser.add_argument('--optimizer', default='adam', type=str, metavar='OPTIM',
                            choices=optimizers,
                            help='optimizer for training the network\n' +
                                 'Choices are: ' +
                                 ' | '.join(optimizers) +
                                 ' (default: adam)')
        parser.add_argument('--dataset', default='mscoco', type=str, metavar='DATASET',
                            choices=datasets,
                            help='the dataset we will use\n' +
                                 'Choices are: ' +
                                 ' | '.join(datasets) +
                                 ' (default: mscoco)')
        parser.add_argument('--color-augmentation', default=0.5, type=float,
                            metavar='COLOR_AUG_PROBABILITY', help='The probability of applying color augmentation to the images (default: 0.5).')
        parser.add_argument('--dali_cpu', action='store_true',
                            help='Runs CPU based version of DALI pipeline.')
        parser.add_argument("--local_rank", default=0, type=int)
        parser.add_argument("--global_rank", default=0, type=int)
        parser.add_argument('-t', '--test', action='store_true',
                            help='Launch test mode with preset arguments')
        parser.add_argument('-v', '--verbose', action='store_true',
                            help='provides additional details as to what the program is doing')
        args = parser.parse_args()
        return args


def main():
        global args
        args = parse()


        if not len(args.data):
                raise Exception("error: No data set provided")

        args.distributed = False
        if 'WORLD_SIZE' in os.environ:
                args.distributed = int(os.environ['WORLD_SIZE']) > 1
        

        best_prec1 = 0
        args.gpu = 0
        args.world_size = 1
        
        if args.distributed:
                args.gpu = args.local_rank

                torch.cuda.set_device(args.gpu)
                # torch.distributed.init_process_group(backend='mpi', init_method='env://')
                torch.distributed.init_process_group(backend='gloo', init_method='env://')
                # torch.distributed.init_process_group(backend='nccl', init_method='env://')
                args.world_size = torch.distributed.get_world_size()
                args.global_rank = torch.distributed.get_rank()
                if args.verbose:
                        print('distributed is True, then world size is {}, global rank is {} and local rank number {} is mapped in device number {}'
                              .format(args.world_size, args.global_rank, args.local_rank, args.gpu))


        args.total_batch_size = args.world_size * args.batch_size


        # Set the device
        device = torch.device('cpu' if args.dali_cpu else 'cuda:' + str(args.gpu))
        if args.verbose:
               print('Using device {} in global rank number {}, local rank number {}' .format(device, args.global_rank, args.local_rank))


        # Set fuction_f
        function_f = rn.ResNet.ResNet50()
        function_f = function_f.to(device)
        if args.verbose:
               print('function_f created from global rank number {}, local rank {}' .format(args.global_rank, args.local_rank))
        

        # Set function_g
        function_g = mlp.MLP(512*4*4, 1024, 128)
        function_g = function_g.to(device)
        if args.verbose:
               print('function_g created from global rank number {}, local rank {}' .format(args.global_rank, args.local_rank))
        

        # Set SimCLR model
        img_size = (30,30)
        model = SimCLR.SimCLR_Module(function_f, function_g, args.batch_size, img_size, device)
        model = model.to(device)
        if args.verbose:
               print('SimCLR_Module created from global rank number {}, local rank {}' .format(args.global_rank, args.local_rank))
        
        
        path = args.data
        os.environ['DALI_EXTRA_PATH']=path
        test_data_root = os.environ['DALI_EXTRA_PATH']
        # This is pipe1, using this we bring image batches for training
        if args.dataset == 'mscoco':
                file_root = os.path.join(test_data_root, 'MSCOCO', 'cocoapi', 'images', 'train2014')
                annotations_file = os.path.join(test_data_root, 'MSCOCO', 'cocoapi', 'annotations', 'instances_train2014.json')

                pipe1 = NDP.COCOReader(batch_size=args.batch_size,
                                       num_threads=args.workers,
                                       device_id=args.local_rank,
                                       file_root=file_root,
                                       annotations_file=annotations_file,
                                       shard_id=args.local_rank,
                                       num_shards=args.world_size,
                                       dali_cpu=args.dali_cpu)

                pipe1_reader_name = 'COCOReader'

        elif args.dataset == 'imagenet':
                file_root = os.path.join(test_data_root, 'ImageNet', 'ILSVRC', 'Data', 'CLS-LOC', 'train')

                pipe1 = NDP.ImagenetReader(batch_size=args.batch_size,
                                           num_threads=args.workers,
                                           device_id=args.local_rank,
                                           file_root=file_root,
                                           shard_id=args.local_rank,
                                           num_shards=args.world_size,
                                           dali_cpu=args.dali_cpu)

                pipe1_reader_name = 'ImagesReader'

        else:
                raise Exception("error: I do not accept {} as a viable dataset".format(args.dataset))

        start = time()
        pipe1.build()
        total_time = time() - start
        meta_data = pipe1.reader_meta()[pipe1_reader_name]
        if args.verbose:
               print('pipe1 built by global rank number {}, local rank number {} in {} seconds' .format(args.global_rank, args.local_rank, total_time))
               print('pipe1 dataset information from global rank number {}, local rank {} is {}' .format(args.global_rank, args.local_rank, meta_data))
               print('pipe1 shard size for global rank number {}, local rank {} is {}' .format(args.global_rank, args.local_rank, NDP.compute_shard_size(pipe1, pipe1_reader_name)))





        # This is pipe2, which is used to augment the batches brought by pipe1 or pipe3 utilizing foveated saccades
        images = NDP.ImageCollector()
        fixation = NDP.FixationCommand(args.batch_size)
        color = NDP.ColorCommand(args.batch_size)
        pipe2 = NDP.UnlabeledFoveatedRetinalProcessor(batch_size=args.batch_size,
                                                      num_threads=args.workers,
                                                      device_id=args.local_rank,
                                                      fixation_information=fixation,
                                                      color_information=color,
                                                      images=images,
                                                      dali_cpu=args.dali_cpu)


        start = time()
        pipe2.build()
        total_time = time() - start
        if args.verbose:
               print('pipe2 built by global rank number {}, local rank number {} in {} seconds' .format(args.global_rank, args.local_rank, total_time))









        path = args.data
        os.environ['DALI_EXTRA_PATH']=path
        test_data_root = os.environ['DALI_EXTRA_PATH']
        # This is pipe3, using this we bring image batches for validation
        if args.dataset == 'mscoco':
                file_root = os.path.join(test_data_root, 'MSCOCO', 'cocoapi', 'images', 'val2014')
                annotations_file = os.path.join(test_data_root, 'MSCOCO', 'cocoapi', 'annotations', 'instances_val2014.json')

                pipe3 = NDP.COCOReader(batch_size=args.batch_size,
                                       num_threads=args.workers,
                                       device_id=args.local_rank,
                                       file_root=file_root,
                                       annotations_file=annotations_file,
                                       shard_id=args.local_rank,
                                       num_shards=args.world_size,
                                       dali_cpu=args.dali_cpu)

                pipe3_reader_name = 'COCOReader'

        elif args.dataset == 'imagenet':
                file_root = os.path.join(test_data_root, 'ImageNet', 'ILSVRC', 'Data', 'CLS-LOC', 'val')

                pipe3 = NDP.ImagenetReader(batch_size=args.batch_size,
                                           num_threads=args.workers,
                                           device_id=args.local_rank,
                                           file_root=file_root,
                                           shard_id=args.local_rank,
                                           num_shards=args.world_size,
                                           dali_cpu=args.dali_cpu)

                pipe3_reader_name = 'ImagesReader'

        else:
                raise Exception("error: I do not accept {} as a viable dataset".format(args.dataset))

        start = time()
        pipe3.build()
        total_time = time() - start
        meta_data = pipe3.reader_meta()[pipe3_reader_name]
        if args.verbose:
               print('pipe3 built by global rank number {}, local rank number {} in {} seconds' .format(args.global_rank, args.local_rank, total_time))
               print('pipe3 dataset information from global rank number {}, local rank {} is {}' .format(args.global_rank, args.local_rank, meta_data))
               print('pipe3 shard size for global rank number {}, local rank {} is {}' .format(args.global_rank, args.local_rank, NDP.compute_shard_size(pipe3, pipe3_reader_name)))







        # For distributed training, wrap the model with torch.nn.parallel.DistributedDataParallel.
        if args.distributed:
                if args.dali_cpu:
                        model = DDP(model)
                else:
                        model = DDP(model, device_ids=[args.gpu], output_device=args.gpu)

                if args.verbose:
                        print('Since we are in a distributed setting the model is replicated here in global rank number {}, local rank {}'
                                        .format(args.global_rank, args.local_rank))


        # Set optimizer
        optimizer = Model_Util.get_optimizer(model, args)
        if args.global_rank==0 and args.verbose:
                print('Optimizer used for this run is {}'.format(args.optimizer))

        # Optionally resume from a checkpoint
        if args.resume:
             # Use a local scope to avoid dangling references
             def resume():
                if os.path.isfile(args.resume):
                    print("=> loading checkpoint '{}'" .format(args.resume))
                    checkpoint = torch.load(args.resume, map_location = lambda storage, loc: storage.cuda(args.gpu))
                    start_epoch = checkpoint['epoch']
                    best_prec1 = checkpoint['best_prec1']
                    model.load_state_dict(checkpoint['state_dict'])
                    optimizer.load_state_dict(checkpoint['optimizer'])
                    print("=> loaded checkpoint '{}' (epoch {})"
                                    .format(args.resume, checkpoint['epoch']))
                    print("Model best precision saved was {}" .format(best_prec1))
                    return start_epoch, best_prec1, model, optimizer
                else:
                    print("=> no checkpoint found at '{}'" .format(args.resume))
        
             args.start_epoch, best_prec1, model, optimizer = resume()


        total_time = Utilities.AverageMeter()
        for epoch in range(args.start_epoch, args.epochs):
                arguments = {'pipe1': pipe1,
                             'pipe2': pipe2,
                             'pipe3': pipe3,
                             'images': images,
                             'model': model,
                             'optimizer': optimizer,
                             'warmup_epochs': args.warmup_epochs,
                             'train_epochs': args.epochs,
                             'num_examples': NDP.compute_shard_size(pipe1, pipe1_reader_name),
                             'pipe1_reader_name': pipe1_reader_name,
                             'pipe3_reader_name': pipe3_reader_name,
                             'epoch': epoch,
                             'num_fixations': args.num_fixations,
                             'batch_size': args.batch_size,
                             'world_size': args.world_size,
                             'base_learning_rate': args.lr,
                             'learning_rate_scaling': args.lrs,
                             'temperature': args.temperature,
                             'is_testing': args.test,
                             'device': device}

                # train for one epoch
                avg_train_time = train(arguments)
                total_time.update(avg_train_time)
                if args.test:
                        break

                # evaluate on validation set
                [prec1, prec5] = validate(arguments)

                # remember the best prec@1 and save checkpoint
                if args.global_rank == 0:
                        print('From validation we have prec1 is {} while best_prec1 is {}'.format(prec1, best_prec1))
                        is_best = prec1 > best_prec1
                        best_prec1 = max(prec1, best_prec1)
                        Model_Util.save_checkpoint({
                                'epoch': epoch + 1,
                                'state_dict': model.state_dict(),
                                'best_prec1': best_prec1,
                                'optimizer': optimizer.state_dict(),
                        }, is_best)

                        print('##Contrastive Top-1 {0}\n'
                              '##Contrastive Top-5 {1}\n'
                              '##Best Contrastive Top-1 saved {2}\n'
                              '##Perf {3}'.format(
                              prec1,
                              prec5,
                              best_prec1,
                              args.total_batch_size / total_time.avg))

                pipe1.reset()
                pipe3.reset()


































def train(arguments):
        batch_time = Utilities.AverageMeter()
        losses = Utilities.AverageMeter()

        # switch to train mode
        arguments['model'].train()
        end = time()

        shard_size = NDP.compute_shard_size(arguments['pipe1'], arguments['pipe1_reader_name'])
        i = 0
        while i*arguments['pipe1'].batch_size < shard_size:
                # bring a new batch
                pipe1_output = arguments['pipe1'].run()
                images_gpu = pipe1_output[0]

                # bboxes_cpu = pipe1_output[1]
                # labels_cpu = pipe1_output[2]

                train_loader_len = int(math.ceil(shard_size / arguments['pipe1'].batch_size))

                # set images for pipe2
                arguments['images'].data = images_gpu

                # set fixation parameters for pipe2
                NDP.fixation_pos_x = torch.rand((args.batch_size,1))
                NDP.fixation_pos_y = torch.rand((args.batch_size,1))
                NDP.fixation_angle = (torch.rand((args.batch_size,1))-0.5)*260

                # set color parameters for pipe2
                if random.uniform(0, 1) < args.color_augmentation:
                        NDP.brightness = torch.rand((args.batch_size,1))*2
                        NDP.contrast = torch.rand((args.batch_size,1))*2
                        NDP.hue = torch.rand((args.batch_size,1))*360
                        NDP.saturation = torch.rand((args.batch_size,1))
                else:
                        NDP.brightness = torch.repeat_interleave(torch.Tensor([1.0]), args.batch_size).view(-1,1)
                        NDP.contrast = torch.repeat_interleave(torch.Tensor([1.0]), args.batch_size).view(-1,1)
                        NDP.hue = torch.repeat_interleave(torch.Tensor([0.0]), args.batch_size).view(-1,1)
                        NDP.saturation = torch.repeat_interleave(torch.Tensor([1.0]), args.batch_size).view(-1,1)

                # make the first saccade
                pipe2_output = NDP.pytorch_wrapper([arguments['pipe2']])
                outputs1 = arguments['model'](pipe2_output[0])

                for j in range(args.num_fixations):
                        # set the second round of fixation parameters for pipe2
                        NDP.fixation_pos_x = torch.rand((args.batch_size,1))
                        NDP.fixation_pos_y = torch.rand((args.batch_size,1))
                        NDP.fixation_angle = (torch.rand((args.batch_size,1))-0.5)*260
                        
                        # set color parameters for pipe2
                        if random.uniform(0, 1) < args.color_augmentation:
                                NDP.brightness = torch.rand((args.batch_size,1))*2
                                NDP.contrast = torch.rand((args.batch_size,1))*2
                                NDP.hue = torch.rand((args.batch_size,1))*360
                                NDP.saturation = torch.rand((args.batch_size,1))
                        else:
                                NDP.brightness = torch.repeat_interleave(torch.Tensor([1.0]), args.batch_size).view(-1,1)
                                NDP.contrast = torch.repeat_interleave(torch.Tensor([1.0]), args.batch_size).view(-1,1)
                                NDP.hue = torch.repeat_interleave(torch.Tensor([0.0]), args.batch_size).view(-1,1)
                                NDP.saturation = torch.repeat_interleave(torch.Tensor([1.0]), args.batch_size).view(-1,1)

                        # make the second saccade
                        pipe2_output = NDP.pytorch_wrapper([arguments['pipe2']])
                        outputs2 = arguments['model'](pipe2_output[0])

                        # Compute Huber loss
                        loss, logits, labels = Objective.contrastive_loss(hidden1=outputs1.data,
                                                                          hidden2=outputs2,
                                                                          temperature=arguments['temperature'],
                                                                          local_rank=args.local_rank,
                                                                          world_size=args.world_size,
                                                                          device=arguments['device'])


                        # Adjust learning rate
                        Model_Util.learning_rate_schedule(arguments)

                        # compute gradient and do SGD step
                        arguments['optimizer'].zero_grad()
                        loss.backward()
                        arguments['optimizer'].step()
                        outputs1=outputs2


                if arguments['is_testing']:
                        if i > 10:
                                break

                if i%args.print_freq == 0:
                        # Every print_freq iterations, check the loss and speed.
                        # For best performance, it doesn't make sense to print these metrics every
                        # iteration, since they incur an allreduce and some host<->device syncs.

                        # Average loss across processes for logging
                        if args.distributed:
                                reduced_loss = Utilities.reduce_tensor(loss.data, args.world_size)
                        else:
                                reduced_loss = loss.data

                        # to_python_float incurs a host<->device sync
                        losses.update(Utilities.to_python_float(reduced_loss), pipe2_output[0][0].size(0))

                        torch.cuda.synchronize()
                        batch_time.update((time() - end)/args.print_freq)
                        end = time()

                        if args.local_rank == 0:
                                print('Epoch: [{0}][{1}/{2}]\t'
                                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                                      'Speed {3:.3f} ({4:.3f})\t'
                                      'Loss {loss.val:.10f} ({loss.avg:.4f})'.format(
                                      arguments['epoch'], i, train_loader_len,
                                      args.world_size*args.batch_size/batch_time.val,
                                      args.world_size*args.batch_size/batch_time.avg,
                                      batch_time=batch_time,
                                      loss=losses))

                i += 1

        return batch_time.avg










def validate(arguments):
        batch_time = Utilities.AverageMeter()
        top1_prec = Utilities.AverageMeter()
        top5_prec = Utilities.AverageMeter()

        # switch to evaluate mode
        arguments['model'].eval()

        end = time()

        accuracy = Accuracy()

        shard_size = NDP.compute_shard_size(arguments['pipe3'], arguments['pipe3_reader_name'])
        i = 0
        while i*arguments['pipe3'].batch_size < shard_size:
                # bring a new batch
                pipe3_output = arguments['pipe3'].run()
                images_gpu = pipe3_output[0]

                # bboxes_cpu = pipe3_output[1]
                # labels_cpu = pipe3_output[2]

                val_loader_len = int(math.ceil(shard_size / arguments['pipe3'].batch_size))

                with torch.no_grad():
                        # set images for pipe2
                        arguments['images'].data = images_gpu

                        # set fixation parameters for pipe2
                        NDP.fixation_pos_x = torch.rand((args.batch_size,1))
                        NDP.fixation_pos_y = torch.rand((args.batch_size,1))
                        NDP.fixation_angle = (torch.rand((args.batch_size,1))-0.5)*60

                        # set color parameters for pipe2
                        if random.uniform(0, 1) < args.color_augmentation:
                                NDP.brightness = torch.rand((args.batch_size,1))*2
                                NDP.contrast = torch.rand((args.batch_size,1))*2
                                NDP.hue = torch.rand((args.batch_size,1))*360
                                NDP.saturation = torch.rand((args.batch_size,1))
                        else:
                                NDP.brightness = torch.repeat_interleave(torch.Tensor([1.0]), args.batch_size).view(-1,1)
                                NDP.contrast = torch.repeat_interleave(torch.Tensor([1.0]), args.batch_size).view(-1,1)
                                NDP.hue = torch.repeat_interleave(torch.Tensor([0.0]), args.batch_size).view(-1,1)
                                NDP.saturation = torch.repeat_interleave(torch.Tensor([1.0]), args.batch_size).view(-1,1)

                        # make the first saccade
                        pipe2_output = NDP.pytorch_wrapper([arguments['pipe2']])
                        outputs1 = arguments['model'](pipe2_output[0])

                        # set the second round of fixation parameters for pipe2
                        NDP.fixation_pos_x = torch.rand((args.batch_size,1))
                        NDP.fixation_pos_y = torch.rand((args.batch_size,1))
                        NDP.fixation_angle = (torch.rand((args.batch_size,1))-0.5)*60
                        
                        # set the second round of color parameters for pipe2
                        if random.uniform(0, 1) < args.color_augmentation:
                                NDP.brightness = torch.rand((args.batch_size,1))*2
                                NDP.contrast = torch.rand((args.batch_size,1))*2
                                NDP.hue = torch.rand((args.batch_size,1))*360
                                NDP.saturation = torch.rand((args.batch_size,1))
                        else:
                                NDP.brightness = torch.repeat_interleave(torch.Tensor([1.0]), args.batch_size).view(-1,1)
                                NDP.contrast = torch.repeat_interleave(torch.Tensor([1.0]), args.batch_size).view(-1,1)
                                NDP.hue = torch.repeat_interleave(torch.Tensor([0.0]), args.batch_size).view(-1,1)
                                NDP.saturation = torch.repeat_interleave(torch.Tensor([1.0]), args.batch_size).view(-1,1)

                        # make the second saccade
                        pipe2_output = NDP.pytorch_wrapper([arguments['pipe2']])
                        outputs2 = arguments['model'](pipe2_output[0])

                        # Compute Huber loss
                        loss, logits, labels = Objective.contrastive_loss(hidden1=outputs1.data,
                                                                          hidden2=outputs2,
                                                                          temperature=arguments['temperature'],
                                                                          local_rank=args.local_rank,
                                                                          world_size=args.world_size,
                                                                          device=arguments['device'])

                        # contrastive_top_1_accuracy = accuracy(torch.argmax(logits, dim=1).to('cpu'), torch.argmax(labels, dim=1).to('cpu'))
                        contrastive_top_1_accuracy = Model_Util.top_k_accuracy(logits, labels, 1)
                        contrastive_top_5_accuracy = Model_Util.top_k_accuracy(logits, labels, 5)


                if args.distributed:
                        reduced_top1 = Utilities.reduce_tensor(contrastive_top_1_accuracy.data, args.world_size)
                        reduced_top5 = Utilities.reduce_tensor(contrastive_top_5_accuracy.data, args.world_size)
                else:
                        reduced_top1 = contrastive_top_1_accuracy.data
                        reduced_top5 = contrastive_top_5_accuracy.data


                top1_prec.update(Utilities.to_python_float(reduced_top1), pipe2_output[0][0].size(0))
                top5_prec.update(Utilities.to_python_float(reduced_top5), pipe2_output[0][0].size(0))

                # measure elapsed time
                batch_time.update((time() - end)/args.print_freq)
                end = time()

                if args.local_rank == 0 and i % args.print_freq == 0:
                        print('Test: [{0}/{1}]\t'
                              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                              'Speed {2:.3f} ({3:.3f})\t'
                              'Cont. Top-1 Acc {top1.val:.4f} ({top1.avg:.4f})\t'
                              'Cont. Top-5 Acc {top5.val:.4f} ({top5.avg:.4f})'.format(
                              i, val_loader_len,
                              args.world_size*args.batch_size/batch_time.val,
                              args.world_size*args.batch_size/batch_time.avg,
                              batch_time=batch_time,
                              top1=top1_prec,
                              top5=top5_prec))
                
                i += 1

        return [top1_prec.avg, top5_prec.avg]






































if __name__ == '__main__':
        main()

