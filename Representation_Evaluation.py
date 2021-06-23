# This file contains a program to evaluate the learned representations
# of our model pre-trained by means of self-supervised contrastive learning.

# To evaluate the learned representations, we follow the widely used
# evaluation protocol, where a classifier is trained on top of
# the frozen base network, and test accuracy is used as a proxy for representation
# quality.


import argparse
import sys
import os
import math
import socket

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import torch.distributed as dist
import torch.distributed.autograd as dist_autograd
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.optim import DistributedOptimizer
from torch.distributed.rpc import RRef

from time import time

sys.path.append('SimCLR/NVIDIA DALI')
import NVIDIA_DALI_Pipelines as NDP
sys.path.append('SimCLR/ResNet')
import resnet as rn
# import ResNet as rn
sys.path.append('SimCLR/MLP')
import multilayerPerceptron as mlp
sys.path.append('SimCLR/MLR')
import multivariateLogisticRegression as mlr
sys.path.append('SimCLR')
import SimCLR
import Objective
import Model_Util
import Utilities

# Set global variables for rank, local_rank, world size
try:
    from mpi4py import MPI

    with_ddp=True
    local_rank = os.environ['OMPI_COMM_WORLD_LOCAL_RANK']
    size = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()


    # Pytorch will look for these:
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(size)
    # os.environ['CUDA_VISIBLE_DEVICES'] = str(local_rank)

    # It will want the master address too, which we'll broadcast:
    if rank == 0:
        master_addr = socket.gethostname()
    else:
        master_addr = None

    master_addr = MPI.COMM_WORLD.bcast(master_addr, root=0)
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(2345)


except Exception as e:
    with_ddp=False
    local_rank = 0
    size = 1
    rank = 0
    print("MPI initialization failed!")
    print(e)



def parse():
        model_names = ['ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152']

        datasets = ['imagenet']

        optimizers = ['sgd', 'adam', 'lars']

        classifiers = ['logistic_regression', 'multilayer_perceptron', 'support_vector_machine']

        parser = argparse.ArgumentParser(prog='Representation_Evaluation',
                                         description='This program evaluates the learned representaions of the pretrained model\n' +
                                                     'training a classifier on top of the frozen base network')
        parser.add_argument('model', metavar='MODEL_DIR',
                            help='path to the pre-trained model')
        parser.add_argument('data', metavar='DATASET_DIR',
                            help='path to IMAGENET dataset')
        parser.add_argument('--arch', '-a', metavar='ARCH', default='ResNet18',
                            choices=model_names,
                            help='model architecture: ' +
                            ' | '.join(model_names) +
                            ' (default: ResNet18)')
        parser.add_argument('--classifier', metavar='CLASSIFIER',
                            default='logistic_regression', type=str,
                            choices=classifiers,
                            help='Classifier to enable evaluation\n' +
                                 'Choices are: ' +
                                 ' | '.join(classifiers) +
                                 ' (default: logistic_regression)')
        parser.add_argument('--dataset', default='imagenet', type=str, metavar='DATASET',
                            choices=datasets,
                            help='the dataset we will use\n' +
                                 'Choices are: ' +
                                 ' | '.join(datasets) +
                                 ' (default: imagenet)')
        parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                            help='number of data loading workers (default: 4)')
        parser.add_argument('--epochs', default=90, type=int, metavar='N',
                            help='number of total epochs to run')
        parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                            help='manual epoch number (useful on restarts)')
        parser.add_argument('-b', '--batch-size', default=256, type=int,
                            metavar='N', help='mini-batch size per process (default: 256)')
        parser.add_argument('-f', '--num-fixations', default=2, type=int,
                            metavar='F', help='Number of fixations per image (default: 2)')
        parser.add_argument('--lr', '--learning-rate', default=0.0000001, type=float,
                            metavar='LR',
                            help='Initial learning rate.  Will be scaled by <global batch size>/256:\n' +
                            'args.lr = args.lr*float(args.batch_size*args.world_size)/256.\n' +
                            'A warmup schedule will also be applied over the first 5 epochs.')
        parser.add_argument('--lrs', '--learning-rate-scaling', default='linear', type=str,
                            metavar='LRS', help='Function to scale the learning rate value (default: \'linear\').')
        parser.add_argument('--warmup-epochs', default=10, type=int, metavar='W',
                            help='Number of warmup epochs (default: 10)')
        parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                            help='momentum')
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
        parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                            help='evaluate model on validation set')
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
        global best_prec1, args
        best_prec1 = 0
        args = parse()


        if not len(args.data):
                raise Exception("error: No dataset provided")

        if not len(args.model):
                raise Exception("error: No model provided")

        args.distributed = False
        if 'WORLD_SIZE' in os.environ:
                args.distributed = int(os.environ['WORLD_SIZE']) > 1
        

        args.gpu = 0
        args.world_size = 1
        
        if args.distributed:

                # torch.distributed.init_process_group(backend='mpi', init_method='env://')
                # torch.distributed.init_process_group(backend='gloo', init_method='env://')
                torch.distributed.init_process_group(backend='nccl', init_method='env://')
                args.world_size = torch.distributed.get_world_size()
                args.global_rank = torch.distributed.get_rank()

                args.local_rank = int(local_rank)
                args.gpu = args.local_rank
                torch.cuda.set_device(args.gpu)

                if args.verbose:
                        print('distributed is True, then world size is {}, global rank is {} and local rank number {} is mapped in device number {}'
                              .format(args.world_size, args.global_rank, args.local_rank, args.gpu))

        args.total_batch_size = args.world_size * args.batch_size


        # Set the device
        device = torch.device('cpu' if args.dali_cpu else 'cuda:' + str(args.gpu))
        if args.verbose:
               print('Using device {} in global rank number {}, local rank number {}' .format(device, args.global_rank, args.local_rank))









        path = args.data
        os.environ['DALI_EXTRA_PATH']=path
        test_data_root = os.environ['DALI_EXTRA_PATH']
        # This is pipe1, using this we bring image batches for training the classifier
        if args.dataset == 'imagenet':
                file_root = os.path.join(test_data_root, 'ImageNet', 'ILSVRC', 'Data', 'CLS-LOC', 'train')

                pipe1 = NDP.ImagenetReader(batch_size=args.batch_size,
                                           num_threads=args.workers,
                                           device_id=args.gpu,
                                           file_root=file_root,
                                           shard_id=args.gpu,
                                           num_shards=args.world_size,
                                           random_shuffle=True,
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
        labels = NDP.LabelCollector()
        fixation = NDP.FixationCommand(args.batch_size)
        pipe2 = NDP.LabeledFoveatedRetinalProcessor(batch_size=args.batch_size,
                                                    num_threads=args.workers,
                                                    device_id=args.gpu,
                                                    fixation_information=fixation,
                                                    images=images,
                                                    labels=labels,
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
        if args.dataset == 'imagenet':
                aux='/lus/theta-fs0/software/datascience/'
                file_root = os.path.join(aux, 'ImageNet', 'ILSVRC', 'Data', 'CLS-LOC', 'val')
                # file_root = os.path.join(test_data_root, 'ImageNet', 'ILSVRC', 'Data', 'CLS-LOC', 'val')

                pipe3 = NDP.ImagenetReader(batch_size=args.batch_size,
                                           num_threads=args.workers,
                                           device_id=args.gpu,
                                           file_root=file_root,
                                           shard_id=args.gpu,
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


















        # Set fuction_f
        if args.arch == 'ResNet18':
            function_f = rn.resnet18(norm_layer=nn.SyncBatchNorm)
            # function_f = rn.ResNet.ResNet18()
        elif args.arch == 'ResNet34':
            function_f = rn.resnet34(norm_layer=nn.SyncBatchNorm)
            # function_f = rn.ResNet.ResNet34()
        elif args.arch == 'ResNet50':
            function_f = rn.resnet50(norm_layer=nn.SyncBatchNorm)
            # function_f = rn.ResNet.ResNet50()
        elif args.arch == 'ResNet101':
            function_f = rn.resnet101(norm_layer=nn.SyncBatchNorm)
            # function_f = rn.ResNet.ResNet101()
        elif args.arch == 'ResNet152':
            function_f = rn.resnet152(norm_layer=nn.SyncBatchNorm)
            # function_f = rn.ResNet.ResNet152()
        else:
            raise Exception("error: Unrecognized {} architecture" .format(args.arch))

        function_f = function_f.to(device)
        if args.verbose:
               print('function_f created from global rank number {}, local rank {}' .format(args.global_rank, args.local_rank))
        

        # Set function_g
        if args.arch == 'ResNet18' or args.arch == 'ResNet34':
            # function_g = mlp.MLP(512, 1024, 128)
            function_g = mlp.MLP(512*4*4, 1024, 128)
        elif args.arch == 'ResNet50' or args.arch == 'ResNet101' or args.arch == 'ResNet152':
            # function_g = mlp.MLP(2048, 4096, 128)
            function_g = mlp.MLP(2048*4*4, 1024, 128)
        else:
            raise Exception("error: Unrecognized {} architecture" .format(args.arch))

        function_g = function_g.to(device)
        if args.verbose:
               print('function_g created from global rank number {}, local rank {}' .format(args.global_rank, args.local_rank))
        

        # Set SimCLR model
        img_size = (30,30)
        model = SimCLR.SimCLR_Module(function_f, function_g, args.batch_size, img_size, device)
        model = model.to(device)
        if args.verbose:
               print('SimCLR_Module created from global rank number {}, local rank {}' .format(args.global_rank, args.local_rank))


        # For distributed training, wrap the model with torch.nn.parallel.DistributedDataParallel.
        if args.distributed:
                if args.dali_cpu:
                        model = DDP(model)
                else:
                        model = DDP(model, device_ids=[args.gpu], output_device=args.gpu)

                if args.verbose:
                        print('Since we are in a distributed setting the model is replicated here in global rank number {}, local rank {}'
                                        .format(args.global_rank, args.local_rank))

        # Load pretrined model to get its representations
        if args.model:
             # Use a local scope to avoid dangling references
             def load_model():
                if os.path.isfile(args.model):
                    print("=> loading pretrained model '{}'" .format(args.model))
                    checkpoint = torch.load(args.model, map_location = lambda storage, loc: storage.cuda(args.gpu))
                    model.load_state_dict(checkpoint['state_dict'])
                    print("=> loaded pretrained model '{}'"
                                    .format(args.model))
                    model.module.g = Model_Util.Identity().to(device)
                    return model
                else:
                    print("=> no pretrained model found at '{}'" .format(args.model))
        
             model = load_model()
        else:
             raise Exception("error: No pretrained model provided")




        if args.classifier == 'logistic_regression':
            if args.arch == 'ResNet18' or args.arch == 'ResNet34':
                # classifier = mlr.LogisticRegression(512*args.num_fixations, 1000)
                classifier = mlr.LogisticRegression(512*4*4*args.num_fixations, 1000)
            elif args.arch == 'ResNet50' or args.arch == 'ResNet101' or args.arch == 'ResNet152':
                # classifier = mlr.LogisticRegression(2048*args.num_fixations, 1000)
                classifier = mlr.LogisticRegression(2048*4*4*args.num_fixations, 1000)
            else:
                raise Exception("error: Unrecognized {} architecture" .format(args.arch))
        else:
            raise Exception("error: Unknown classifier {}" .format(args.classifier))

        classifier = classifier.to(device)
        if args.verbose:
               print('A {} classifier created from global rank number {}, local rank {}' .format(args.classifier, args.global_rank, args.local_rank))



        # For distributed training, wrap the classifier with torch.nn.parallel.DistributedDataParallel.
        if args.distributed:
                if args.dali_cpu:
                        classifier = DDP(classifier)
                else:
                        classifier = DDP(classifier, device_ids=[args.gpu], output_device=args.gpu)

                if args.verbose:
                        print('Since we are in a distributed setting the classifier is replicated here in global rank number {}, local rank {}'
                                        .format(args.global_rank, args.local_rank))





        criterion = nn.CrossEntropyLoss()
        # Set optimizer
        optimizer = Model_Util.get_optimizer(classifier, args)
        if args.global_rank==0 and args.verbose:
                print('Optimizer used for this run is {}'.format(args.optimizer))

        # Optionally resume from a checkpoint
        if args.resume:
             # Use a local scope to avoid dangling references
             def resume():
                if os.path.isfile(args.resume):
                    print("=> loading classifier checkpoint '{}'" .format(args.resume))
                    checkpoint = torch.load(args.resume, map_location = lambda storage, loc: storage.cuda(args.gpu))
                    start_epoch = checkpoint['epoch']
                    best_prec1 = checkpoint['best_prec1']
                    classifier.load_state_dict(checkpoint['state_dict'])
                    optimizer.load_state_dict(checkpoint['optimizer'])
                    print("=> loaded classifier checkpoint '{}' (epoch {})"
                                    .format(args.resume, checkpoint['epoch']))
                    print("Classifier best precision saved was {}" .format(best_prec1))
                    return start_epoch, best_prec1, classifier, optimizer
                else:
                    print("=> no checkpoint found at '{}'" .format(args.resume))
        
             args.start_epoch, best_prec1, classifier, optimizer = resume()


        total_time = Utilities.AverageMeter()
        for epoch in range(args.start_epoch, args.epochs):
                arguments = {'pipe1': pipe1,
                             'pipe2': pipe2,
                             'pipe3': pipe3,
                             'images': images,
                             'labels': labels,
                             'model': model,
                             'classifier': classifier,
                             'criterion': criterion,
                             'optimizer': optimizer,
                             'warmup_epochs': args.warmup_epochs,
                             'train_epochs': args.epochs,
                             'arch': args.arch,
                             'num_examples': NDP.compute_shard_size(pipe1, pipe1_reader_name),
                             'pipe1_reader_name': pipe1_reader_name,
                             'pipe3_reader_name': pipe3_reader_name,
                             'epoch': epoch,
                             'num_fixations': args.num_fixations,
                             'batch_size': args.batch_size,
                             'world_size': args.world_size,
                             'base_learning_rate': args.lr,
                             'learning_rate_scaling': args.lrs,
                             'is_testing': args.test,
                             'device': device}

                # train for one epoch
                avg_train_time = train_classifier(arguments)
                total_time.update(avg_train_time)
                if args.test:
                        break

                # evaluate on validation set
                [prec1, prec5] = val_classifier(arguments)

                # remember the best prec@1 and save checkpoint
                if args.global_rank == 0:
                        print('From validation we have prec1 is {} while best_prec1 is {}'.format(prec1, best_prec1))
                        is_best = prec1 > best_prec1
                        best_prec1 = max(prec1, best_prec1)
                        Model_Util.save_checkpoint({
                                'epoch': epoch + 1,
                                'state_dict': classifier.state_dict(),
                                'best_prec1': best_prec1,
                                'optimizer': optimizer.state_dict(),
                        }, is_best, filename='classifier_checkpoint.pth.tar', best_filename='classifier_best.pth.tar')

                        print('##Top-1 ACC {0}\n'
                              '##Top-5 ACC {1}\n'
                              '##Best Top-1 ACC saved {2}\n'
                              '##Perf {3}'.format(
                              prec1,
                              prec5,
                              best_prec1,
                              args.total_batch_size / total_time.avg))

                pipe1.reset()
                pipe3.reset()




















































def train_classifier(arguments):
        batch_time = Utilities.AverageMeter()
        losses = Utilities.AverageMeter()

        # switch model to evaluate mode
        arguments['model'].eval()
        arguments['classifier'].train()
        end = time()

        shard_size = NDP.compute_shard_size(arguments['pipe1'], arguments['pipe1_reader_name'])
        i = 0
        while i*arguments['pipe1'].batch_size < shard_size:
                # bring a new batch
                pipe1_output = arguments['pipe1'].run()
                images_gpu = pipe1_output[0]
                labels_cpu = pipe1_output[1]

                train_loader_len = int(math.ceil(shard_size / arguments['pipe1'].batch_size))

                # set images and labels for pipe2
                arguments['images'].data = images_gpu
                arguments['labels'].data = labels_cpu

                inputs = []
                with torch.no_grad():
                    # set fixation angle for pipe2
                    # NDP.fixation_angle = (torch.rand((args.batch_size,1))-0.5)*10
                    NDP.fixation_angle = torch.repeat_interleave(torch.Tensor([0]), args.batch_size).view(-1,1)
                    for j in range(args.num_fixations):
                        # set fixation position for pipe2
                        NDP.fixation_pos_x = torch.rand((args.batch_size,1))
                        NDP.fixation_pos_y = torch.rand((args.batch_size,1))

                        # make the fixation
                        pipe2_output = NDP.pytorch_wrapper([arguments['pipe2']])
                        fixation = arguments['model'](pipe2_output[0][:4])
                        # fixation = arguments['model'](pipe2_output[0][:5])
                        if arguments['arch'] == 'ResNet18' or arguments['arch'] == 'ResNet34':
                            # fixation = fixation.view(arguments['batch_size'], 512)
                            fixation = fixation.view(arguments['batch_size'], 512*4*4)
                        elif arguments['arch'] == 'ResNet50' or arguments['arch'] == 'ResNet101' or arguments['arch'] == 'ResNet152':
                            # fixation = fixation.view(arguments['batch_size'], 2048)
                            fixation = fixation.view(arguments['batch_size'], 2048*4*4)
                        else:
                            raise Exception("error: Unrecognized {} architecture" .format(args.arch))

                        inputs.append(fixation)

                    inputs = torch.stack(inputs, dim=2)
                    if arguments['arch'] == 'ResNet18' or arguments['arch'] == 'ResNet34':
                        # inputs = inputs.view(arguments['batch_size'], 512*args.num_fixations)
                        inputs = inputs.view(arguments['batch_size'], 512*4*4*args.num_fixations)
                    elif arguments['arch'] == 'ResNet50' or arguments['arch'] == 'ResNet101' or arguments['arch'] == 'ResNet152':
                        # inputs = inputs.view(arguments['batch_size'], 2048*args.num_fixations)
                        inputs = inputs.view(arguments['batch_size'], 2048*4*4*args.num_fixations)
                    else:
                        raise Exception("error: Unrecognized {} architecture" .format(args.arch))


                # bring the labels of this batch of images
                labels = pipe2_output[0][4]
                # labels = pipe2_output[0][5]
                labels = torch.transpose(labels,0,1).squeeze(0)

                # Forward pass
                outputs = arguments['classifier'](inputs)

                # Compute Loss
                loss = arguments['criterion'](outputs, labels.type(torch.long))

                # Adjust learning rate
                Model_Util.learning_rate_schedule(arguments)

                # Backward pass
                arguments['optimizer'].zero_grad()
                loss.backward()
                arguments['optimizer'].step()

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

                        if args.global_rank == 0:
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
















def val_classifier(arguments):
        batch_time = Utilities.AverageMeter()
        top1_prec = Utilities.AverageMeter()
        top5_prec = Utilities.AverageMeter()

        # switch model to evaluate mode
        arguments['model'].eval()
        arguments['classifier'].eval()
        end = time()

        shard_size = NDP.compute_shard_size(arguments['pipe3'], arguments['pipe1_reader_name'])
        i = 0
        while i*arguments['pipe3'].batch_size < shard_size:
                # bring a new batch
                pipe1_output = arguments['pipe3'].run()
                images_gpu = pipe1_output[0]
                labels_cpu = pipe1_output[1]

                val_loader_len = int(math.ceil(shard_size / arguments['pipe3'].batch_size))

                # set images and labels for pipe2
                arguments['images'].data = images_gpu
                arguments['labels'].data = labels_cpu

                inputs = []
                with torch.no_grad():
                    # set fixation angle for pipe2
                    # NDP.fixation_angle = (torch.rand((args.batch_size,1))-0.5)*10
                    NDP.fixation_angle = torch.repeat_interleave(torch.Tensor([0]), args.batch_size).view(-1,1)
                    for j in range(args.num_fixations):
                        # set fixation position for pipe2
                        NDP.fixation_pos_x = torch.rand((args.batch_size,1))
                        NDP.fixation_pos_y = torch.rand((args.batch_size,1))

                        # make the fixation
                        pipe2_output = NDP.pytorch_wrapper([arguments['pipe2']])
                        fixation = arguments['model'](pipe2_output[0][:4])
                        # fixation = arguments['model'](pipe2_output[0][:5])
                        if arguments['arch'] == 'ResNet18' or arguments['arch'] == 'ResNet34':
                            # fixation = fixation.view(arguments['batch_size'], 512)
                            fixation = fixation.view(arguments['batch_size'], 512*4*4)
                        elif arguments['arch'] == 'ResNet50' or arguments['arch'] == 'ResNet101' or arguments['arch'] == 'ResNet152':
                            # fixation = fixation.view(arguments['batch_size'], 2048)
                            fixation = fixation.view(arguments['batch_size'], 2048*4*4)
                        else:
                            raise Exception("error: Unrecognized {} architecture" .format(args.arch))

                        inputs.append(fixation)

                    inputs = torch.stack(inputs, dim=2)
                    if arguments['arch'] == 'ResNet18' or arguments['arch'] == 'ResNet34':
                        # inputs = inputs.view(arguments['batch_size'], 512*args.num_fixations)
                        inputs = inputs.view(arguments['batch_size'], 512*4*4*args.num_fixations)
                    elif arguments['arch'] == 'ResNet50' or arguments['arch'] == 'ResNet101' or arguments['arch'] == 'ResNet152':
                        # inputs = inputs.view(arguments['batch_size'], 2048*args.num_fixations)
                        inputs = inputs.view(arguments['batch_size'], 2048*4*4*args.num_fixations)
                    else:
                        raise Exception("error: Unrecognized {} architecture" .format(args.arch))


                # bring the labels of this batch of images
                labels = pipe2_output[0][4]
                # labels = pipe2_output[0][5]
                labels = torch.transpose(labels,0,1).squeeze(0)

                # Forward pass
                logits = arguments['classifier'](inputs)

                # compute top 1 and top 5 classification accuracy
                top_1_accuracy = Model_Util.top_k_accuracy(logits, labels, 1)
                top_5_accuracy = Model_Util.top_k_accuracy(logits, labels, 5)


                if args.distributed:
                        reduced_top1 = Utilities.reduce_tensor(top_1_accuracy.data, args.world_size)
                        reduced_top5 = Utilities.reduce_tensor(top_5_accuracy.data, args.world_size)
                else:
                        reduced_top1 = top_1_accuracy.data
                        reduced_top5 = top_5_accuracy.data


                top1_prec.update(Utilities.to_python_float(reduced_top1), pipe2_output[0][0].size(0))
                top5_prec.update(Utilities.to_python_float(reduced_top5), pipe2_output[0][0].size(0))

                # measure elapsed time
                batch_time.update((time() - end)/args.print_freq)
                end = time()

                if args.global_rank == 0 and i % args.print_freq == 0:
                        print('Test: [{0}/{1}]\t'
                              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                              'Speed {2:.3f} ({3:.3f})\t'
                              'Top-1 Acc {top1.val:.4f} ({top1.avg:.4f})\t'
                              'Top-5 Acc {top5.val:.4f} ({top5.avg:.4f})'.format(
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

