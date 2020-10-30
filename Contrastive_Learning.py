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

import numpy as np
import torch
import torch.optim as optim

import torch.distributed.autograd as dist_autograd
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.optim import DistributedOptimizer
from torch.distributed.rpc import RRef

from time import time

sys.path.append('SimCLR/NVIDIA DALI')
import NVIDIA_DALI_Pipelines as NDP
sys.path.append('SimCLR/ResNet')
import ResNet as rn
sys.path.append('SimCLR/MLP')
import multilayerPerceptron as mlp
sys.path.append('SimCLR')
import SimCLR

def parse():
        parser = argparse.ArgumentParser(prog='Contrastive_Learning',
                                         description='This program executes the Contrastive Learning Algorithm using foveated saccades')
        parser.add_argument('data', metavar='DIR', default='/projects/neurophon', type=str,
                            help='path to the MSCOCO dataset')
        parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                            help='number of data loading workers (default: 4)')
        parser.add_argument('--epochs', default=90, type=int, metavar='N',
                            help='number of total epochs to run')
        parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                            help='manual epoch number (useful on restarts)')
        parser.add_argument('-b', '--batch-size', default=256, type=int,
                            metavar='N', help='mini-batch size per process (default: 256)')
        parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                            metavar='LR', help='Initial learning rate.  Will be scaled by <global batch size>/256: args.lr = args.lr*float(args.batch_size*args.world_size)/256. A warmup schedule will also be applied over the first 5 epochs.')
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
        parser.add_argument('--dali_cpu', action='store_true',
                            help='Runs CPU based version of DALI pipeline.')
        parser.add_argument('--prof', default=-1, type=int,
                            help='Only run 10 iterations for profiling.')
        parser.add_argument('--deterministic', action='store_true')
        parser.add_argument("--local_rank", default=0, type=int)
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
        

        args.gpu = 0
        args.world_size = 1
        
        if args.distributed:
                if torch.cuda.device_count() > 1:
                        args.gpu = args.local_rank

                torch.cuda.set_device(args.gpu)
                torch.distributed.init_process_group(backend='gloo', init_method='env://')
                # torch.distributed.init_process_group(backend='nccl', init_method='env://')
                args.world_size = torch.distributed.get_world_size()
                if args.verbose:
                        print('distributed is True, then rank number {} is mapped in device number {}' .format(args.local_rank, args.gpu))

        args.total_batch_size = args.world_size * args.batch_size


        # Set the device
        device = torch.device('cpu' if args.dali_cpu else 'cuda:' + str(args.gpu))
        if args.verbose:
               print('Using device {} in rank number {}' .format(device, args.local_rank))


        # Set fuction_f
        function_f = rn.ResNet.ResNet18()
        function_f.to(device)
        if args.verbose:
               print('function_f created from rank {}' .format(args.local_rank))
        

        # Set function_g
        function_g = mlp.MLP(512*4*4, 1024, 128)
        function_g.to(device)
        if args.verbose:
               print('function_g created from rank {}' .format(args.local_rank))
        

        # Set SimCLR model
        img_size = (30,30)
        model = SimCLR.SimCLR_Module(args.temperature, function_f, function_g, args.batch_size, img_size, device)
        model.to(device)
        if args.verbose:
               print('SimCLR_Module created from rank {}' .format(args.local_rank))
        
        
        # Optionally resume from a checkpoint
        if args.resume:
             # Use a local scope to avoid dangling references
             def resume():
                if os.path.isfile(args.resume):
                    print("=> loading checkpoint '{}'" .format(args.resume))
                    checkpoint = torch.load(args.resume, map_location = lambda storage, loc: storage.cuda(args.gpu))
                    args.start_epoch = checkpoint['epoch']
                    model.load_state_dict(checkpoint['state_dict'])
                    optimizer.load_state_dict(checkpoint['optimizer'])
                    print("=> loaded checkpoint '{}' (epoch {})"
                                    .format(args.resume, checkpoint['epoch']))
                else:
                    print("=> no checkpoint found at '{}'" .format(args.resume))
        
             resume()


        path = args.data
        os.environ['DALI_EXTRA_PATH']=path
        test_data_root = os.environ['DALI_EXTRA_PATH']
        file_root = os.path.join(test_data_root, 'MSCOCO', 'cocoapi', 'images', 'val2014')
        annotations_file = os.path.join(test_data_root, 'MSCOCO', 'cocoapi', 'annotations', 'instances_val2014.json')

        # This is pipe1, using this we bring image batches from MSCOCO dataset
        # If there is more than one GPU, it will try to map each rank in a different GPU device
        if torch.cuda.device_count() > 1:
                pipe1 = NDP.COCOReader(batch_size=args.batch_size,
                                       num_threads=args.workers,
                                       device_id=args.local_rank,
                                       file_root=file_root,
                                       annotations_file=annotations_file,
                                       shard_id=args.local_rank,
                                       num_shards=args.world_size,
                                       dali_cpu=args.dali_cpu)
        # Otherwise, it will map all ranks on the same GPU device
        else:
                pipe1 = NDP.COCOReader(batch_size=args.batch_size,
                                       num_threads=args.workers,
                                       device_id=0,
                                       file_root=file_root,
                                       annotations_file=annotations_file,
                                       shard_id=args.local_rank,
                                       num_shards=args.world_size,
                                       dali_cpu=args.dali_cpu)
        start = time()
        pipe1.build()
        total_time = time() - start
        meta_data = pipe1.reader_meta()['__COCOReader_1']
        if args.verbose:
               print('pipe1 built by rank number {} in {} seconds' .format(args.local_rank, total_time))
               print('pipe1 dataset information from rank {} is {}' .format(args.local_rank, meta_data))
               print('pipe1 shard size for rank {} is {}' .format(args.local_rank, calculate_shard_size(pipe1)))


        # This is pipe2, which is used to augment the batches brought by pipe1 utilizing foveated saccades
        images = NDP.ImageCollector()
        fixation = NDP.FixationCommand(args.batch_size)
        # If there is more than one GPU, it will try to map each rank in a different GPU device
        if torch.cuda.device_count() > 1:
                pipe2 = NDP.FoveatedRetinalProcessor(batch_size=args.batch_size,
                                                     num_threads=args.workers,
                                                     device_id=args.local_rank,
                                                     fixation_information=fixation,
                                                     images=images,
                                                     dali_cpu=args.dali_cpu)
        # Otherwise, it will map all ranks on the same GPU device
        else:
                pipe2 = NDP.FoveatedRetinalProcessor(batch_size=args.batch_size,
                                                     num_threads=args.workers,
                                                     device_id=0,
                                                     fixation_information=fixation,
                                                     images=images,
                                                     dali_cpu=args.dali_cpu)


        start = time()
        pipe2.build()
        total_time = time() - start
        if args.verbose:
               print('pipe2 built by rank number {} in {} seconds' .format(args.local_rank, total_time))


        # For distributed training, wrap the model with torch.nn.parallel.DistributedDataParallel.
        if args.distributed:
                if args.dali_cpu:
                        model = DDP(model)
                else:
                        model = DDP(model, device_ids=[args.gpu], output_device=args.gpu)

                if args.verbose:
                        print('Since we are in a distributed setting the model is replicated here in rank {}' .format(args.local_rank))


        # # Set optimizer and scale larning rate on global batch size
        # args.lr = args.lr*float(args.batch_size*args.world_size)/256.
        # optimizer = optim.SGD(model.parameters(), args.lr,
                              # momentum=args.momentum,
                              # weight_decay=args.weight_decay)


        # total_time = AverageMeter()
        # for epoch in range(args.start_epoch, args.epochs):
                # # train for one epoch
                # arguments = {'pipe1': pipe1, 'pipe2': pipe2, 'images': images, 'model': model, 'optimizer': optimizer, 'epoch': epoch}
                # avg_train_time = train(arguments)
                # total_time.update(avg_train_time)
                # if args.test:
                        # break

                # pipe1.reset()


































def train(arguments):
        batch_time = AverageMeter()
        losses = AverageMeter()

        number_of_fixations = 10

        # switch to train mode
        arguments['model'].train()
        end = time()

        shard_size = calculate_shard_size(arguments['pipe1'])
        i = 0
        while i*arguments['pipe1'].batch_size < shard_size:
                # bring a new batch
                pipe1_output = arguments['pipe1'].run()
                images_gpu = pipe1_output[0]
                bboxes_cpu = pipe1_output[1]
                labels_cpu = pipe1_output[2]

                # set images for pipe2
                arguments['images'].data = images_gpu

                # set fixation parameters for pipe2
                NDP.fixation_pos_x = torch.rand((args.batch_size,1))
                NDP.fixation_pos_y = torch.rand((args.batch_size,1))
                NDP.fixation_angle = (torch.rand((args.batch_size,1))-0.5)*60

                # make the first saccade
                pipe2_output = NDP.pytorch_wrapper([arguments['pipe2']])
                outputs1 = arguments['model'](pipe2_output[0][5:])

                for j in range(number_of_fixations):
                        # set the second round of fixation parameters for pipe2
                        NDP.fixation_pos_x = torch.rand((args.batch_size,1))
                        NDP.fixation_pos_y = torch.rand((args.batch_size,1))
                        NDP.fixation_angle = (torch.rand((args.batch_size,1))-0.5)*60
                        
                        # make the second saccade
                        pipe2_output = NDP.pytorch_wrapper([arguments['pipe2']])
                        outputs2 = arguments['model'](pipe2_output[0][5:])

                        # Compute Huber loss
                        loss = arguments['model'].compute_loss(outputs1.detach(), outputs2)

                        # compute gradient and do SGD step
                        arguments['optimizer'].zero_grad()
                        loss.backward()
                        arguments['optimizer'].step()
                        outputs1=outputs2


                if i%args.print_freq == 0:
                        # Every print_freq iterations, check the loss and speed.
                        # For best performance, it doesn't make sense to print these metrics every
                        # iteration, since they incur an allreduce and some host<->device syncs.

                        # Average loss across processes for logging
                        if args.distributed:
                                reduced_loss = reduce_tensor(loss.data)
                        else:
                                reduced_loss = loss.data

                        # to_python_float incurs a host<->device sync
                        losses.update(to_python_float(reduced_loss), input.size(0))

                        torch.cuda.synchronize()
                        batch_time.update((time.time() - end)/args.print_freq)
                        end = time.time()

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















class AverageMeter(object):
    """Computes and stores the average current value"""

    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count











def calculate_shard_size(pipe):
        meta_data = pipe.reader_meta()['__COCOReader_1']

        if meta_data['pad_last_batch'] == 1:
                shards_beg = np.floor(meta_data['shard_id'] * meta_data['epoch_size_padded'] / meta_data['number_of_shards']).astype(np.int)
                shards_end = np.floor((meta_data['shard_id'] + 1) * meta_data['epoch_size_padded'] / meta_data['number_of_shards']).astype(np.int)
        else:
                shards_beg = np.floor(meta_data['shard_id'] * meta_data['epoch_size'] / meta_data['number_of_shards']).astype(np.int)
                shards_end = np.floor((meta_data['shard_id'] + 1) * meta_data['epoch_size'] / meta_data['number_of_shards']).astype(np.int)

        return shards_end - shards_beg





def reduce_tensor(tensor):
        rt = tensor.clone()
        dist.all_reduce(rt, op=dist.reduce_op.SUM)
        rt /= args.world_size
        return rt





if __name__ == '__main__':
        main()

