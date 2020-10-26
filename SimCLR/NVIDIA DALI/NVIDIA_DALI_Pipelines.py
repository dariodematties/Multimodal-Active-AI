from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types
import os.path

import numpy as np
from nvidia.dali.plugin.pytorch import feed_ndarray, to_torch_type
import torch
from nvidia.dali.backend import TensorGPU, TensorListGPU

global fixation_pos_x
global fixation_pos_y
global fixation_angle

# path = '/projects/neurophon'
# os.environ['DALI_EXTRA_PATH']=path
# test_data_root = os.environ['DALI_EXTRA_PATH']
# file_root = os.path.join(test_data_root, 'MSCOCO', 'cocoapi', 'images', 'val2014')
# annotations_file = os.path.join(test_data_root, 'MSCOCO', 'cocoapi', 'annotations', 'instances_val2014.json')


class COCOReader(Pipeline):
    def __init__(self, batch_size, num_threads, device_id,
                 file_root, annotations_file,
                 shard_id, num_shards, dali_cpu=False):

        super(COCOReader, self).__init__(batch_size,
                                         num_threads,
                                         device_id,
                                         seed=15 + device_id,
                                         exec_pipelined=False,
                                         exec_async=False,
                                         prefetch_queue_depth=1)

        self.input = ops.COCOReader(file_root = file_root,
                                    annotations_file = annotations_file,
                                    shard_id = shard_id,
                                    num_shards = num_shards,
                                    ratio=True,
                                    ltrb=True,
                                    random_shuffle=False)

        #let user decide which pipeline works him bets for RN version he runs
        dali_device = 'cpu' if dali_cpu else 'gpu'
        decoder_device = 'cpu' if dali_cpu else 'mixed'
        
        self.decode = ops.ImageDecoder(device = decoder_device, output_type = types.RGB)
        
        self.flip   = ops.Flip(device=dali_device)
        self.bbflip = ops.BbFlip(device="cpu", ltrb=True)
        
        self.coin = ops.CoinFlip(probability=0.5)


    def define_graph(self):
        rng = self.coin()
        inputs, bboxes, labels = self.input()
        images = self.decode(inputs)
        
        images = self.flip(images, horizontal=rng)
        bboxes = self.bbflip(bboxes, horizontal=rng)
        
        return (images, bboxes, labels)








class ImageCollector(object):
    def __init__(self):
        pass
    def __iter__(self):
        return self
    def __next__(self):
        return self.data
    next = __next__




     






class FixationCommand(object):
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def _get_vectors(self):
        if 'fixation_pos_x' in globals() and 'fixation_pos_y' in globals() and 'fixation_angle' in globals():
            self.vector1 = fixation_pos_x
            self.vector2 = fixation_pos_y
            self.vector3 = fixation_angle
        else:
            print('Initialating fixation\n')
            fixation_pos_x = torch.rand((self.batch_size,1))
            fixation_pos_y = torch.rand((self.batch_size,1))
            fixation_angle = (torch.rand((self.batch_size,1))-0.5)*60

            self.vector1 = fixation_pos_x
            self.vector2 = fixation_pos_y
            self.vector3 = fixation_angle


    def __iter__(self):
        self._get_vectors()
        assert len(self.vector1) == self.batch_size
        assert len(self.vector2) == self.batch_size
        assert len(self.vector3) == self.batch_size
        self.i = 0
        self.n = len(self.vector1)
        return self

    def __next__(self):
        batch1 = []
        batch2 = []
        batch3 = []
        self._get_vectors()
        for _ in range(self.batch_size):
            element1 = self.vector1[self.i]
            batch1.append(element1)
            element2 = self.vector2[self.i]
            batch2.append(element2)
            element3 = self.vector3[self.i]
            batch3.append(element3)
            self.i = (self.i + 1) % self.n
        return batch1, batch2, batch3




class FoveatedRetinalProcessor(Pipeline):
    def __init__(self, batch_size, num_threads, device_id,
                 fixation_information, images,
                 dali_cpu=False):

        super(FoveatedRetinalProcessor, self).__init__(batch_size,
                                                       num_threads,
                                                       device_id,
                                                       seed=15 + device_id,
                                                       exec_pipelined=False,
                                                       exec_async=False,
                                                       prefetch_queue_depth=1)

        #let user decide which pipeline works him bets for RN version he runs
        dali_device = 'cpu' if dali_cpu else 'gpu'

        self.resize_zero = ops.Resize(device = dali_device, resize_x = 640, resize_y = 640)
        
        self.rotate = ops.Rotate(device = dali_device)
        
        self.resize_one  = ops.Resize(device = dali_device, resize_x = 30, resize_y = 30)

        self.crop_zero  = ops.Crop(device = dali_device, crop_h = 640, crop_w = 640)
        self.crop_one   = ops.Crop(device = dali_device, crop_h = 400, crop_w = 400)
        self.crop_two   = ops.Crop(device = dali_device, crop_h = 240, crop_w = 240)
        self.crop_three = ops.Crop(device = dali_device, crop_h = 100, crop_w = 100)
        self.crop_four  = ops.Crop(device = dali_device, crop_h = 30, crop_w = 30)
        
        self.img_batch = ops.ExternalSource(device=dali_device, source = images)
        self.fixation_source = ops.ExternalSource(source = fixation_information, num_outputs = 3)

    def define_graph(self):
        images = self.img_batch()

        crop_pos_x, crop_pos_y, angle = self.fixation_source()
        
        images   = self.rotate(self.resize_zero(images), angle=angle)

        cropped0 = self.crop_zero(images)
        cropped1 = self.crop_one(cropped0, crop_pos_x=crop_pos_x, crop_pos_y=crop_pos_y)
        cropped2 = self.crop_two(cropped0, crop_pos_x=crop_pos_x, crop_pos_y=crop_pos_y)
        cropped3 = self.crop_three(cropped0, crop_pos_x=crop_pos_x, crop_pos_y=crop_pos_y)
        cropped4 = self.crop_four(cropped0, crop_pos_x=crop_pos_x, crop_pos_y=crop_pos_y)
        
        sized0   = self.resize_one(cropped0)
        sized1   = self.resize_one(cropped1)
        sized2   = self.resize_one(cropped2)
        sized3   = self.resize_one(cropped3)
        sized4   = self.resize_one(cropped4)
        
        return (cropped0, cropped1, cropped2, cropped3, cropped4, sized0, sized1, sized2, sized3, sized4)



def pytorch_wrapper(pipes):
    outs = []
    for p in pipes:
        p.schedule_run()
    for p in pipes:
        outs.append([])
        dev_id = p.device_id
        torch_gpu_device = torch.device('cuda', dev_id)
        torch_cpu_device = torch.device('cpu')
        out = p.share_outputs()
        for o in out:
            o = o.as_tensor()
            if type(o) is TensorGPU:
                tensor_type = torch_gpu_device
            else:
                tensor_type = torch_cpu_device

            t = torch.empty(o.shape(),
                            dtype=to_torch_type[np.dtype(o.dtype())],
                            device=tensor_type)
            if isinstance(o, (TensorGPU, TensorListGPU)):
                # Using same cuda_stream used by torch.zeros to set the memory
                stream = torch.cuda.current_stream(device=dev_id)
                feed_ndarray(o, t, cuda_stream=stream)
            else:
                feed_ndarray(o, t)
            outs[-1].append(t)
        p.release_outputs()
    return outs
