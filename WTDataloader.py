from torchvision.datasets.utils import list_dir
from torchvision.datasets.folder import make_dataset
from torchvision.datasets.video_utils import VideoClips
from torchvision.datasets import VisionDataset
import torchvision
import os
from PIL import Image
import torch
import decord
import numpy as np
import random


class DecordInit(object):

    def __init__(self, num_threads=1, **kwargs):
        self.num_threads = num_threads
        self.ctx = decord.cpu(0)
        self.kwargs = kwargs
        
    def __call__(self, filename):
        
        reader = decord.VideoReader(filename,
                                    ctx=self.ctx,
                                    num_threads=self.num_threads)
        return reader

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'sr={self.sr},'
                    f'num_threads={self.num_threads})')
        return repr_str

class WT_dataset_1vid(torch.utils.data.Dataset):
    

    def __init__(self,
                 video_path,
                 num_frames,
                 step_between_clips,
                 transform=None):
        
        self.path = video_path

        self.transform = transform
        self.num_frames = num_frames
        self.step_between_clips = step_between_clips
        self.v_decoder = DecordInit()
        v_reader = self.v_decoder(self.path)
        total_frames = len(v_reader)
        
        self.total_frames = total_frames 

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __getitem__(self, index):
        while True:
            try:
               
                v_reader = self.v_decoder(self.path)
                total_frames = len(v_reader)
                
                # Sampling video frames
                start_frame_ind = index 
                end_frame_ind = start_frame_ind + (self.num_frames * self.step_between_clips)


                frame_indice = np.arange(start_frame_ind, end_frame_ind, self.step_between_clips, dtype=int)
                
                video = v_reader.get_batch(frame_indice).asnumpy()
                del v_reader
                break
            except Exception as e:
                print(e)
                
        with torch.no_grad():
            video = torch.from_numpy(video)
            if self.transform is not None:
                video = self.transform(video)
                
        return video


    def __len__(self):
        return self.total_frames - (self.num_frames * self.step_between_clips)


