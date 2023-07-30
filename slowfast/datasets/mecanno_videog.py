#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from argparse import Namespace
import os
import random
import torch
import torch.utils.data
from fvcore.common.file_io import PathManager
import torchvision.io as io
import torch.multiprocessing as mp
from slowfast.config.defaults import assert_and_infer_cfg

# Set the start method to 'spawn' to avoid CUDA-related issues in multiprocessing.
# This should be called before any GPU-related code is executed.
# mp.set_start_method('spawn', force=True)


import slowfast.utils.logging as logging
from slowfast.utils.parser import load_config

from ..datasets  import decoder as decoder
from ..datasets  import transform as transform
from ..datasets  import utils as utils
from ..datasets  import video_container as container
from .build import DATASET_REGISTRY
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
from ..datasets  import sampling
logger = logging.get_logger(__name__)
# from nonechucks import SafeDataset
import torchvision
try:
    torchvision.set_video_backend('cuda')
except:
        torchvision.set_video_backend('pyav')


@DATASET_REGISTRY.register()
class Meccano_videog(torch.utils.data.Dataset):
    """
    MECCANO video loader. Construct the MECCANO video loader, then sample
    clips from the videos. For training and validation, a single clip is
    randomly sampled from every video with random cropping, scaling, and
    flipping. For testing, multiple clips are uniformaly sampled from every
    video with uniform cropping. For uniform cropping, we take the left, center,
    and right crop if the width is larger than height, or take top, center, and
    bottom crop if the height is larger than the width.
    """

    def __init__(self, cfg, mode, num_retries=10):
        """
        Construct the MECCANO images loader with a given csv file. The format of 
        the csv file is:
        '''
        video_id_1, action_id_1, action_name_1, frame_start_1, frame_end_1
        video_id_2, action_id_2, action_name_2, frame_start_2, frame_end_2
        ...
        video_id_N, action_id_N, action_name_N, frame_start_N, frame_end_N
        '''
        Args:
            cfg (CfgNode): configs.
            mode (string): Options includes `train`, `val`, or `test` mode.
                For the train and val mode, the data loader will take data
                from the train or val set, and sample one clip per video.
                For the test mode, the data loader will take data from test set,
                and sample multiple clips per video.
            num_retries (int): number of retries.
        """
        # Only support train, val, and test mode.
        self.device= torch.device("cuda" if torch.cuda.is_available() else "cpu")

        assert mode in [
            "train",
            "val",
            "test",
        ], "Split '{}' not supported for MECCANO".format(mode)
        self.mode = mode
        self.cfg = cfg

        self._video_meta = {}
        self._num_retries = num_retries
        # For training or validation mode, one single clip is sampled from every
        # video. For testing, NUM_ENSEMBLE_VIEWS clips are sampled from every
        # video. For every clip, NUM_SPATIAL_CROPS is cropped spatially from
        # the frames.
        if self.mode in ["train", "val"]:
            self._num_clips = 1
        elif self.mode in ["test"]:
            self._num_clips = (
                cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS
            )

        logger.info("Constructing MECCANO {}...".format(mode))
        self._construct_loader()

    def _construct_loader(self):
        """
        Construct the data loader.
        """
        path_to_file = os.path.join(
            self.cfg.DATA.PATH_TO_DATA_DIR, "{}.csv".format(self.mode)
        )
        #print("path_to_file:", path_to_file)
        assert PathManager.exists(path_to_file), "{} dir not found".format(
            path_to_file
        )

        self._path_to_videos = []
        self._labels = []
        self._spatial_temporal_idx = []
        self._frame_start = []
        self._frame_end = []
        with PathManager.open(path_to_file, "r") as f:
            #print("splitlines", f.read().splitlines())
            for clip_idx, path_label in enumerate(f.read().splitlines()):
                if clip_idx == 0:
                    continue
                #print("clip_idx:", clip_idx)
                #print("path_label:", path_label)
                assert len(path_label.split(',')) == 5
                video_path, action_label, action_noun, frame_start, frame_end  = path_label.split(',')
                for idx in range(self._num_clips):
                    self._path_to_videos.append(
                        os.path.join(self.cfg.DATA.PATH_PREFIX, video_path)
                    )
                    self._frame_start.append(frame_start)
                    self._frame_end.append(frame_end)
                    self._labels.append(int(action_label))
                    self._spatial_temporal_idx.append(idx)
                    self._video_meta[clip_idx * self._num_clips + idx] = {}
        assert (
            len(self._path_to_videos) > 0
        ), "Failed to load MECCANO split {} from {}"+path_to_file
        logger.info(
            "Constructing MECCANO dataloader (size: {}) from {}".format(
                len(self._path_to_videos), path_to_file
            )
        )

    def __getitem__(self, index):
        """
        Given the video index, return the list of frames, label, and video
        index.
        Args:
            index (int): the video index provided by the pytorch sampler.
        Returns:
            frames (tensor): the frames of sampled from the video. The dimension
                is `channel` x `num frames` x `height` x `width`.
            label (int): the label of the current video.
            index (int): if the video provided by pytorch sampler.
        """
        if self.mode in ["train", "val"]:
            # -1 indicates random sampling.
            temporal_sample_index = -1
            spatial_sample_index = -1
            min_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[0]
            max_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[1]
            crop_size = self.cfg.DATA.TRAIN_CROP_SIZE
        elif self.mode in ["test"]:
            temporal_sample_index = (
                self._spatial_temporal_idx[index]
                // self.cfg.TEST.NUM_SPATIAL_CROPS
            )
            # spatial_sample_index is in [0, 1, 2]. Corresponding to left,
            # center, or right if width is larger than height, and top, middle,
            # or bottom if height is larger than width.
            spatial_sample_index = (
                self._spatial_temporal_idx[index]
                % self.cfg.TEST.NUM_SPATIAL_CROPS
            )
            min_scale, max_scale, crop_size = [self.cfg.DATA.TEST_CROP_SIZE] * 3
            # The testing is deterministic and no jitter should be performed.
            # min_scale, max_scale, and crop_size are expect to be the same.
            assert len({min_scale, max_scale, crop_size}) == 1
        else:
            raise NotImplementedError(
                "Does not support {} mode".format(self.mode)
            )

        # Recover frames
        # frames = []
        folder_name = self._path_to_videos[index]
        start_frame = int(self._frame_start[index][:-4])
        name_frame = str(self._frame_start[index][:-4])
        # if(len(name_frame) == 4): #add a prefix 0
        #     name_frame = "0"+name_frame
        # elif(len(name_frame) == 3): #add two prefix 0
        #     name_frame = "00"+name_frame
        # elif(len(name_frame) == 2): #add three prefix 0
        #     name_frame = "000"+name_frame
        # elif(len(name_frame) == 1): #add four prefix 0
        #     name_frame = "0000"+name_frame
        # Construct the video file path.
        # print('pissi')
        video_path = os.path.join(
            self.cfg.DATA.PATH_TO_DATA_DIR,self.mode, f"{folder_name}_{name_frame}.mp4"
        )
        depth_path = os.path.join(
            self.cfg.DATA.PATH_TO_DATA_DIR,self.mode+'d', f"{folder_name}_{name_frame}.mp4"
        )
        gaze_path = os.path.join(
            self.cfg.DATA.PATH_TO_DATA_DIR,self.mode+'g', f"{folder_name}_{name_frame}.mp4"
        )
        # print(video_path)
        if os.path.exists(video_path):
            # Read video data as bytes.
            # with open(video_path, 'rb') as f:
                # video_data = f.read()

            # Read video frames and audio frames using torchvision.io._read_video_from_memory.
            # vframes, aframes = io._read_video_from_memory(video_data)
            # frames=[]
            # reader = torchvision.io.VideoReader(video_path, "video")
            # reader.seek(2)
            # for frame in reader:
                # frames.append(frame['data'])
            # frames= torch.stack(frames)
            vframes, aframes, info = io.read_video(video_path)
            dframes, aframes, info = io.read_video(depth_path)
            gframes, aframes, info = io.read_video(gaze_path)

            # frames = vframes.to(self.device)
            # vframes will be of shape (T, H, W, C)
            # aframes will be of shape (L, K) where L is the number of audio points and K is the number of channels.

            # Since you only want to work with video frames, you can keep using the vframes tensor.
            # You can convert it to torch.Tensor and use it in the rest of your function as before.
            frames = vframes
            framesd = dframes
            framesg=gframes


            # If you want to use audio frames, you can do so by using the aframes tensor.

        else:
            print(video_path)
        #sampling frames
        
        frames = sampling.temporal_sampling(frames, int(self._frame_start[index][:-4]), int(self._frame_end[index][:-4]), self.cfg.DATA.NUM_FRAMES)
        framesd = sampling.temporal_sampling(framesd, int(self._frame_start[index][:-4]), int(self._frame_end[index][:-4]), self.cfg.DATA.NUM_FRAMES)
        framesg = sampling.temporal_sampling(framesg, int(self._frame_start[index][:-4]), int(self._frame_end[index][:-4]), self.cfg.DATA.NUM_FRAMES)

        # Perform color normalization.
        frames = frames / 255.0
        framesd = framesd / 255.0
        framesg = framesg / 255.0
        mm=torch.tensor(self.cfg.DATA.MEAN).to(frames.device)
        std=torch.tensor(self.cfg.DATA.STD).to(frames.device)
        frames = frames - mm
        frames = frames / std
        framesd = framesd - mm
        framesd = framesd / std
        framesg = framesg - mm
        framesg = framesg / std
        del mm
        del std
        # T H W C -> C T H W.
        frames = frames.permute(3, 0, 1, 2)
        framesd = framesd.permute(3, 0, 1, 2)
        framesg = framesg.permute(3, 0, 1, 2)
        framesd=torch.nn.functional.interpolate(
        framesd,
        size=(frames.shape[2], frames.shape[3]),
        mode="bilinear",
        align_corners=False,
        )
        # Perform data augmentation.
        lener=frames.shape[1]
        frames=torch.cat((frames,framesd,framesg),dim=1)
        frames = self.spatial_sampling(
                frames,
                spatial_idx=spatial_sample_index,
                min_scale=min_scale,
                max_scale=max_scale,
                crop_size=crop_size,
                random_horizontal_flip=self.cfg.DATA.RANDOM_FLIP,
            )
        frames,framesd,framesg=torch.split(frames,lener,dim=1)
        framesd=self.pack_pathway_output(framesd)
        label = torch.tensor(self._labels[index])
        index = torch.tensor(index)
        frames = [framesd, frames,framesg]
        
        return frames, label, index, {}
    def pack_pathway_output( self,frames):
            """
            Prepare output as a list of tensors. Each tensor corresponding to a
            unique pathway.
            Args:
                frames (tensor): frames of images sampled from the video. The
                    dimension is `channel` x `num frames` x `height` x `width`.
            Returns:
                frame_list (list): list of tensors with the dimension of
                    `channel` x `num frames` x `height` x `width`.
            """
            fast_pathway = frames
            # Perform temporal sampling from the fast pathway.
            slow_pathway = torch.index_select(
                frames,
                1,
                torch.linspace(
                    0, frames.shape[1] - 1, frames.shape[1] // self.cfg.SLOWFAST.ALPHA
                ).long().to(fast_pathway.device),
            )
            return slow_pathway
            
    def __len__(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return len(self._path_to_videos)
    @property
    def num_videos(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return len(self._path_to_videos)

    def spatial_sampling(
        self,
        frames,
        spatial_idx=-1,
        min_scale=256,
        max_scale=320,
        crop_size=224,
        random_horizontal_flip=True,
    ):
        """
        Perform spatial sampling on the given video frames. If spatial_idx is
        -1, perform random scale, random crop, and random flip on the given
        frames. If spatial_idx is 0, 1, or 2, perform spatial uniform sampling
        with the given spatial_idx.
        Args:
            frames (tensor): frames of images sampled from the video. The
                dimension is `num frames` x `height` x `width` x `channel`.
            spatial_idx (int): if -1, perform random spatial sampling. If 0, 1,
                or 2, perform left, center, right crop if width is larger than
                height, and perform top, center, buttom crop if height is larger
                than width.
            min_scale (int): the minimal size of scaling.
            max_scale (int): the maximal size of scaling.
            crop_size (int): the size of height and width used to crop the
                frames.
        Returns:
            frames (tensor): spatially sampled frames.
        """
        assert spatial_idx in [-1, 0, 1, 2]
        if spatial_idx == -1:
            frames, _ = transform.random_short_side_scale_jitter(
                images=frames,
                min_size=min_scale,
                max_size=max_scale,
                inverse_uniform_sampling=self.cfg.DATA.INV_UNIFORM_SAMPLE,
            )
            frames, _ = transform.random_crop(frames, crop_size)
            if random_horizontal_flip:
                frames, _ = transform.horizontal_flip(0.5, frames)
        else:
            # The testing is deterministic and no jitter should be performed.
            # min_scale, max_scale, and crop_size are expect to be the same.
            assert len({min_scale, max_scale, crop_size}) == 1
            frames, _ = transform.random_short_side_scale_jitter(
                frames, min_scale, max_scale
            )
            frames, _ = transform.uniform_crop(frames, crop_size, spatial_idx)
        return frames
    
