import torch
import numpy as np
from espnet.transform.spec_augment import spec_augment

class SpecAugment(object):
    def __init__(self,
                 resize_mode="PIL",
                 max_time_warp=80,
                 max_freq_width=27,
                 n_freq_mask=2,
                 max_time_width=100,
                 n_time_mask=2,
                 inplace=True,
                 replace_with_zero=True):
        self.__resize_mode = resize_mode
        self.__max_time_warp = max_time_warp
        self.__max_freq_width = max_freq_width
        self.__n_freq_mask = n_freq_mask
        self.__maxt_time_width = max_time_width
        self.__n_time_mask = n_time_mask
        self.__inplace = inplace
        self.__replace_with_zero = replace_with_zero

    def __call__(self, sample):
        feat = np.array(sample['feats'])
        feat = spec_augment(feat,
                            resize_mode=self.__resize_mode,
                            max_time_warp=self.__max_time_warp,
                            max_freq_width=self.__max_freq_width,
                            n_freq_mask=self.__n_freq_mask,
                            max_time_width=self.__maxt_time_width,
                            n_time_mask=self.__n_time_mask,
                            inplace=self.__inplace,
                            replace_with_zero=self.__replace_with_zero)
        sample['feats'] = torch.from_numpy(feat)
        return sample
