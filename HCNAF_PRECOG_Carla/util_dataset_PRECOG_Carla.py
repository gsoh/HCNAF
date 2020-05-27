"""
# Author      : GS Oh
# Experiment  : PRECOG_Carla
# Note        : Dataset-specific script to read dataset (JSON files), modified from carla_json_loader.py of https://github.com/nrhine1/precog_carla_dataset
"""
import numpy as np
import math, json, os
import matplotlib.pyplot as plt
import attrdict
from random import shuffle

import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler


# Split an input idx_list into three lists of training, validation, test set indices.
def split_dataidx(idx_list, trainset_p=70, valset_p=20, shuffle=False):
    idx_list_len = len(idx_list)
    if shuffle:
        np.random.seed(np.random.randint(0,1000))
        np.random.shuffle(idx_list)

    assert((trainset_p + valset_p) < 100)

    trainset_idx = idx_list[:(idx_list_len*trainset_p)//100]
    valset_idx = idx_list[(idx_list_len*trainset_p)//100:(idx_list_len*(trainset_p+valset_p))//100]
    testset_idx = idx_list[(idx_list_len*(trainset_p+valset_p))//100:]

    print('Total: {}'.format(idx_list_len))
    print('Train set: {}, Validation set: {}, Test set: {}'.format(len(trainset_idx), len(valset_idx), len(testset_idx)))

    return trainset_idx, valset_idx, testset_idx

# A custom sampler
class SubsetSequentialSampler(Sampler):
    r"""Samples elements sequentially from a given list of indices, inspired by torch.utils.data.sampler.SubsetRandomSampler

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)

# Given tr, val, test idx sets, return torch.utils.data.DataLoader
def split_dataloader(dataset, sampler, trainset_idx, valset_idx, testset_idx, batch_size, batch_size_test, N):
    trainset_sampler = sampler(trainset_idx)
    validset_sampler = sampler(valset_idx)
    testset_sampler = sampler(testset_idx)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=trainset_sampler, num_workers=N, pin_memory=False)
    val_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=validset_sampler, num_workers=N, pin_memory=False)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size_test, sampler=testset_sampler, num_workers=0)

    train_batch = next(iter(train_loader))
    val_batch = next(iter(val_loader))
    test_batch = next(iter(test_loader))

    print('number of batches per epoch: (train_set: {}, val_set: {}, test_set: {})'.format(len(train_loader), len(val_loader), len(test_loader)))
    print('size of batches:', train_batch[0].size(0), val_batch[0].size(0), test_batch[0].size(0))

    return train_loader, val_loader, test_loader
    
def load_json(json_fn):
    """Load a json datum.

    :param json_fn: <str> the path to the json datum.
    :returns: dict of postprocess json data.
    """
    assert(os.path.isfile(json_fn))
    json_datum = json.load(open(json_fn, 'r'))
    postprocessed_datum = from_json_dict(json_datum)
    return postprocessed_datum
    
def from_json_dict(json_datum):
    """Postprocess the json datum to ndarray-ify things

    :param json_datum: dict of the loaded json datum.
    :returns: dict of the postprocessed json datum.
    """
    pp = attrdict.AttrDict()
    for k, v in json_datum.items():
        if isinstance(v, list):
            pp[k] = np.asarray(v)
        elif isinstance(v, dict) or isinstance(v, int) or isinstance(v, str):
            pp[k] = v
        else:
            raise ValueError("Unrecognized type")
    return pp


# Given train, val, test datasets, return torch.utils.data.DataLoader
def set_dataloader_PRECOG_Carla(dataset_tr, dataset_val, dataset_test, batch_size, batch_size_test, N):
    train_loader = torch.utils.data.DataLoader(dataset_tr, batch_size=batch_size, num_workers=N, shuffle=True, pin_memory=False)   # Random sampler is used when shuffle is set True
    val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size, num_workers=N, pin_memory=False)                  # Sequential sampler is used when shuffle is set False
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size_test, num_workers=0, pin_memory=False)

    train_batch = next(iter(train_loader))
    val_batch = next(iter(val_loader))
    test_batch = next(iter(test_loader))

    print('number of batches per epoch: (train_set: {}, val_set: {}, test_set: {})'.format(len(train_loader), len(val_loader), len(test_loader)))
    print('size of batches:', train_batch[0].size(0), val_batch[0].size(0), test_batch[0].size(0))

    return train_loader, val_loader, test_loader


class DatasetFolderSamples_PRECOG_Carla(data.Dataset):
    def __init__(self, data_subset, args):       
        assert data_subset in ['town1/train/', 'town1/val/', 'town1/test/', 'town1/train_tiny/', 'town1/val_tiny/', 'town1/test_tiny/', 'town2/test/', 'town2/test_small/'], "Invalid data_subset name: pick one from ['town1/train/', 'town1/val/', 'town1/test/', 'town1/train_tiny/', 'town1/val_tiny/', 'town1/test_tiny/', 'town2/test/', 'town2/test_small/']"

        self.ablation_mode = args.ablation_mode
        self.max_num_actors = args.max_num_actors # used in creating traffic contex
        self.device = args.device
        self.normalization = args.normalization

        self.forecast_which_frame = args.forecast_which_frame
        self.num_seq_input_sdv = args.num_seq_input_sdv
        self.sdv_hist_tgap = args.sdv_hist_tgap
        self.sdv_hist_length_sec = args.sdv_hist_length_sec

        if args.ablation_mode in ['All', 'All_faster_temporal', 'No_lidar', 'No_lidar_faster_temporal']:
            self.traffic_hist_length_sec = args.traffic_hist_length_sec
            self.traffic_hist_tgap = args.traffic_hist_tgap

        self.data_subset = data_subset
        self.len_hist = args.len_hist
        self.len_future = args.len_future
        self.path_json = args.path_json + self.data_subset
        if self.data_subset in ['town1/train/', 'town1/train_tiny/']:
            self.list_seq_filename = [file for file in os.listdir(self.path_json)]
        elif self.data_subset in ['town1/val/', 'town1/val_tiny/']:
            self.list_seq_filename = [file for file in os.listdir(self.path_json)]
        elif self.data_subset in ['town1/test/', 'town1/test_tiny/', 'town2/test/', 'town2/test_small/']:
            self.list_seq_filename = sorted([file for file in os.listdir(self.path_json)])
        
        self.pix_per_m = args.pix_per_m
        self.frame_no_curr = args.frame_no_curr
        self.HW_img = args.HW_img
        self.img_center = self.HW_img//2
        self.len_seq = args.len_seq
        if self.sdv_hist_length_sec == 0:
            self.list_idx_ref_hist = range(self.frame_no_curr, self.frame_no_curr+1)
        elif self.sdv_hist_length_sec == 1:
            if self.sdv_hist_tgap == 0.1:
                self.list_idx_ref_hist = range(0,10,1) # frame 0,1,2,...,9 are used.
            elif self.sdv_hist_tgap == 0.2:
                self.list_idx_ref_hist = range(1,10,2) # frame 1,3,5,7,9 are used.

        if args.ablation_mode in ['All', 'All_faster_temporal', 'No_lidar', 'No_lidar_faster_temporal']:
            if self.traffic_hist_length_sec == 1:
                if self.traffic_hist_tgap == 0.1:
                    self.list_idx_traffic_hist = range(0,10,1) # frame 0,1,2,...,9 are used.
                elif self.traffic_hist_tgap == 0.2:
                    self.list_idx_traffic_hist = range(1,10,2) # frame 1,3,5,7,9 are used.
        else:
            self.list_idx_traffic_hist = range(self.frame_no_curr, self.frame_no_curr+1)
            
    def __getitem__(self, sequence_idx):
        json_fn = self.list_seq_filename[sequence_idx]
        datum = load_json(self.path_json+json_fn)
        
        refcar_states_hist_interest = datum['player_past'][self.list_idx_ref_hist, 0:2]                                  # (seqlen_hist, 2) -> (seqlen_hist*2). 2 represents (x,y). Time gap b/t seqs = 0.1s
        refcar_states_future_interest = datum['player_future'][self.forecast_which_frame, 0:2]                           # (seqlen_future, 2) -> (seqlen_future*2)

        traffic_states_hist_interest = datum['agent_pasts'][:self.max_num_actors, self.list_idx_traffic_hist, 0:2]       # (Num_actors, seqlen_hist, 2) -> (Num_actors, seqlen_hist*2)
        traffic_states_future_interest = datum['agent_futures'][:self.max_num_actors, self.forecast_which_frame, 0:2]    # (Num_actors, seqlen_future, 2) -> (Num_actors, seqlen_future*2)
            
        traffic_transform = datum['agent_transforms']                                        # N x 4 x 4 rotation and translation of the other agents.
        refcar_transform = datum['player_transform']                                         # 4 x 4 rotation and translation matrix
        lidar_overhead_f = datum['overhead_features']                                        # H(200) x W(200) x C(4), lidar overhead features

        assert self.pix_per_m == datum['lidar_params']['pixels_per_meter'], 'inconsistent pixels_per_meter'
        if self.data_subset == 'town2/test/':                                                # Test2 has episode, frame information so that sequences can be attached/detached
            params = [datum['episode'], datum['frame'], datum['lidar_params']]
        else:
            params = datum['lidar_params']                                                   # dict of parameters used to create the overhead features.
        identifier = self.data_subset + json_fn        
    
        ## Normalization        
        if self.normalization: # All normalization is done in pixels
            refcar_states_hist_interest = refcar_states_hist_interest/50
            refcar_states_future_interest = refcar_states_future_interest/50
            traffic_states_hist_interest = traffic_states_hist_interest/50
            traffic_states_future_interest = traffic_states_future_interest/50

        return (refcar_states_hist_interest.astype(np.float32), refcar_states_future_interest.astype(np.float32), refcar_transform.astype(np.float32), traffic_states_hist_interest.astype(np.float32), traffic_states_future_interest.astype(np.float32), traffic_transform.astype(np.float32), lidar_overhead_f.astype(np.float32), params, identifier)


    def __len__(self):
        return len(self.list_seq_filename)


