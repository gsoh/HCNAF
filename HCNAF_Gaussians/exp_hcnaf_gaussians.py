"""
# Author      : GS Oh
# Experiment  : The gaussian experiments presented in the HCNAF paper
# Note        : Define the gaussian experiments
"""
# GS: if condition == 1, then generate 8 gaussians, else 4 gaussians
import torch
import sklearn
import numpy as np


def sample_dataset(num_samples, args, flag_sample_target_condition=False, target_condition=None):

    rng = np.random.RandomState()
    
    if args.dataset == 'gaussians_exp1':
        dataset = []
        list_condition = [0,1,2]
        list_condition_prob = [1/3, 1/3, 1/3]
        for i in range(num_samples):
            if flag_sample_target_condition:
                idx_condition = target_condition
            else:
                idx_condition = np.random.choice(list_condition, 1, p=list_condition_prob)
            if idx_condition == list_condition[0]:
                point = args.distr_twobytwo.sampler(1)
            elif  idx_condition == list_condition[1]:
                point = args.distr_fivebyfive.sampler(1)
            elif  idx_condition == list_condition[2]:
                point = args.distr_tenbyten.sampler(1)
            if args.add_noise == 'Y':
                dataset.append(np.append(idx_condition + 0.01*np.random.randn(), point))
            else:
                dataset.append(np.append(idx_condition, point))
        dataset = np.array(dataset, dtype='float32')
        
        return dataset

    elif args.dataset == 'gaussians_exp2':
        dataset = []

        conditions_train = [[4,4], [4,12], [8,8], [12,4], [12,12]]
        Num_c_train = len(conditions_train)
        for i in range(num_samples):
            if flag_sample_target_condition:
                center = target_condition
            else:
                idx_condition = np.random.choice(range(Num_c_train), 1, p=1/(Num_c_train*np.ones(Num_c_train)))
                center = conditions_train[idx_condition.item()]
            point = rng.randn(2) * 0.5
            point[0] += center[0]
            point[1] += center[1]
            if args.add_noise == 'Y':
                dataset.append(np.append(center + 0.1*np.random.randn(1,2), point))
            else:
                dataset.append(np.append(center, point))
        dataset = np.array(dataset, dtype='float32')
        
        return dataset

    else:
        raise RuntimeError