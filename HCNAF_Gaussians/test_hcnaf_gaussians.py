"""
# Author      : GS Oh
# Experiment  : The gaussian experiments presented in the HCNAF paper
# Note        : Test the HCNAF models for the gaussian experiments using 2D plots
"""
# Import pytorch libraries
import torch

# Import datasets
import exp_naf                                   # Import naf datasets
from exp_hcnaf_gaussians import sample_dataset   # Import hcnaf dataset

# Import ETC
import argparse, pprint, json
from types import SimpleNamespace
import copy, os
import time, datetime 
import numpy as np
import matplotlib.pyplot as plt

# Import HCNAF files
from hcnaf_gaussians import *
from util_gaussians import *


def main():
    parser = argparse.ArgumentParser(description="Testing HCNAF models for the gaussian experiments presented in the HCNAF paper")

    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--dataparallel', type=bool, default=False)
    parser.add_argument('--loadpath', type=str, default='results/gaussians_exp1_noiseN_hypernetlayers2.1_normmw_layers2_h64_B64/')
    parser.add_argument('--loadfilename', type=str, default='iteration39500_checkpoint_Bestiteration39500_BestLoss3.9658.pt')
    parser.add_argument('--task', type=str, default='NLL', choices=['NLL', 'plot'])

    args_new = parser.parse_args()

    print('Arguments for testing:')
    pprint.pprint(args_new.__dict__)

    # load the args for the model to be tested
    args = json.load(open(args_new.loadpath + "args.json","r"))
    args = SimpleNamespace(**args) # convert dictionary to namespace
    args.device = args_new.device
    args.dataparallel = args_new.dataparallel
    args.loadfilename = args_new.loadfilename

    print('Arguments of the loaded model:')
    pprint.pprint(args.__dict__)
    
    print('Creating HCNAF model')
    best_model = create_HCNAF(args)

    print('Loading state_dicts for the model')
    model_statedict, _, _, _, _, best_model_statedict, _ = load_state(args_new.loadpath + args_new.loadfilename)
    best_model.load_state_dict(best_model_statedict)
    #best_model.load_state_dict(model_statedict)

    print('Initializing distributions')
    if args.dataset == 'gaussians_exp1':
        args.distr_twobytwo = exp_naf.NByN(2)
        args.distr_fivebyfive = exp_naf.NByN(5)
        args.distr_tenbyten = exp_naf.NByN(10)
    
    print('Testing')
    if args.dataset == 'gaussians_exp1':
        if args_new.task == 'plot':
            plot_de_2d(best_model, args, condition=0, batch_size_plot=500)            
            plot_de_2d(best_model, args, condition=1, batch_size_plot=500)            
            plot_de_2d(best_model, args, condition=2, batch_size_plot=500)
        elif args_new.task == 'NLL':
            compute_NLL(best_model, args, target_condition=0)
            compute_NLL(best_model, args, target_condition=1)
            compute_NLL(best_model, args, target_condition=2)
    elif args.dataset == 'gaussians_exp2':
        if args_new.task == 'plot':
            plot_de_2d_multiple(best_model, args, conditions_name='seen', batch_size_plot=200)
            plot_de_2d_multiple(best_model, args, conditions_name='unseen', batch_size_plot=200)
            # plot_de_2d(best_model, args, condition=[4, 4], batch_size_plot=200)
        elif args_new.task == 'NLL':
            compute_NLL(best_model, args, target_condition=[4, 4])
            compute_NLL(best_model, args, target_condition=[4, 12])
            compute_NLL(best_model, args, target_condition=[8, 8])
            compute_NLL(best_model, args, target_condition=[12, 4])
            compute_NLL(best_model, args, target_condition=[12, 12])
            compute_NLL(best_model, args, target_condition=[4, 8])
            compute_NLL(best_model, args, target_condition=[12, 8])
            compute_NLL(best_model, args, target_condition=[8, 4])
            compute_NLL(best_model, args, target_condition=[8, 12])
    else:
        raise RuntimeError
            
if __name__ == '__main__':
    main()