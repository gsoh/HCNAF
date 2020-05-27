"""
# Author      : GS Oh
# Experiment  : The gaussian experiments presented in the HCNAF paper
# Note        : Train HCNAF models for the gaussian experiments
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



def train_gaussians(model, optimizer, scheduler, args):
    flag_save = 0

    if args.resume_training:
        last_iteration = args.last_iteration
        val_loss_min = args.best_loss
        best_iteration = args.best_iteration
        best_model_state = model.state_dict()
    else:
        last_iteration = 0
        val_loss_min = float('inf')
        best_iteration = 'NA'
        best_model_state = None

    for iteration in range(last_iteration+1, args.iterations):
        model.train()
        data = torch.from_numpy(sample_dataset(args.batch_dim, args)).float().to(args.device)
        loss = - compute_log_p_x(model, data).mean()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_gradnorm_max)

        optimizer.step()
        optimizer.zero_grad()

        if iteration%args.savemodel_period == 0 and iteration > 0:
            val_loss = 0

            model.eval()
            with torch.no_grad():
                # mean of 1,000 samples
                for idx_val in range(100):
                    data = torch.from_numpy(sample_dataset(10, args)).float().to(args.device)
                    val_loss += - compute_log_p_x(model, data).mean()
                val_loss /= 100
                val_loss = round(val_loss.item(), 4)

                if val_loss < val_loss_min:
                    val_loss_min = val_loss
                    best_iteration = iteration 
                    best_model_state = model.state_dict()
                    flag_save = 1

                if flag_save:
                    print('Saving progress, saving the current best model, at iteration {}, best loss {}'.format(best_iteration, val_loss_min))
                    save_name = 'iteration' + str(iteration) + '_checkpoint'
                    save_dict = {
                        'save_path': args.path + save_name,
                        'last_iteration': iteration,
                        'best_iteration': best_iteration,
                        'best_loss': val_loss_min,
                        'best_model_state_dict': best_model_state,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict()
                    }
                    save_state(save_dict)

                    flag_save = 0

                scheduler.step(loss)
            
            # print & save loss & val_loss
            perf_eval = 'Iteration [{}/{}], Val_Loss: {:.4f}, Val_Loss_min: {:.4f}'.format(
                iteration, args.iterations, val_loss, val_loss_min)
            print(perf_eval, file=open(args.path + 'Perf_epoch.txt', "a"))

        print('Iteration {}, Loss: {}'.format(iteration, round(loss.item(), 4)))
    
    # At the end of training, save the current model.
    print('Saving the final model state, saving the current best model, at iteration {}, best loss {}'.format(best_iteration, val_loss_min))
    save_name = 'iteration' + str(iteration) + '_checkpoint'
    save_dict = {
        'save_path': args.path + save_name,
        'last_iteration': iteration-1,
        'best_iteration': best_iteration,
        'best_loss': val_loss_min,
        'best_model_state_dict': best_model_state,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict()
    }
    save_state(save_dict) 


def main():
    parser = argparse.ArgumentParser(description="Process Hyper-parameters of HCNAF training for Gaussian experiments")

    # Arguments for main objective of the script
    parser.add_argument('--device', type=str, default='cuda:0', choices=['cpu', 'cuda:0'])
    parser.add_argument('--dataparallel', type=bool, default=False)
    parser.add_argument('--iterations', type=int, default=200000)
    parser.add_argument('--resume_training', type=bool, default=False)

    # Below arguments are used only when resume_training == true 
    parser.add_argument('--loadpath', type=str, default='')
    parser.add_argument('--loadfilename', type=str, default='')
    
    #### Below arguments are used only when resume_training = false. Otherwise, the program loads arguments from args.json.
    # Dataset-related hyper-parameters
    parser.add_argument('--dataset', type=str, default='gaussians_exp1', choices=['gaussians_exp1', 'gaussians_exp2'])
    parser.add_argument('--add_noise', type=str, default='N')

    # Optimizer-related hyper-parameters
    parser.add_argument('--learning_rate', type=float, default=5e-3)
    parser.add_argument('--batch_dim', type=int, default=64)
    parser.add_argument('--clip_gradnorm_max', type=float, default=0.1, help='Threshold the magnitude of gradients of Hypernetwork parameters')
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--decay', type=float, default=0.5)
    
    # Model hyper-parameters
    parser.add_argument('--n_layers_flow', type=int, default=2)
    parser.add_argument('--dim_h_flow', type=int, default=64)
    parser.add_argument('--hypernet_layers', type=str, default='2', choices=['2s', '2', '2b', '3'])
    parser.add_argument('--norm_HW', type=str, default='modified_weightnorm', choices=['modified_weightnorm', 'scaled_frobenius'])

    # ETC
    parser.add_argument('--savemodel_period', type=int, default=100)

    args_new = parser.parse_args()

    print('Arguments:')
    pprint.pprint(args_new.__dict__)

    if args_new.dataset in ['gaussians_exp2']:
        args_new.cond_dim = 2
        args_new.dim_o = 2
        args_new.epoch = 'NA'
    elif args_new.dataset in ['gaussians_exp1']:
        args_new.cond_dim = 1
        args_new.dim_o = 2
        args_new.epoch = 'NA'
        
    if not os.path.exists("results"):
        os.mkdir("results")
    
    if args_new.resume_training: # Resume training
        print('Resuming a previous training')
        # resume_training the model, optimizer, and scheduler
        model_state, optimizer_state, scheduler_state, best_iteration, best_loss, best_model_state, last_iteration = load_state(args_new.loadpath + args_new.loadfilename)

        # resume_training args
        args = json.load(open(args_new.loadpath + "args.json","r"))
        args = SimpleNamespace(**args) # convert dictionary to namespace

        print("Arguments are loaded from args.json except the following parameters: device, dataparallel, epoch")
        if args_new.resume_training:
            args.device = args_new.device
            args.iterations = args_new.iterations
            args.dataparallel = args_new.dataparallel
            args.resume_training = args_new.resume_training
    else: # start a new training
        print('Starting a new training') 
        args = args_new

        args.loadpath = 'N/A'
        args.loadfilename = 'N/A'
        if args.norm_HW == 'scaled_frobenius':
            norm_HW_short = 'sf'
        elif args.norm_HW == 'modified_weightnorm':
            norm_HW_short = 'mw'
        args.path = 'results/{}_noise{}_hypernetlayers{}_norm{}_layers{}_h{}_B{}_{}/'.format(args.dataset, args.add_noise, args.hypernet_layers, norm_HW_short, args.n_layers_flow, args.dim_h_flow, args.batch_dim, str(datetime.datetime.now())[:-7].replace(' ', '-').replace(':', '-'))

        print('create a directory and save the arguments')
        os.mkdir(args.path)
        with open(args.path + 'args.json', 'w') as f:
            json.dump(args.__dict__, f, indent=4, sort_keys=True)

    # Create a HCNAF model, an optimizer, and a scheduler
    print('Creating model, optimizer, and scheduler')
    model = create_HCNAF(args)
    if args.dataparallel:
        model = torch.nn.DataParallel(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=args.decay, patience=args.patience, min_lr=1e-5, verbose=True, threshold_mode='abs')

    if args_new.resume_training:
        model.load_state_dict(best_model_state)
        optimizer.load_state_dict(optimizer_state)
        scheduler.load_state_dict(scheduler_state)
        args.last_iteration = last_iteration
        args.best_iteration = best_iteration
        args.best_loss = best_loss
        args.last_iteration = scheduler_state['last_epoch']
    
    print('Initializing distributions')
    if args.dataset == 'gaussians_exp1':
        args.distr_twobytwo = exp_naf.NByN(2)
        args.distr_fivebyfive = exp_naf.NByN(5)
        args.distr_tenbyten = exp_naf.NByN(10)
    
    print('Starting the training process')
    train_gaussians(model, optimizer, scheduler, args)
    

if __name__ == '__main__':
    main()
