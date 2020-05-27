"""
# Author      : GS Oh
# Experiment  : PRECOG_Carla
# Note        : Run trained models to plot POM forecastings & compute NLL
"""
# Import pytorch libraries
import torch
from torch import nn
from torch import optim
from torch import autograd
import torch.functional as F

# Import ETC
import argparse, pprint, json, os
from types import SimpleNamespace
import time, datetime 
import numpy as np

# Import plotting libraries
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['font.size']=15
matplotlib.style.use('dark_background')

# Import HCNAF functions and utils
from util_dataset_PRECOG_Carla import *
from util_PRECOG_Carla import *
from util_visualize_PRECOG_Carla import plot_POMap_PRECOG_Carla
from hcnaf_PRECOG_Carla import *

from train_PRECOG_Carla import collect_inputs_and_compute_loss_PRECOG


def compute_extra_nats(model, data_loader, args):
    print('Computing extra nats ...')
    diff_entropy_eta = -6.372463 # (TAD)/2 * (1+log(2*pi*|Sigma|))

    if args.dim_c_EncInput == 2:
        args.lidar_input_c = [1,2]
    elif args.dim_c_EncInput == 3:
        args.lidar_input_c = [1,2,3]

    Gaussian2D = MultivariateNormal(torch.zeros(2), 0.0001*torch.eye(2))

    with torch.no_grad():
        data_loader_iter = iter(data_loader)
        args.loss_collector = np.zeros(len(args.forecast_which_frame))
        loss_sum = 0
        
        model.eval() # switch to the eval mode (disable batch_norm & drop_out if there is any)
        for i, loaded_data in enumerate(data_loader_iter):
            loss_sum += collect_inputs_and_compute_loss_PRECOG(model, None, loaded_data, Gaussian2D, True, args)
            if i%1000 == 0 and i>0:
                print(i, ' iterations finished')
                print('printing out intermediate results...')
                for idx_eval, idx_time_eval in enumerate(args.forecast_which_frame):
                    print('Testset, loss at time {}: {}'.format(idx_time_eval, args.loss_collector[idx_eval]/i))

        for idx_eval, idx_time_eval in enumerate(args.forecast_which_frame):
            print('Testset, loss at time {}: {}, extra nats: {}'.format(idx_time_eval, args.loss_collector[idx_eval]/len(data_loader), (args.loss_collector[idx_eval]/len(data_loader)-diff_entropy_eta)/2))

        loss_mean = loss_sum/(len(args.forecast_which_frame)*len(data_loader))
        Extra_nats = (loss_mean - diff_entropy_eta)/2        
    
    PNLL_n_enats = "Dataset:{}, Num_p:{}, Perturbed cross entropy:{}, extra_nats:{}".format(data_loader.dataset.data_subset, args.num_perturbations, loss_mean, Extra_nats)
    print(PNLL_n_enats)
    txt_fn_enats = 'extra_nats_' + args.loadfilename + '.txt'
    print(PNLL_n_enats, file=open(os.path.join(args.path, txt_fn_enats), "a"))

def main():
    #### Hyper-parameters for plotting POMs ####
    parser = argparse.ArgumentParser(description = "Process Hyper-parameters for plotting POMs for PRECOG exp")

    parser.add_argument('--loadpath', type=str, default='results/PRECOG_Carla_tiny_AModeNo_lidar_faster_temporal_temporal1_normHWmw_hdt10_lossPNLL_output_layers3_h100_layersRNN2_hRNN24_sdvhistsec1_tgap0.2_B4/')
    parser.add_argument('--loadfilename', type=str, default='epoch3_BestEpoch3_BestLoss-5.6301.pt')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--task', type=str, default='plot', choices=['extra_PNLL', 'plot'])
    parser.add_argument('--whichset_to_plot', type=str, default='test_set', choices=['train_set','val_set','test_set'])
    parser.add_argument('--plot_method', type=str, default='Log', choices=['Log','Max'])
    parser.add_argument('--plot_index', type=str, default=list(range(200,500,5)), help='Two ways to define this variable: (1) using range iterator: list(range()), (2) manually listing indices of interest: [1,3,612,...]') 

    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--batch_size_test', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=10)
    parser.add_argument('--b_size_plot', type=int, default=128*2)
    parser.add_argument('--res', type=int, default=1)

    args_new = parser.parse_args()
    args_new.idx_last_seq_toplot = max(args_new.plot_index)+1

    print('Plotting, Arguments:')
    pprint.pprint(args_new.__dict__)


    #### Load hyper-parameters for the trained model ####
    args = json.load(open(args_new.loadpath + "args.json","r"))
    args = SimpleNamespace(**args) # convert dictionary to namespace
    if not hasattr(args, 'Pool_method'): # args doesn't have 'Pool_method', set to 'fmp'
        args.Pool_method = 'fmp'
    args.num_perturbations = 10
    args.loadfilename = args_new.loadfilename

    print('Arguments:')
    pprint.pprint(args.__dict__)

    # Load the dataset
    if args.dataset_name == 'PRECOG_Carla':
        dataset_tr = DatasetFolderSamples_PRECOG_Carla('town1/train/', args)
        dataset_val = DatasetFolderSamples_PRECOG_Carla('town1/val/', args)
        dataset_test = DatasetFolderSamples_PRECOG_Carla('town1/test/', args)
    elif args.dataset_name == 'PRECOG_Carla_tiny':
        dataset_tr = DatasetFolderSamples_PRECOG_Carla('town1/train_tiny/', args) # 5% of town1_train
        dataset_val = DatasetFolderSamples_PRECOG_Carla('town1/val_tiny/', args) # 5% of town1_val
        dataset_test = DatasetFolderSamples_PRECOG_Carla('town1/test_tiny/', args) # 5% of town1_test
    train_loader, val_loader, test_loader = set_dataloader_PRECOG_Carla(dataset_tr, dataset_val, dataset_test, args_new.batch_size, args_new.batch_size_test, args_new.num_workers)

    # Check the dimensions of data
    refcar_states_hist, refcar_states_future, refcar_transform, traffic_states_hist, traffic_states_future, traffic_transform, lidar_overhead_f, params, identifier = next(iter(train_loader))
    print('Dim: refcar_hist, refcar_future, traffic_hist, traffic_future, lidar_overhead_features')
    print(refcar_states_hist.shape)       # (B, seq_len_hist, 2)
    print(refcar_states_future.shape)     # (B, seq_len_future, 2)
    print(traffic_states_hist.shape)      # (B, N, seq_len_hist, 2)
    print(traffic_states_future.shape)    # (B, N, seq_len_future, 2)
    print(lidar_overhead_f.shape)                  # (B, H, W, C)

    print('Dim: States_sdv, States_traffic, States_map_emb, States_SS, States_sdv_outputs')
    print(refcar_transform.shape)                  # (B, 4, 4) [history of (x,y,sin(theta),cos(theta),spd)), width & length of sdv]
    print(traffic_transform.shape)                 # (B, N, 4, 4) [(is_exist_actor,x,y,sin(theta),cos(theta),spd)] at the current timestampe (50th frame) 

    args.dim_sdv_hist = refcar_states_hist.shape[1] * refcar_states_hist.shape[2] # seq_len_hist * 2
    args.dim_traffic_hist = traffic_states_hist.shape[2] * traffic_states_hist.shape[3] # seq_len_hist * 2
    args.dim_o = 2

    # Load the model
    best_model = create_HCNAF(args)
    _, _, model_statedict, best_model_statedict, _, _= load_state(args_new.loadpath + args_new.loadfilename)
    #_, best_model_statedict = load_state_intermediate(args_new.loadpath + args_new.loadfilename)
    best_model.load_state_dict(best_model_statedict)
    
    args_new.save_folder_POMap = args_new.whichset_to_plot + '_results_' + args_new.loadfilename + '_' + args_new.plot_method
    if not os.path.exists(os.path.join(args_new.loadpath, args_new.save_folder_POMap)):
        os.mkdir(os.path.join(args_new.loadpath, args_new.save_folder_POMap))

    
    if args_new.task == 'extra_PNLL':          #### Task 1: Compute the extra nats using perturbed distributions
        compute_extra_nats(best_model, test_loader, args)
    elif args_new.task == 'plot':              #### Task 2: Plot POM
        if args_new.whichset_to_plot == 'train_set':
            plot_POMap_PRECOG_Carla(best_model, args, train_loader, args_new)
        elif args_new.whichset_to_plot == 'val_set':
            plot_POMap_PRECOG_Carla(best_model, args, val_loader, args_new)
        elif args_new.whichset_to_plot == 'test_set':
            plot_POMap_PRECOG_Carla(best_model, args, test_loader, args_new)

if __name__ == '__main__':
    main()
