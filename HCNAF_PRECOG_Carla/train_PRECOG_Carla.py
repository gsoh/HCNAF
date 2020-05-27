"""
# Author      : GS Oh
# Experiment  : PRECOG_Carla
# Note        : Train HCNAF models for POM forecastings.
"""
# Import pytorch libraries
import torch
from torch import autograd

# Import pytorch_vision libraries
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler
from torch.distributions import MultivariateNormal

# Import ETC
import argparse, pprint, json, os
from types import SimpleNamespace
import time, datetime 
import numpy as np

# Import HCNAF functions and utils
from util_dataset_PRECOG_Carla import *
from util_PRECOG_Carla import *
from hcnaf_PRECOG_Carla import *


#
def collect_inputs_and_compute_loss_PRECOG(model, optimizer, loaded_data, Gaussian2D, flag_eval, args):
    if args.ablation_mode in ['All_faster_temporal', 'No_lidar_faster_temporal']: # For these ablation modes, batch size = 1, temporal = 1
        model[0].is_WB_computed = False # reset 'is_WB_computed'

    refcar_states_hist, refcar_states_future, refcar_transform, traffic_states_hist, _, traffic_transform, lidar_overhead_f, params, identifier = loaded_data # traffic_states_future is not used in training. Only for plotting.
    
    lidar_overhead_f = lidar_overhead_f.permute(0,3,1,2)
    inputs_imgs = lidar_overhead_f[:, args.lidar_input_c, :, :]
    batch_size_loop, N_hist_refcar, N_hist_traffic, N_future = refcar_states_hist.size(0), refcar_states_hist.size(1), traffic_states_hist.size(2), refcar_states_future.size(1)

    refcar_states_future = refcar_states_future.transpose(0, 1)                                                               # (B, T, 2)    -> (T, B, 2)
    if args.loss in ['PNLL', 'PNLL_output']:
        inputs_imgs_ptb = inputs_imgs.unsqueeze(1).repeat(1, args.num_perturbations, 1, 1, 1)                                 # (B, C, H, W)  -> (B, P, C, H, W) 
        
        if args.loss == 'PNLL':
            if args.PNLL_method1 == 1: # Identical perturbation given P and input group. Each perturbation is identical for all batches, N_hist_refcar, max_num_actors, and N_hist_traffic.
                Noise_refcar_future = Gaussian2D.sample((N_future, 1, args.num_perturbations))                                # (T, 1, P, 2)    
                Noise_refcar_hist   = Gaussian2D.sample((1, args.num_perturbations, 1))                                       # (1, P, 1, 2) 
                Noise_traffic_hist  = Gaussian2D.sample((1, args.num_perturbations, 1, 1))                                    # (1, P, 1, 2, 2) 

                refcar_states_future_ptb = refcar_states_future.unsqueeze(2).repeat(1, 1, args.num_perturbations, 1)   + Noise_refcar_future.repeat((1, batch_size_loop, 1, 1))                                        # (T, B, 2)                      -> (T, B, P, 2)
                refcar_states_hist_ptb   = refcar_states_hist.unsqueeze(1).repeat(1, args.num_perturbations, 1, 1)     + Noise_refcar_hist.repeat((batch_size_loop, 1, N_hist_refcar, 1))/10                           # (B,    seqlen_hist_refcar,  2) -> (B, P,    seqlen_hist_refcar,  2)
                traffic_states_hist_ptb  = traffic_states_hist.unsqueeze(1).repeat(1, args.num_perturbations, 1, 1, 1) + Noise_traffic_hist.repeat((batch_size_loop, 1, args.max_num_actors, N_hist_traffic, 1))/10    # (B, A, seqlen_hist_traffic, 2) -> (B, P, A, seqlen_hist_traffic, 2) 
            elif args.PNLL_method1 == 2: # Identical perturbation given P for all input groups. Each perturbation is identical for all batches, N_hist_refcar, max_num_actors, and N_hist_traffic.
                Noise = Gaussian2D.sample((args.num_perturbations,))                                                                                         # (P, 2)

                Noise_refcar_future = Noise.unsqueeze(0).unsqueeze(0).repeat(N_future, batch_size_loop, 1, 1)                                                # (P, 2) -> (T, B, P, 2)
                Noise_refcar_hist   = Noise.unsqueeze(0).unsqueeze(2).repeat(batch_size_loop, 1, N_hist_refcar, 1)                                           # (P, 2) -> (B, P,    seqlen_hist_refcar,  2)
                Noise_traffic_hist  = Noise.unsqueeze(0).unsqueeze(2).unsqueeze(3).repeat(batch_size_loop, 1, args.max_num_actors, N_hist_traffic, 1)        # (P, 2) -> (B, P, A, seqlen_hist_traffic, 2)  

                refcar_states_future_ptb = refcar_states_future.unsqueeze(2).repeat(1, 1, args.num_perturbations, 1)   + Noise_refcar_future                 # (T, B, 2)                       -> (T, B, P, 2)
                refcar_states_hist_ptb   = refcar_states_hist.unsqueeze(1).repeat(1, args.num_perturbations, 1, 1)     + Noise_refcar_hist/10                # (B,    seqlen_hist_refcar,  2)  -> (B, P,    seqlen_hist_refcar,  2)
                traffic_states_hist_ptb  = traffic_states_hist.unsqueeze(1).repeat(1, args.num_perturbations, 1, 1, 1) + Noise_traffic_hist/10               # (B, A, seqlen_hist_traffic, 2)  -> (B, P, A, seqlen_hist_traffic, 2)     
            elif args.PNLL_method1 == 99: # Independent perturbation
                Noise_refcar_future = Gaussian2D.sample((N_future, batch_size_loop, args.num_perturbations))                                                 # (T, B, P, 2)    
                Noise_refcar_hist   = Gaussian2D.sample((batch_size_loop, args.num_perturbations, N_hist_refcar))                                            # (B, P,    seqlen_hist_refcar,  2)
                Noise_traffic_hist  = Gaussian2D.sample((batch_size_loop, args.num_perturbations, args.max_num_actors, N_hist_traffic))                      # (B, P, A, seqlen_hist_traffic, 2) 

                refcar_states_future_ptb = refcar_states_future.unsqueeze(2).repeat(1, 1, args.num_perturbations, 1)   + Noise_refcar_future                 # (T, B, 2)                      -> (T, B, P, 2)
                refcar_states_hist_ptb   = refcar_states_hist.unsqueeze(1).repeat(1, args.num_perturbations, 1, 1)     + Noise_refcar_hist/10                # (B,    seqlen_hist_refcar,  2) -> (B, P,    seqlen_hist_refcar,  2)
                traffic_states_hist_ptb  = traffic_states_hist.unsqueeze(1).repeat(1, args.num_perturbations, 1, 1, 1) + Noise_traffic_hist/10               # (B, A, seqlen_hist_traffic, 2) -> (B, P, A, seqlen_hist_traffic, 2) 

        elif args.loss == 'PNLL_output':
            Noise_refcar_future = Gaussian2D.sample((N_future, 1, args.num_perturbations))                                                     
            Noise_refcar_future = Noise_refcar_future.repeat(1, batch_size_loop, 1, 1)                                                           # (T, 1, P, 2) -> (T, B, P, 2)
            
            refcar_states_future_ptb = refcar_states_future.unsqueeze(2).repeat(1, 1, args.num_perturbations, 1) + Noise_refcar_future           # (T, B, 2)    -> (T, B, P, 2)   
            refcar_states_hist_ptb = refcar_states_hist.unsqueeze(1).repeat(1, args.num_perturbations, 1, 1)                                     # (B,    seqlen_hist_refcar,  2) -> (B, P,    seqlen_hist_refcar,  2) 
            traffic_states_hist_ptb = traffic_states_hist.unsqueeze(1).repeat(1, args.num_perturbations, 1, 1, 1)                                # (B, A, seqlen_hist_traffic, 2) -> (B, P, A, seqlen_hist_traffic, 2)
        
        # Reshape tensors    
        refcar_states_future_int = refcar_states_future_ptb                                                                                      # (T, B, P, 2)
        refcar_states_hist_int   = refcar_states_hist_ptb.view(batch_size_loop*args.num_perturbations, N_hist_refcar*2)                          # (B, P, seqlen_hist_refcar,   2)     -> (B*P, seqlen_hist_refcar*2)                    
        traffic_states_hist_int  = traffic_states_hist_ptb.view(batch_size_loop*args.num_perturbations, args.max_num_actors*N_hist_traffic*2)    # (B, P, A, seqlen_hist_traffic, 2)   -> (B*P, A*seqlen_hist_traffic*2)
        inputs_imgs_int          = inputs_imgs_ptb.view(batch_size_loop*args.num_perturbations, args.dim_c_EncInput, args.HW_img, args.HW_img)   # (B, P, C, H, W)                     -> (B*P, C, H, W)
    else:
        refcar_states_future_int = refcar_states_future                                                                                          # (T, B, 2)
        refcar_states_hist_int = refcar_states_hist.view(batch_size_loop, N_hist_refcar*2)                                                       # (B, seqlen_hist, 2)                 -> (B, seqlen_hist*2)                    
        traffic_states_hist_int = traffic_states_hist.view(batch_size_loop, args.max_num_actors*N_hist_traffic*2)                                # (B, A, seqlen_hist, 2)              -> (B, A*seqlen_hist*2)
        inputs_imgs_int = inputs_imgs

        
    if args.temporal: 
        sec_into_future = (torch.tensor(args.forecast_which_frame, dtype=torch.float32)+1)/5 # frame 0: 0.2s, 9: 2.0s, 19: 4.0s
        if args.time_normalization:
            sec_into_future = sec_into_future/4 # (0, 4] -> (0, 1] 
        
        if args.loss in ['PNLL', 'PNLL_output']: 
            sec_into_future_int = sec_into_future.unsqueeze(-1).unsqueeze(-1).repeat(1, batch_size_loop, args.num_perturbations)                                                                    # T -> (T, 1) -> (T, B, P)
            # if args.temporal_training == 'p': use parallel computing for T dimension. 
            if args.temporal_training == 'p':  
                refcar_states_future_int_all = refcar_states_future_int.view(N_future*batch_size_loop*args.num_perturbations, 2)                                                                    # (T, B, P, 2) -> (T*B*P, 2)
                sec_into_future_int = sec_into_future_int.view(N_future*batch_size_loop*args.num_perturbations, 1)                                                                                  # (T, B, P) -> (T*B*P, 1)
                refcar_states_hist_int = refcar_states_hist_int.unsqueeze(0).repeat(N_future, 1, 1).view(N_future*batch_size_loop*args.num_perturbations, -1)                                       # (B*P, seqlen_hist_refcar*2)     -> (T*B*P, seqlen_hist_refcar*2)  
                traffic_states_hist_int = traffic_states_hist_int.unsqueeze(0).repeat(N_future, 1, 1).view(N_future*batch_size_loop*args.num_perturbations, -1)                                     # (B*P, A*seqlen_hist_traffic*2)  -> (T*B*P, A*seqlen_hist_traffic*2)
                inputs_imgs_int = inputs_imgs_int.unsqueeze(0).repeat(N_future, 1, 1, 1, 1).view(N_future*batch_size_loop*args.num_perturbations, args.dim_c_EncInput, args.HW_img, args.HW_img)    # (B*P, C, H, W)                  -> (T*B*P, C, H, W)
        else:
            sec_into_future_int = sec_into_future.unsqueeze(-1).repeat(1, batch_size_loop)                                                                                                          # T -> (T, 1) -> (T, B)
            if args.temporal_training == 'p':  
                refcar_states_future_int_all = refcar_states_future_int.view(N_future*batch_size_loop, 2)                                                                                           # (T, B, 2) -> (T*B, 2)
                sec_into_future_int = sec_into_future_int.view(N_future*batch_size_loop, 1)                                                                                                         # (T, B)   -> (T*B, 1)
                refcar_states_hist_int = refcar_states_hist_int.unsqueeze(0).repeat(N_future, 1, 1).view(N_future*batch_size_loop, -1)                                                              # (B, seqlen_hist_refcar*2)     -> (T*B, seqlen_hist_refcar*2)  
                traffic_states_hist_int = traffic_states_hist_int.unsqueeze(0).repeat(N_future, 1, 1).view(N_future*batch_size_loop, -1)                                                            # (B, A*seqlen_hist_traffic*2)  -> (T*B, A*seqlen_hist_traffic*2)
                inputs_imgs_int = inputs_imgs_int.unsqueeze(0).repeat(N_future, 1, 1, 1, 1).view(N_future*batch_size_loop, args.dim_c_EncInput, args.HW_img, args.HW_img)                           # (B, C, H, W)                  -> (T*B, C, H, W)
            
    # Formulating inputs to the model & Calculate loss and back-prop it
    if args.temporal and args.temporal_training == 'p':
        if args.ablation_mode in ['All', 'All_faster_temporal']:
            if args.temporal:
                inputs = [torch.cat((refcar_states_hist_int, traffic_states_hist_int, sec_into_future_int, refcar_states_future_int_all), -1), inputs_imgs_int]
            else:
                inputs = [torch.cat((refcar_states_hist_int, traffic_states_hist_int, refcar_states_future_int_all), -1), inputs_imgs_int]
        elif args.ablation_mode in ['No_lidar', 'No_lidar_faster_temporal']:
            if args.temporal:
                inputs = torch.cat((refcar_states_hist_int, traffic_states_hist_int, sec_into_future_int, refcar_states_future_int_all), -1)
            else:
                inputs = torch.cat((refcar_states_hist_int, traffic_states_hist_int, refcar_states_future_int_all), -1)

        for idx_inputs in range(len(inputs)):
            inputs[idx_inputs] = inputs[idx_inputs].to(args.device)
            
        if flag_eval:
            loss = - compute_log_p_x(model, inputs).view(N_future, -1).mean(-1) 
        else:
            with autograd.detect_anomaly():
                loss = - compute_log_p_x(model, inputs).view(N_future, -1).mean(-1)                 # (T*B*P, 1) -> (T, B*P) or (T*B, 1) -> (T, B)

                for idx, idx_time in enumerate(args.forecast_which_frame):
                    if idx < len(args.forecast_which_frame)-1:
                        flag_retain_graph = True
                    elif idx == len(args.forecast_which_frame)-1:
                        flag_retain_graph = False
                    loss[idx].backward(retain_graph=flag_retain_graph)

                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_gradnorm_max)

                    optimizer.step()
                    optimizer.zero_grad()

                    print('Epoch {}, Step {}, Time {}, Loss: {}'.format(args.epoch, args.iteration, (idx_time+1)/5, round(loss[idx].item(), 4)))
        
        args.loss_collector += loss.data.detach().cpu().numpy()
        loss_sum_batch = loss.data.sum().item() 

    else:
        loss_sum_batch = 0
        
        for idx, idx_time in enumerate(args.forecast_which_frame):
            if args.loss in ['PNLL', 'PNLL_output']: 
                refcar_states_future_int_t = refcar_states_future_int[idx, :, :, :].view(batch_size_loop*args.num_perturbations, 2)        # (T, B, P, 2) -> (B*P, 2)
                sec_into_future_int_t = sec_into_future_int[idx, :, :].view(batch_size_loop*args.num_perturbations, 1)                     # (T, B, P)    -> (B*P, 1)
            else:
                refcar_states_future_int_t = refcar_states_future_int[idx, :, :].view(batch_size_loop, 2)                                  # (T, B, 2)    -> (B, 2)
                sec_into_future_int_t = sec_into_future_int[idx, :].view(batch_size_loop, 1)                                               # (T, B)        > (B, 1)

            if args.ablation_mode in ['All', 'All_faster_temporal']:
                if args.temporal:
                    inputs = [torch.cat((refcar_states_hist_int, traffic_states_hist_int, sec_into_future_int_t, refcar_states_future_int_t), -1), inputs_imgs_int]
                else:
                    inputs = [torch.cat((refcar_states_hist_int, traffic_states_hist_int, refcar_states_future_int_t), -1), inputs_imgs_int]
            elif args.ablation_mode in ['No_lidar', 'No_lidar_faster_temporal']:
                if args.temporal:
                    inputs = torch.cat((refcar_states_hist_int, traffic_states_hist_int, sec_into_future_int_t, refcar_states_future_int_t), -1)
                else:
                    inputs = torch.cat((refcar_states_hist_int, traffic_states_hist_int, refcar_states_future_int_t), -1)
                inputs = inputs.to(args.device)

            for idx_inputs in range(len(inputs)):
                inputs[idx_inputs] = inputs[idx_inputs].to(args.device)

            # Calculate loss and back-prop it
            if flag_eval:
                if args.ablation_mode in ['All_faster_temporal', 'No_lidar_faster_temporal']:
                    loss = - compute_log_p_x_skip_HyperNet(model, inputs).mean()
                else:
                    loss = - compute_log_p_x(model, inputs).mean()
                loss_value = loss.item()
            else:
                with autograd.detect_anomaly():
                    if args.ablation_mode in ['All_faster_temporal', 'No_lidar_faster_temporal']:
                        loss = - compute_log_p_x_skip_HyperNet(model, inputs).mean()      # (B*P, 1) or (B, 1) -> 1
                        if idx < len(args.forecast_which_frame)-1:
                            flag_retain_graph = True
                        elif idx == len(args.forecast_which_frame)-1:
                            flag_retain_graph = False
                        loss.backward(retain_graph=flag_retain_graph)
                        loss_value = loss.item()
                        loss = None # remove unneeded parts of graph. Note that these parts will be freed from memory (even if they were on GPU), due to python's garbage collection
                    else:
                        loss = - compute_log_p_x(model, inputs).mean()                    # (B*P, 1) or (B, 1) -> 1     
                        loss.backward()
                        loss_value = loss.item()

                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_gradnorm_max)

                    optimizer.step()
                    optimizer.zero_grad()

                    print('Epoch {}, Step {}, Time {}, Loss: {}'.format(args.epoch, args.iteration, (idx_time+1)/5, round(loss_value, 4)))
            
            args.loss_collector[idx] += loss_value
            loss_sum_batch += loss_value

    return loss_sum_batch

#
def train_HCNAF_PRECOG_Carla(model, optimizer, scheduler, train_loader, val_loader, args):
    if args.dim_c_EncInput == 2:
        args.lidar_input_c = [1,2]
    elif args.dim_c_EncInput == 3:
        args.lidar_input_c = [1,2,3]

    Gaussian2D = MultivariateNormal(torch.zeros(2), 0.0001*torch.eye(2))

    if args.resume_training:
        last_epoch = args.last_epoch
        best_epoch = args.best_epoch
        val_loss_min = args.best_loss
        best_model_state = model.state_dict()
        best_optimizer_state = optimizer.state_dict()
    else:
        last_epoch = 0
        val_loss_min = float('inf')
        best_model_state = None
        best_optimizer_state = None

    # Main Training Loop
    for epoch in range(last_epoch+1, args.num_epoch):
        tr_loss_sum = 0 # sum of loss per epoch, keep in track (this is used to print per epoch loss)
        args.loss_collector = np.zeros(len(args.forecast_which_frame))
        train_loader_iter = iter(train_loader)

        model.train()
        test_time = time.time() 
        for iteration, loaded_data in enumerate(train_loader_iter):
            args.epoch = epoch
            args.iteration = iteration
            tr_loss_sum += collect_inputs_and_compute_loss_PRECOG(model, optimizer, loaded_data, Gaussian2D, False, args)

            if iteration == 50:
                print(time.time() - test_time)
            if iteration%5000 == 0:
                print('Saving progress, saving the current best model, at iteration {}'.format(iteration))
                save_name = 'epoch' + str(epoch)
                save_state_intermediate(iteration, model.state_dict(), os.path.join(args.path, save_name))
        
        for idx_eval, idx_time_eval in enumerate(args.forecast_which_frame):
            print('TRset, loss at time {}: {}'.format(idx_time_eval, args.loss_collector[idx_eval]/len(train_loader)))

        args.loss_collector = np.zeros(len(args.forecast_which_frame))
        # At the end of each epoch, obtain validation loss
        with torch.no_grad():
            val_loss_sum = 0
            val_loader_iter = iter(val_loader)
            
            model.eval() # switch to the eval mode (disable batch_norm & drop_out if there is any)
            for iteration, loaded_data in enumerate(val_loader_iter):
                args.epoch = epoch
                args.iteration = iteration
                val_loss_sum += collect_inputs_and_compute_loss_PRECOG(model, optimizer, loaded_data, Gaussian2D, True, args)
        
        for idx_eval, idx_time_eval in enumerate(args.forecast_which_frame):
            print('Valset, loss at time {}: {}'.format(idx_time_eval, args.loss_collector[idx_eval]/len(val_loader)))

        # Obtain avg.loss and avg.val_loss
        tr_loss_mean = tr_loss_sum/(len(args.forecast_which_frame)*len(train_loader)) # divided by the number of batches to get the mean values
        val_loss_mean = val_loss_sum/(len(args.forecast_which_frame)*len(val_loader))

        # Reduce the learning rates
        scheduler.step(val_loss_sum)

        # print & save loss & val_loss
        perf_eval = 'Epoch [{}/{}], Loss: {:.4f}, Val_Loss: {:.4f}'.format(
            epoch, args.num_epoch, tr_loss_mean, val_loss_mean)
        print(perf_eval)
        print(perf_eval, file=open(os.path.join(args.path, 'Perf_epoch.txt'), "a"))

        if val_loss_mean < val_loss_min:
            val_loss_min = val_loss_mean
            best_epoch = epoch
            best_model_state = model.state_dict()
            best_optimizer_state = optimizer.state_dict()

        if epoch%args.savemodel_period == 0:
            print('Saving progress, saving the current best model, at epoch {}'.format(best_epoch))
            save_name = 'epoch' + str(epoch)
            save_state(best_epoch, round(val_loss_min, 4), model.state_dict(), best_model_state, best_optimizer_state, scheduler.state_dict(), os.path.join(args.path, save_name))
    
    return best_epoch, best_model_state, best_optimizer_state


def main():
    parser = argparse.ArgumentParser(description = "Process Hyper-parameters of HCNAF training for PRECOG exp")

    parser.add_argument('--resume_training', type=bool, default=False)
    parser.add_argument('--loadpath', type=str, default='')
    parser.add_argument('--loadfilename', type=str, default='')
    parser.add_argument('--dataparallel', type=bool, default=False)

    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--dataset_name', type=str, default='PRECOG_Carla', choices=['PRECOG_Carla','PRECOG_Carla_tiny'])
    parser.add_argument('--path_json', type=str, default='data/precog_carla/')   
    parser.add_argument('--ablation_mode', type=str, default='All_faster_temporal', 
        choices=['All', 'All_faster_temporal', 'No_lidar', 'No_lidar_faster_temporal'],
        help='All: All information, All_faster_temporal: for temporal=1, skip repeated computations, No_lidar: No lidar is used, No_lidar_faster_temporal: skip repeated computation for No_lidar')

    parser.add_argument('--normalization', type=int, default=1)
    parser.add_argument('--sdv_hist_length_sec', type=int, default=1)      # 3: 3 secs of sdv history is used.
    parser.add_argument('--sdv_hist_tgap', type=float, default=0.2)        # if sdv_hist_length_sec = 3, time_gap=0.5, then num_seq_input_sdv = 6
    parser.add_argument('--loss', type=str, default = 'PNLL_output', choices=['NLL', 'PNLL', 'PNLL_output']) # negative log-likelihood (NLL), perturbed NLL (PNLL) where all localization inputs are perturbed. PNLL_output: only the outputs are perturbed.

    parser.add_argument('--temporal', type=int, default = 1) # include time as part of inputs. This is independent from ablation_mode
    #parser.add_argument('--forecast_which_frame', type=list, default=[19])  # PRECOG_Carla_dataset: 10 past frames (2s), 20 future frames (4s).
    #parser.add_argument('--forecast_which_frame', type=list, default=[9,19])  
    parser.add_argument('--forecast_which_frame', type=list, default=list(range(20)) )
    parser.add_argument('--time_normalization', type=int, default=1)

    parser.add_argument('--n_layers_flow', type=int, default=3)
    parser.add_argument('--dim_h_flow', type=int, default=100)
    parser.add_argument('--norm_HW', type=str, default='scaled_frobenius', choices=['scaled_frobenius', 'modified_weightnorm'])

    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--batch_size_test', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=12)
    parser.add_argument('--clip_gradnorm_max', type=float, default=.1)
    parser.add_argument('--num_epoch', type=int, default=30)
    parser.add_argument('--decay', type=float, default=0.4)

    parser.add_argument('--savemodel_period', type=int, default=1)

    args_new = parser.parse_args()
    ####################################################################################

    if args_new.resume_training: # Resume training
        print('Resuming a previous training')
        best_epoch, best_loss, model_state, best_model_state, optimizer_state, scheduler_state = load_state(os.path.join(args_new.loadpath, args_new.loadfilename))
        #_, best_model_state = load_state_intermediate(os.path.join(args_new.loadpath, args_new.loadfilename))
        
        # resume_training args
        args = json.load(open(os.path.join(args_new.loadpath, "args.json"),"r"))
        args = SimpleNamespace(**args) # convert dictionary to namespace

        print("Arguments are loaded from args.json except the following parameters: device, dataparallel, num_epoch")
        if args_new.resume_training:
            args.device = args_new.device
            args.num_epoch = args_new.num_epoch
            args.dataparallel = args_new.dataparallel
            args.resume_training = args_new.resume_training # True
    else: # start a new training
        print('Starting a new training') 
        args = args_new

        args.loadpath = 'N/A'
        args.loadfilename = 'N/A'
        if args.norm_HW == 'scaled_frobenius':
            norm_HW_short = 'sf'
        elif args.norm_HW == 'modified_weightnorm':
            norm_HW_short = 'mw'

        # Set hyper-parameters that are used to create a HCNAF model
        args.patience = 1 # patience is based on val_loss. If patience = 1, then the learning rate reduces the second time val_loss is not improved.
        args.max_num_actors = 4                             # by default
        
        args.n_layers_RNN = 2
        args.num_seq_input_sdv = int(round(args.sdv_hist_length_sec/args.sdv_hist_tgap))
        
        if args.loss in ['PNLL', 'PNLL_output']:
            args.num_perturbations = 10
        if args.ablation_mode in ['All', 'All_faster_temporal', 'No_lidar', 'No_lidar_faster_temporal']:
            args.traffic_hist_length_sec = 1
            args.traffic_hist_tgap = 0.2
            args.dim_h_traffic = 20
            args.dim_per_traffic = args.num_seq_input_sdv
        else:
            args.dim_traffic = args.max_num_actors * 2      # dim_traffic = 8
            args.dim_h_traffic = args.dim_traffic * 2
        
        args.len_hist = 10 
        args.len_future = 20
        args.len_seq = 30
        args.dim_h_RNN = 24
        
        args.HW_img = 200
        args.pix_per_m = 2
        args.frame_no_curr = 9

        args.dim_h_Enc = 96
        args.dim_c_EncInput = 2 # 2: use 2 channels (1,2), 3: use 3 channels (1,2,3)
        args.Enc_Model = 'coordconv_simple' # 'conv_simple', 'conv_complex', 'coordconv_simple', 'coordconv_complex'
        args.Enc_norm_method = 'BN' # 'BN' (default), 'GN', 'GN2', 'IN', 'LN'
        args.Pool_method = 'mp'

        if args.ablation_mode in ['No_lidar', 'No_lidar_faster_temporal']:
            args.dim_h_Enc = 'N/A'
            args.Enc_Model = 'N/A'
            args.Enc_norm_method = 'N/A'
            args.Pool_method = 'N/A'

        args.img_center = args.HW_img//2
        if args.temporal:
            args.dim_h_dt = 10
            args.temporal_training = 'NA' # 'p':parallel, 'NA'
            #args.temporal_training = 'p' 
        else:
            args.dim_h_dt = None 
            args.temporal_training = 'NA'    

        if args.loss == 'PNLL':
            args.PNLL_method1 = 2
        ########################################################################################
        
        if not os.path.exists("results"):
            os.mkdir("results")

        args.path = os.path.join('results', '{}_AMode{}_temporal{}_normHW{}_hdt{}_loss{}_layers{}_h{}'.format(
            args.dataset_name, args.ablation_mode, args.temporal, norm_HW_short, args.dim_h_dt, args.loss, args.n_layers_flow, args.dim_h_flow))

        if args.ablation_mode in ['All', 'All_faster_temporal']: # encoder info is added to path
            args.path += '_Enc{}_{}_E{}_c{}_PL{}'.format(args.Enc_Model, args.Enc_norm_method, args.dim_h_Enc, args.dim_c_EncInput, args.Pool_method)

        args.path += '_layersRNN{}_hRNN{}_sdvhistsec{}_tgap{}_B{}_{}'.format(args.n_layers_RNN, args.dim_h_RNN, args.sdv_hist_length_sec, args.sdv_hist_tgap, args.batch_size, str(datetime.datetime.now())[:-7].replace(' ', '-').replace(':', '-'))

        os.mkdir(args.path)
        with open(os.path.join(args.path, 'args.json'), 'w') as f:
            json.dump(args.__dict__, f, indent=4, sort_keys=True)

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
    train_loader, val_loader, test_loader = set_dataloader_PRECOG_Carla(dataset_tr, dataset_val, dataset_test, args.batch_size, args.batch_size_test, args.num_workers)

    # Check the dimensions of data
    refcar_states_hist, refcar_states_future, refcar_transform, traffic_states_hist, traffic_states_future, traffic_transform, lidar_overhead_f, params, identifier = next(iter(train_loader))
    print('Dim: refcar_hist, refcar_future, traffic_hist, traffic_future, lidar_overhead_features')
    print(refcar_states_hist.shape)                # (B, seq_len_hist, 2)
    print(refcar_states_future.shape)              # (B, seq_len_future, 2)
    print(traffic_states_hist.shape)               # (B, N, seq_len_hist, 2)
    print(traffic_states_future.shape)             # (B, N, seq_len_future, 2)
    print(lidar_overhead_f.shape)                  # (B, H, W, C)

    print('Dim: States_sdv, States_traffic, States_map_emb, States_SS, States_sdv_outputs')
    print(refcar_transform.shape)                  # (B, 4, 4) [history of (x,y,sin(theta),cos(theta),spd)), width & length of sdv]
    print(traffic_transform.shape)                 # (B, N, 4, 4) [(is_exist_actor,x,y,sin(theta),cos(theta),spd)] at the current timestampe (50th frame) 

    args.dim_sdv_hist = refcar_states_hist.shape[1] * refcar_states_hist.shape[2] # seq_len_hist * 2
    args.dim_traffic_hist = traffic_states_hist.shape[2] * traffic_states_hist.shape[3] # seq_len_hist * 2
    args.dim_o = 2

    # Create a HCNAF model, an optimizer, and a scheduler
    print('Creating model, optimizer, and scheduler')
    model = create_HCNAF(args)
    if args.dataparallel:
        model = torch.nn.DataParallel(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=args.decay, patience=args.patience, min_lr=1e-7, verbose=True, threshold_mode='abs')

    # Resume training or start new training
    if args.resume_training == True: # Resume training
        print('Loading previously trained models')
        #model.load_state_dict(model_state)
        model.load_state_dict(best_model_state)
        #optimizer.load_state_dict(optimizer_state)
        #scheduler.load_state_dict(scheduler_state)
        args.best_epoch = best_epoch
        args.best_loss = best_loss
        args.last_epoch = scheduler_state['last_epoch']

    print('Starting the training process')
    best_epoch, best_model_state, best_optimizer_state = train_HCNAF_PRECOG_Carla(model, optimizer, scheduler, train_loader, val_loader, args)

if __name__ == '__main__':
    main()
