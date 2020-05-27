"""
# Author      : GS Oh
# Experiment  : PRECOG_Carla
# Note        : Utility functions used for visualizing probabilistic occupancy maps
"""
import matplotlib
import matplotlib.pyplot as plt

import torch
import numpy as np
import time, os

from hcnaf_PRECOG_Carla import compute_log_p_x, compute_log_p_x_skip_HyperNet

# The colors (PRECOG_Plot)
COLORS = """#377eb8
#ff7f00
#4daf4a
#984ea3
#ffd54f""".split('\n')


def plot_POMap_PRECOG_Carla(model, args, data_loader, args_p):
    img_center = args.HW_img//2
    Num_batch_plot = args.HW_img*args.HW_img//args_p.b_size_plot
    if args.dim_c_EncInput == 2:
        lidar_input_c = [1,2]
    elif args.dim_c_EncInput == 3:
        lidar_input_c = [1,2,3]

    if args.normalization:
        lb, ub, step = -1, 1, args_p.res/img_center
        n_bins_x_plot = int((ub-lb)//step)+1
        n_bins_y_plot = int((ub-lb)//step)+1
    else:
        lb, ub, step = 0, args.HW_img, args_p.res

    model.eval()    
    with torch.no_grad():
        plot_data_loader_iter = iter(data_loader)
        for plot_i in range(args_p.idx_last_seq_toplot):
            refcar_states_hist, refcar_states_future, refcar_transform, traffic_states_hist, traffic_states_future, traffic_transform, lidar_overhead_f, params, identifier = next(plot_data_loader_iter) # traffic_states_future is not used in training. Only for plotting.
            if plot_i not in args_p.plot_index:
                continue

            lidar_overhead_f = lidar_overhead_f.permute(0,3,1,2)
            inputs_imgs = lidar_overhead_f[:, lidar_input_c, :, :]
            batch_size_loop, N_hist_refcar, N_hist_traffic, N_future = refcar_states_hist.size(0), refcar_states_hist.size(1), traffic_states_hist.size(2), refcar_states_future.size(1)

            refcar_states_future_int = refcar_states_future.transpose(0, 1)                                                   # (B, T, 2)              -> (T, B, 2)
            refcar_states_hist_int = refcar_states_hist.view(batch_size_loop, N_hist_refcar*2)                                # (B, seqlen_hist, 2)    -> (B, seqlen_hist*2)                    
            traffic_states_hist_int = traffic_states_hist.view(batch_size_loop, args.max_num_actors*N_hist_traffic*2)         # (B, A, seqlen_hist, 2) -> (B, A*seqlen_hist*2)
            inputs_imgs_int = inputs_imgs                                                                                     # (B, C, H, W)

            if args.temporal:
                sec_into_future = (torch.tensor(args.forecast_which_frame, dtype=torch.float32)+1)/5                          # frame 0: 0.2s, 9: 2.0s, 19: 4.0s
                if args.time_normalization:
                    sec_into_future = sec_into_future/4                                                                       # (0, 4] -> (0, 1] 
                
                sec_into_future_int = sec_into_future.unsqueeze(-1).repeat(1, batch_size_loop)                                # T -> (T, 1) -> (T, B)  

            for plot_j in range(batch_size_loop):
                # Plot Future-groundtruth and POM forecasts
                for idx, idx_time in enumerate(args.forecast_which_frame):
                    # reset 'is_WB_computed'
                    model[0].is_WB_computed = False
                    start_time = time.time()
                    if args.ablation_mode in ['All', 'All_faster_temporal']:
                        condition_Map = inputs_imgs_int[plot_j,:,:,:].to(args_p.device)
                        if args.temporal:
                            condition = torch.cat((refcar_states_hist_int[plot_j,:], traffic_states_hist_int[plot_j,:], sec_into_future_int[idx, plot_j].view(1)), -1).to(args_p.device)
                        else:
                            condition = torch.cat((refcar_states_hist_int[plot_j,:], traffic_states_hist_int[plot_j,:]), -1).to(args_p.device)
                        inputs = [torch.cat((condition, refcar_states_future_int[idx, plot_j, :].to(args_p.device)), -1).unsqueeze(0), condition_Map.unsqueeze(0)]
                    elif args.ablation_mode in ['No_lidar', 'No_lidar_faster_temporal']:
                        if args.temporal:
                            condition = torch.cat((refcar_states_hist_int[plot_j,:], traffic_states_hist_int[plot_j,:], sec_into_future_int[idx, plot_j].view(1)), -1).to(args_p.device)
                        else:
                            condition = torch.cat((refcar_states_hist_int[plot_j,:], traffic_states_hist_int[plot_j,:]), -1).to(args_p.device)
                        inputs = torch.cat((condition, refcar_states_future_int[idx, plot_j, :].to(args_p.device)), -1).unsqueeze(0)
                    print("--- Pre-processing takes {0:.6f} seconds ---".format(time.time() - start_time))

                    start_time = time.time()
                    # Compute NLL for the ground truth
                    NLL = round(- compute_log_p_x(model, inputs).item(),4)
                    print("--- Computing NLL takes {0:.6f} seconds ---".format(time.time() - start_time))

                    start_time = time.time()
                    # Create inputs of size (args.HW_img * args.HW_Img) to obtain probablities over the map of interest 
                    grid = condition.unsqueeze(0).repeat(n_bins_x_plot*n_bins_y_plot, 1) 
                    map_xy = torch.Tensor([[b, a] for a in np.arange(lb, ub, step) for b in np.arange(lb, ub, step)]).to(args_p.device)
                    grid = torch.cat((grid, map_xy), -1)
                    print("--- Grid definition takes {0:.6f} seconds ---".format(time.time() - start_time)) # There should still be better way of defining the grid
                    grid_dataset = torch.utils.data.TensorDataset(grid.to(args_p.device))
                    grid_data_loader = torch.utils.data.DataLoader(grid_dataset, batch_size=args_p.b_size_plot, shuffle=False)
                    print("--- Grid Formulation takes {0:.6f} seconds ---".format(time.time() - start_time))
                
                    start_time = time.time()
                
                    prob_list = []
                    for idx_b, data in enumerate(grid_data_loader):
                        data = data[0].to(args_p.device)
                        plotarea_control = 0
                        if plotarea_control: ## To speed the calculation, assign zero probabilities outside the box of interest.
                            idx_assign_zero = ((data[:,-2] < -0.2) | (data[:,-2] > 1.0) | (data[:,-1] < -1.0) | (data[:,-1] > 1.0)).float()
                            list_idx_nonzero = np.where(idx_assign_zero.cpu().numpy() == 0)[0]
                            data_interest = data[list_idx_nonzero, :]
                            prob_list_batch = torch.zeros(data.size(0)).to(args_p.device)
                            if len(data_interest) > 0:
                                if args.ablation_mode in ['No_lidar', 'No_lidar_faster_temporal']:
                                    inputs = data_interest
                                else:
                                    inputs = [data_interest, condition_Map.repeat(data_interest.size(0), 1, 1, 1)]

                                if idx_b == Num_batch_plot:
                                    prob_list_batch[list_idx_nonzero] = torch.exp(compute_log_p_x(model, inputs)).detach()
                                else:            
                                    prob_list_batch[list_idx_nonzero] = torch.exp(compute_log_p_x_skip_HyperNet(model, inputs)).detach()
                            prob_list.append(prob_list_batch)
                        else: ## Calculate probabilities of all area
                            data_interest = data
                            if args.ablation_mode in ['No_lidar', 'No_lidar_faster_temporal']:
                                inputs = data_interest
                            else:
                                inputs = [data_interest, condition_Map.repeat(data_interest.size(0), 1, 1, 1)]

                            if idx_b == Num_batch_plot:
                                prob_list.append(torch.exp(compute_log_p_x(model, inputs)).detach())
                            else:            
                                prob_list.append(torch.exp(compute_log_p_x_skip_HyperNet(model, inputs)).detach())
                    
                    prob = torch.cat(prob_list[:])   
                    if args_p.plot_method == 'Log': # log(p)/10 thresholded at 0.5
                        prob = torch.log(prob + 0.1) - np.log(0.1)
                        prob = (prob/10.0).view(args.HW_img, args.HW_img)
                        prob = torch.where(prob >= 0.5, prob, torch.tensor(0.0).to(args_p.device))
                    elif args_p.plot_method == 'Max':
                        prob = (prob/prob.max()).view(args.HW_img, args.HW_img)
                    print("--- Occupancy map is obtained in {0:.6f} seconds ---".format(time.time() - start_time))

                    start_time = time.time()
                    predicted_ref_car_future = prob.cpu().data
                    if args.normalization: # If the data_loader returned normalized tensors, obtain the original tensors back from normalized tensors, in the shape that scatter plot takes in. (frame_no, 2)
                        refcar_states_future_plot   = refcar_states_future[plot_j, :, :].view(N_future, 2) * 50      # (B, T, 2). Don't get confused with refcar_states_future_int
                        traffic_states_future_plot  = traffic_states_future[plot_j, :, :, :].view(args.max_num_actors, N_future, 2) * 50   # (B, A, seqlen_future, 2)
                        refcar_states_hist_plot     = refcar_states_hist[plot_j,:,:].view(N_hist_refcar, 2) * 50
                        traffic_states_hist_plot    = traffic_states_hist[plot_j,:,:,:].view(args.max_num_actors, len(data_loader.dataset.list_idx_traffic_hist), 2) * 50
                    
                    # xlim, ylim for the plots
                    bev_side_pixels = img_center
                    bev_meters = bev_side_pixels / args.pix_per_m

                    # Plot History (t=-2~0)
                    if idx == 0:
                        fig = plt.figure(figsize=(10, 10))
                        plt.scatter(refcar_states_hist_plot[:,0], refcar_states_hist_plot[:,1], label='Car 1 past', marker='d', facecolor='None', edgecolor=COLORS[0])
                        for other_idx in range(args.max_num_actors):
                            flabel = 'Car {} future'.format(other_idx + 2)
                            plabel = 'Car {} past'.format(other_idx + 2)
                            plt.scatter(traffic_states_future_plot[other_idx,idx,0], traffic_states_future_plot[other_idx,idx,1], label=flabel, facecolor='None', edgecolor=COLORS[other_idx + 1], marker='s')
                            plt.scatter(traffic_states_hist_plot[other_idx,:,0], traffic_states_hist_plot[other_idx,:,1], label=plabel, facecolor='None', edgecolor=COLORS[other_idx + 1], marker='d')
                            
                        plt.imshow(lidar_overhead_f[plot_j, 2, :, :], extent=(-bev_meters, bev_meters, bev_meters, -bev_meters), cmap='Greys', alpha=1.0)
                        plt.imshow(lidar_overhead_f[plot_j, 1, :, :], extent=(-bev_meters, bev_meters, bev_meters, -bev_meters), cmap='Reds', alpha=0.8)
                        
                        plt.title("Lidar + Ground-truth Trajectory")
                        plt.xlim([-bev_meters, bev_meters])
                        plt.ylim([bev_meters, -bev_meters])
                        plt.subplots_adjust(right=0.8)
                        plt.legend(bbox_to_anchor=(1., 1.), loc="upper left", fontsize=14)
                        cbar_ax = fig.add_axes([0.85, 0.10, 0.05, 0.50])
                        plt.colorbar(cax=cbar_ax)

                        data_id = identifier[0].replace('.json','').replace('/','_')
                        plt.savefig(args_p.loadpath + '{}/{}_{}_RGB_Groundtruth_history_{}.jpg'.format(args_p.save_folder_POMap, plot_i, plot_j, data_id))
                        plt.close()
                        
                    # Plot Ground-truth
                    fig = plt.figure(figsize=(10, 10))
                    plt.scatter(refcar_states_future_plot[idx,0], refcar_states_future_plot[idx,1], label='Car 1 future', marker='s', facecolor='None', edgecolor=COLORS[0])
                    plt.scatter(refcar_states_hist_plot[:,0], refcar_states_hist_plot[:,1], label='Car 1 past', marker='d', facecolor='None', edgecolor=COLORS[0])
                    for other_idx in range(args.max_num_actors):
                        flabel = 'Car {} future'.format(other_idx + 2)
                        plabel = 'Car {} past'.format(other_idx + 2)
                        plt.scatter(traffic_states_future_plot[other_idx,idx,0], traffic_states_future_plot[other_idx,idx,1], label=flabel, facecolor='None', edgecolor=COLORS[other_idx + 1], marker='s')
                        plt.scatter(traffic_states_hist_plot[other_idx,:,0], traffic_states_hist_plot[other_idx,:,1], label=plabel, facecolor='None', edgecolor=COLORS[other_idx + 1], marker='d')
                    #plt.imshow(test2, extent=(-bev_meters, bev_meters, bev_meters, -bev_meters), cmap='RdGy')
                    plt.imshow(lidar_overhead_f[plot_j, 2, :, :], extent=(-bev_meters, bev_meters, bev_meters, -bev_meters), cmap='Greys', alpha=1.0)
                    plt.imshow(lidar_overhead_f[plot_j, 1, :, :], extent=(-bev_meters, bev_meters, bev_meters, -bev_meters), cmap='Reds', alpha=0.8)
                    
                    plt.title("Lidar + Ground-truth Trajectory")
                    plt.xlim([-bev_meters, bev_meters])
                    plt.ylim([bev_meters, -bev_meters])
                    plt.subplots_adjust(right=0.8)
                    plt.legend(bbox_to_anchor=(1., 1.), loc="upper left", fontsize=14)
                    cbar_ax = fig.add_axes([0.85, 0.10, 0.05, 0.50])
                    plt.colorbar(cax=cbar_ax)

                    data_id = identifier[0].replace('.json','').replace('/','_')
                    plt.savefig(args_p.loadpath + '{}/{}_{}_{}sec_RGB_Groundtruth_future_{}.jpg'.format(args_p.save_folder_POMap, plot_i, plot_j, (idx_time+1)/5, data_id))
                    plt.close()

                    # Plot POMap
                    fig = plt.figure(figsize=(10, 10))
                    plt.scatter(refcar_states_future_plot[idx,0], refcar_states_future_plot[idx,1], label=flabel, marker='s', facecolor='None', edgecolor=COLORS[0])
                    plt.imshow(lidar_overhead_f[plot_j, 2, :, :], extent=(-bev_meters, bev_meters, bev_meters, -bev_meters), cmap='Greys', alpha=1.0)
                    plt.imshow(lidar_overhead_f[plot_j, 1, :, :], extent=(-bev_meters, bev_meters, bev_meters, -bev_meters), cmap='Reds', alpha=0.8)
                    plt.imshow(predicted_ref_car_future, extent=(-bev_meters, bev_meters, bev_meters, -bev_meters), interpolation='kaiser', cmap='Reds', alpha=0.8)
                    
                    
                    plt.subplots_adjust(right=0.8)
                    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
                    plt.colorbar(cax=cbar_ax)
                    plt.title("Lidar + Prediction")

                    plt.savefig(os.path.join(args_p.loadpath, '{}/{}_{}_{}sec_RGB_Prediction_future_AMode{}_hist1sec_tgap0.2_NLL{}.jpg'.format(args_p.save_folder_POMap, plot_i, plot_j, (idx_time+1)/5, args.ablation_mode, NLL)))
                    plt.close()

                    print("--- Drawing a POM takes {0:.6f} seconds ---".format(time.time() - start_time))
