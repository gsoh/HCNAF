"""
# Author      : GS Oh
# Experiment  : The gaussian experiments presented in the HCNAF paper
# Note        : Utility functions for HCNAF for the gaussian experiments
"""
import torch
import matplotlib.pyplot as plt
import numpy as np
from hcnaf_gaussians import *


def save_state(save_dict):
    print('Saving model_state_dict, optimizer_state_dict, scheduler_state_dict..')
    print('best iteration: {}, loss: {}'.format(save_dict['best_iteration'], save_dict['best_loss']))
    torch.save({
            'last_iteration': save_dict['last_iteration'],
            'best_iteration': save_dict['best_iteration'],
            'best_loss': save_dict['best_loss'],
            'best_model_state_dict': save_dict['best_model_state_dict'],
            'model_state_dict': save_dict['model_state_dict'],
            'optimizer_state_dict': save_dict['optimizer_state_dict'],
            'scheduler_state_dict': save_dict['scheduler_state_dict']
        }, save_dict['save_path'] + '_Bestiteration' + str(save_dict['best_iteration']) + '_BestLoss' + str(save_dict['best_loss']) + '.pt')     


def load_state(path):
    print('Loading model_state_dict, optimizer_state_dict, scheduler_state_dict..')
    save_dict = torch.load(path)
    last_iteration = save_dict['last_iteration']
    best_iteration = save_dict['best_iteration']
    best_loss = save_dict['best_loss']
    best_model_state = save_dict['best_model_state_dict']
    model_state = save_dict['model_state_dict']
    optimizer_state = save_dict['optimizer_state_dict']
    scheduler_state = save_dict['scheduler_state_dict']

    return model_state, optimizer_state, scheduler_state, best_iteration, best_loss, best_model_state, last_iteration


def plot_de_2d(model, args, condition, step=0.05, batch_size_plot=10):
    if args.dataset == 'gaussians_exp2':
        lim_x_l, lim_y_l, lim_x_u, lim_y_u = 1, 1, 15, 15
    else:
        lim_x_l, lim_y_l, lim_x_u, lim_y_u = -9, -9, 9, 9

    if isinstance(condition, list):
        grid = torch.Tensor([[*condition, a, b] for b in np.arange(lim_x_l, lim_x_u, step) for a in np.arange(lim_y_l, lim_y_u, step)])
    else:
        grid = torch.Tensor([[condition, a, b] for b in np.arange(lim_x_l, lim_x_u, step) for a in np.arange(lim_y_l, lim_y_u, step)])        
    grid_dataset = torch.utils.data.TensorDataset(grid.to(args.device))
    grid_data_loader = torch.utils.data.DataLoader(grid_dataset, batch_size=batch_size_plot, shuffle=False)

    prob = torch.cat([torch.exp(compute_log_p_x(model, data)).detach() for data, in grid_data_loader], 0)
    prob = prob.view(int((lim_x_u - lim_x_l) / step), int((lim_y_u - lim_y_l) / step))

    plt.figure()
    plt.imshow(prob.cpu().data.numpy(), extent=(lim_x_l, lim_x_u, lim_y_l, lim_y_u))
    plt.axis('off')
    plt.savefig(args.path + args.loadfilename + '_condition{}.jpg'.format(condition))

   

def plot_de_2d_multiple(model, args, conditions_name, step=0.05, batch_size_plot=10):
    if args.dataset == 'gaussians_exp2':
        lim_x_l, lim_y_l, lim_x_u, lim_y_u = 1, 1, 15, 15
        if conditions_name == 'seen':
            conditions_list = [[4, 4], [4, 12], [8, 8], [12, 4], [12, 12]]
        elif conditions_name == 'unseen':
            conditions_list = [[4, 8], [12, 8], [8, 4], [8, 12]]
    else:
        lim_x_l, lim_y_l, lim_x_u, lim_y_u = -9, -9, 9, 9

    num_pixels = int((lim_x_u - lim_x_l) / step) * int((lim_y_u - lim_y_l) / step) 
    prob = torch.zeros( num_pixels ).to(args.device)
    for condition in conditions_list:
        if isinstance(condition, list):
            grid = torch.Tensor([[*condition, a, b] for b in np.arange(lim_x_l, lim_x_u, step) for a in np.arange(lim_y_l, lim_y_u, step)])
        else:
            grid = torch.Tensor([[condition, a, b] for b in np.arange(lim_x_l, lim_x_u, step) for a in np.arange(lim_y_l, lim_y_u, step)])        
        grid_dataset = torch.utils.data.TensorDataset(grid.to(args.device))
        grid_data_loader = torch.utils.data.DataLoader(grid_dataset, batch_size=batch_size_plot, shuffle=False)

        prob += torch.cat([torch.exp(compute_log_p_x(model, data)).detach() for data, in grid_data_loader], 0)
    prob = prob.view(int((lim_x_u - lim_x_l) / step), int((lim_y_u - lim_y_l) / step))

    plt.figure()
    plt.imshow(prob.cpu().data.numpy(), extent=(lim_x_l, lim_x_u, lim_y_l, lim_y_u))
    plt.axis('off')
    plt.savefig(args.path + args.loadfilename + '_condition_{}.jpg'.format(conditions_name))



def plot_de_2d_data(model, args, conditions_list, step=0.05, batch_size_plot=10):
    if args.dataset == 'gaussians_exp2':
        lim_x_l, lim_y_l, lim_x_u, lim_y_u = 1, 1, 15, 15
    else:
        lim_x_l, lim_y_l, lim_x_u, lim_y_u = -9, -9, 9, 9

    num_pixels = int((lim_x_u - lim_x_l) / step) * int((lim_y_u - lim_y_l) / step) 
    prob = torch.zeros( num_pixels ).to(args.device)
    grid = torch.Tensor([[a, b] for b in np.arange(lim_x_l, lim_x_u, step) for a in np.arange(lim_y_l, lim_y_u, step)])
    grid_dataset = torch.utils.data.TensorDataset(grid.to(args.device))
    grid_data_loader = torch.utils.data.DataLoader(grid_dataset, batch_size=batch_size_plot, shuffle=False)

    prob += torch.cat([torch.exp(compute_log_p_x(model, data)).detach() for data, in grid_data_loader], 0)
    prob = prob.view(int((lim_x_u - lim_x_l) / step), int((lim_y_u - lim_y_l) / step))

    plt.figure()
    plt.imshow(prob.cpu().data.numpy(), extent=(lim_x_l, lim_x_u, lim_y_l, lim_y_u))
    plt.axis('off')
    plt.savefig(args.path + args.loadfilename + '_targetdist_condition{}.jpg'.format(condition))
 
