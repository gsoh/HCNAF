"""
# Author      : GS Oh
# Experiment  : PRECOG_Carla
# Note        : Utility functions used to save & load & etc.
"""
# Import pytorch libraries
import torch
from torch.distributions import MultivariateNormal

# Import ETC
import numpy as np


def load_state(path):
    print('Loading model_state_dict, optimizer_state_dict..')
    state_dict = torch.load(path, map_location='cpu')

    best_epoch = state_dict['best_epoch']
    best_loss = state_dict['best_loss']
    model_state = state_dict['model_state_dict']
    best_model_state = state_dict['best_model_state_dict']
    optimizer_state = state_dict['optimizer_state_dict']
    scheduler_state = state_dict['scheduler_state_dict']

    return best_epoch, best_loss, model_state, best_model_state, optimizer_state, scheduler_state
    

def save_state(best_epoch, best_loss, model_state, best_model_state, optimizer_state, scheduler_state, path):
    print('Saving model_state_dict & optimizer_state_dict..')
    print('best epoch: {}, loss: {}'.format(best_epoch, best_loss))
    torch.save({
            'best_epoch': best_epoch,
            'best_loss': best_loss,
            'model_state_dict': model_state,
            'best_model_state_dict': best_model_state,
            'optimizer_state_dict': optimizer_state,
            'scheduler_state_dict': scheduler_state
        }, path + '_BestEpoch' + str(best_epoch) + '_BestLoss' + str(best_loss) + '.pt')     


def save_state_intermediate(iteration, model_state, path):
    print('Saving intermediate model_state_dict ...')
    print('Iteration: {}'.format(iteration))
    torch.save({
            'iteration': iteration,
            'model_state_dict': model_state,
        }, path + '_Iteration' + str(iteration) + '.pt')     


def load_state_intermediate(path):
    print('Loading intermediate model_state_dict ...')
    state_dict = torch.load(path, map_location='cpu')
    iteration = state_dict['iteration']
    model_state = state_dict['model_state_dict']  

    return iteration, model_state





