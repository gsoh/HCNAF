"""
# Author      : GS Oh
# Experiment  : PRECOG_Carla
# Note        : Define HCNAF (Hyper-conditioned neural autoregressive flow) customized to PRECOG_Carla exp
"""

import numpy as np
import torch
import math
import torch.nn.utils.spectral_norm as sn_torch
from encoder import *


# Computing Loss function
def compute_log_p_x(model, data):
    model[0].skip_HyperNet = False
    z, log_det_j, HyperParam = model(data)
    log_p_z = torch.distributions.Normal(torch.zeros_like(z), torch.ones_like(z)).log_prob(z).sum(-1)
    return log_p_z + log_det_j

# Computing Loss function (skip hypernet related computations)
def compute_log_p_x_skip_HyperNet(model, data):
    model[0].skip_HyperNet = True
    z, log_det_j, HyperParam = model(data)
    log_p_z = torch.distributions.Normal(torch.zeros_like(z), torch.ones_like(z)).log_prob(z).sum(-1)
    return log_p_z + log_det_j


# Create HCNAF model
def create_HCNAF(args):
    """
        Create a HCNAF model; (1) Hyper-network (2) Conditional Autoregressive Flow
    """

    # Define a hyper-network
    if args.dataset_name in ['PRECOG_Carla', 'PRECOG_Carla_tiny']:
        if args.ablation_mode == 'All':
            HyperLayer = HyperNN_PRECOG_Carla_All(args)
        elif args.ablation_mode == 'All_faster_temporal':
            HyperLayer = HyperNN_PRECOG_Carla_All_faster_temporal(args)
        elif args.ablation_mode == 'No_lidar':
            HyperLayer = HyperNN_PRECOG_Carla_NoLidar(args)
            print('yes')
        elif args.ablation_mode == 'No_lidar_faster_temporal':
            HyperLayer = HyperNN_PRECOG_Carla_NoLidar_faster_temporal(args)

    # Define a conditional AF
    intermediate_layers_cAFs = []
    for _ in range(args.n_layers_flow - 1):
        intermediate_layers_cAFs.append(MaskedWeight(args.dim_o * args.dim_h_flow, args.dim_o * args.dim_h_flow, dim=args.dim_o, norm_w=args.norm_HW))
        intermediate_layers_cAFs.append(Tanh_HCNAF())

    conditional_AFs = conditional_AF_layer(
        *([MaskedWeight(args.dim_o, args.dim_o * args.dim_h_flow, dim=args.dim_o, norm_w=args.norm_HW), Tanh_HCNAF()] + \
        intermediate_layers_cAFs + \
        [MaskedWeight(args.dim_o * args.dim_h_flow, args.dim_o, dim=args.dim_o, norm_w=args.norm_HW)]))

    model = Sequential_HCNAF(HyperLayer, conditional_AFs).to(args.device)

    print('{}'.format(model))
    print('# of parameters={}, dim_flow={}'.format(sum((param != 0).sum() for param in model.parameters()), 2))

    return model


class conditional_AF_layer(torch.nn.Sequential):
    """
        A layer of conditional AF
    """
    def __init__(self, *args):
        super().__init__(*args)
        
    def forward(self, inputs, HyperParam):
        # Go over the layers of conditional AF
        log_det_j = None
        for module in self._modules.values():  
            inputs, log_det_j, HyperParam = module(inputs, log_det_j, HyperParam)
        
        # Summation over the flow dimension
        log_det_j_all = torch.sum(log_det_j.squeeze(), -1)

        return inputs, log_det_j_all, HyperParam
    

class Sequential_HCNAF(torch.nn.Sequential):
    """
        Modification of torch.nn.Sequential class to work with HCNAF. 
        Output: (1) inputs (2) log determinant of jacobians (3) Hyper-parameters
    """
    
    def forward(self, inputs):        
        log_det_j = 0
        HyperParam = []
        for i_sequential, module in enumerate(self._modules.values()):
            if i_sequential == 0: # Hypernetwork
                inputs, HyperParam = module(inputs)
            else: # Conditional AF
                inputs, log_det_j, HyperParam = module(inputs, HyperParam)

        return inputs, log_det_j, HyperParam


class MaskedWeight(torch.nn.Module):
    """
        Inspired by BNAF (https://arxiv.org/abs/1904.04676) MaskedWeight layer and its block propagation.
        Normalization of hyper-weights: (1) Scaled frobenius norm or (2) Modified weight norm.
    """
    
    def __init__(self, in_features, out_features, dim, norm_w='scaled_frobenius'):
        super().__init__()
        self.in_features, self.out_features, self.dim = in_features, out_features, dim
        self.H_f_in = self.in_features // self.dim    # Number of  input hidden units per flow dimension
        self.H_f_out = self.out_features // self.dim  # Number of output hidden units per flow dimension
        self.norm_w = norm_w

        # Parameters for the modified weight norm 
        if self.norm_w == 'modified_weightnorm':
            row_weight_temp = torch.nn.init.uniform_(torch.Tensor(self.out_features, 1)).log()
            self.row_weight = torch.nn.Parameter(row_weight_temp) # Independent of outputs of hyper-networks.
        elif self.norm_w == 'scaled_frobenius':
            self.scale_factor = torch.nn.Parameter(torch.nn.init.uniform_(torch.Tensor(1))) # Independent of outputs of hyper-networks.

        # Diagonal masks
        mask_d = torch.zeros(self.out_features, self.in_features)
        for i_mask in range(dim):
            mask_d[i_mask * self.H_f_out : (i_mask + 1) * self.H_f_out,
                   i_mask * self.H_f_in  : (i_mask + 1) * self.H_f_in
                   ] = 1

        self.register_buffer('mask_d', mask_d)

        # Non-diagonal masks
        mask_o = torch.ones(self.out_features, self.in_features)
        for i_mask in range(self.dim):
            mask_o[i_mask * self.H_f_out : (i_mask + 1) * self.H_f_out,
                   i_mask * self.H_f_in  : ] = 0

        self.register_buffer('mask_o', mask_o)

    def forward(self, inputs, log_grad_cumul = None, HyperParam = None):
        # Dimension of HyperNN_W : (Batch_size, num_in_feature * num_out_feature), HyperNN_B : (Batch_size, num_out_feature)
        HyperNN_W, HyperNN_B = HyperParam 
        Hyper_Weights = HyperNN_W[0].view(-1, self.out_features, self.in_features)  # (B, n_out, n_in)
        Hyper_Biases = HyperNN_B[0].view(-1, self.out_features)                     # (B, n_out)
        batch_size = Hyper_Weights.shape[0]

        w = torch.exp(Hyper_Weights) * self.mask_d.expand(batch_size, self.out_features, self.in_features) + Hyper_Weights * self.mask_o.expand(batch_size, self.out_features, self.in_features)    # w : (B, n_out, n_in). Multiplication from (B, n_out, n_in) * (n_out, n_in) : element-wise multiplication (broadcasted over the batch)
        if self.norm_w == 'scaled_frobenius':
            w_F_squared = (w ** 2).sum(-2, keepdim=True).sum(-1, keepdim=True)
            w_normed = self.scale_factor.exp() * w / w_F_squared.sqrt()

            # Get the diag matrices (i.e., parts of the weight matrix)
            # log_grad_dh : (B, dim, n_out per dim, n_in per dim), self.mask_d.bool() : (n_out, n_in)
            log_grad_dh = self.scale_factor + Hyper_Weights[self.mask_d.bool().repeat(batch_size, 1, 1)].view(batch_size, self.dim, self.H_f_out, self.H_f_in) - 0.5 * torch.log(w_F_squared).unsqueeze(-1)
        elif self.norm_w == 'modified_weightnorm':
            w_row_L2 = (w ** 2).sum(-1, keepdim=True)
            w_normed = self.row_weight.exp() * w / w_row_L2.sqrt()
            
            # Get the diag matrices (i.e., parts of the weight matrix)
            # log_grad_dh : (B, dim, n_out per dim, n_in per dim), self.mask_d.bool() : (n_out, n_in), w_normed_log_interest : (B*n_out*n_in) 
            w_normed_log_interest = self.row_weight.unsqueeze(0) + Hyper_Weights - 0.5 * torch.log(w_row_L2)
            log_grad_dh = w_normed_log_interest[self.mask_d.bool().repeat(batch_size, 1, 1)].view(batch_size, self.dim, self.H_f_out, self.H_f_in) 

        if log_grad_dh.sum() == float("inf") or log_grad_dh.sum() == float("-inf"): # Check the stability
            print("inf error") # Bad sign
        
        inputs = inputs.unsqueeze(-1)                                   
        outputs = torch.matmul(w_normed, inputs)                        # w * inputs : (B, n_out, n_in) * (B, n_in, 1) = (B, n_out, 1) 
        outputs = outputs.squeeze(-1) + Hyper_Biases                    # outputs : (B, n_out)

        # Computation of log_det_j: use logsumexp (addition + element-wise multiplication).
        # Sum over the input dimension (H_f_in). Resulting tensor: (B, dim, H_f_out, 1)
        if log_grad_cumul is None: # first layer
            outputs_log_det_j = log_grad_dh
        else:
            log_grad_sum = log_grad_dh.unsqueeze(-2) + log_grad_cumul.transpose(-2, -1).unsqueeze(-3)
            outputs_log_det_j = torch.logsumexp(log_grad_sum, -1)

        return outputs, outputs_log_det_j, [HyperNN_W[1:], HyperNN_B[1:]]

    def __repr__(self):
        return 'MaskedWeight(num_in_features:{}, num_out_features:{}, flow_dim:{})'.format(self.in_features, self.out_features, self.dim)

    
class Tanh_HCNAF(torch.nn.Tanh):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, log_grad_cumul = None, HyperParam = None):        
        log_grad = - 2 * (inputs - math.log(2) + torch.nn.functional.softplus(- 2 * inputs)) # log(derivative of tanh(inputs) w.r.t inputs)
        
        return torch.tanh(inputs), (log_grad.view(log_grad_cumul.shape) + log_grad_cumul) if log_grad_cumul is not None else log_grad, HyperParam


def define_encoder_2channel_lidar(Enc_Model, dim_h_Map, SAVEDIR, DEVICE, Enc_norm_method='BN', Pool_method='fmp'):
    '''
        define encoder, given input with channel = 2, lidar
    '''
    if Enc_Model == 'coordconv_simple':
        if dim_h_Map == 32: map_enc = coordconv_E32_HW200_lidar_simple(DEVICE, SAVEDIR, Enc_norm_method, Pool_method)
        elif dim_h_Map == 64: map_enc = coordconv_E64_HW200_lidar_simple(DEVICE, SAVEDIR, Enc_norm_method, Pool_method)
        elif dim_h_Map == 96: map_enc = coordconv_E96_HW200_lidar_simple(DEVICE, SAVEDIR, Enc_norm_method, Pool_method)
    elif Enc_Model == 'coordconv_complex':
        if dim_h_Map == 64: map_enc = coordconv_E64_HW200_lidar_complex(DEVICE, SAVEDIR, Enc_norm_method, Pool_method)
    elif Enc_Model == 'conv_simple':
        pass
    elif Enc_Model == 'conv_complex':
        pass
    
    return map_enc

class HyperNN_WB(torch.nn.Module):
    """
        Define weights, biases for the hyper-network of HCNAF (definition)
        Compute weights, biases (forward)
    """
    def __init__(self):
        super().__init__()

    def define_WB(self):
        self.w1_fc1 = torch.nn.Linear(self.dim_i_HyperNet, (self.dim_i_HyperNet)*5)
        #self.w1_act1 = torch.nn.Tanh()
        self.w1_act1 = torch.nn.ReLU()
        self.w1_fc2 = torch.nn.Linear((self.dim_i_HyperNet)*5, self.dim_o * (self.dim_o * self.dim_h_flow))

        self.b1_fc1 = torch.nn.Linear(self.dim_i_HyperNet, (self.dim_i_HyperNet)*3)
        #self.b1_act1 = torch.nn.Tanh()
        self.b1_act1 = torch.nn.ReLU()
        self.b1_fc2 = torch.nn.Linear((self.dim_i_HyperNet)*3, (self.dim_o * self.dim_h_flow))

        self.wn_fc1 = torch.nn.Linear(self.dim_i_HyperNet, (self.dim_i_HyperNet)*5)
        #self.wn_act1 = torch.nn.Tanh()
        self.wn_act1 = torch.nn.ReLU()
        self.wn_fc2 = torch.nn.Linear((self.dim_i_HyperNet)*5, (self.dim_o * self.dim_h_flow) * self.dim_o)

        self.bn_fc1 = torch.nn.Linear(self.dim_i_HyperNet, (self.dim_i_HyperNet)*3)
        #self.bn_act1 = torch.nn.Tanh()
        self.bn_act1 = torch.nn.ReLU()
        self.bn_fc2 = torch.nn.Linear((self.dim_i_HyperNet)*3, self.dim_o)

        self.weights_between = torch.nn.ModuleList()
        self.biases_between = torch.nn.ModuleList()
        for _ in range(self.n_layers_flow-1):
            self.weights_between.append(torch.nn.Linear(self.dim_i_HyperNet, (self.dim_i_HyperNet)*10))
            #self.weights_between.append(torch.nn.Tanh())
            self.weights_between.append(torch.nn.ReLU())
            self.weights_between.append(torch.nn.Linear((self.dim_i_HyperNet)*10, (self.dim_o * self.dim_h_flow) * (self.dim_o * self.dim_h_flow)))
            
            self.biases_between.append(torch.nn.Linear(self.dim_i_HyperNet, (self.dim_i_HyperNet)*3))
            #self.biases_between.append(torch.nn.Tanh())
            self.biases_between.append(torch.nn.ReLU())
            self.biases_between.append(torch.nn.Linear((self.dim_i_HyperNet)*3, (self.dim_o * self.dim_h_flow)))

    def compute_WB(self, inputs_to_HyperNet):
        weight_1 = self.w1_fc1(inputs_to_HyperNet)
        weight_1 = self.w1_act1(weight_1)
        weight_1 = self.w1_fc2(weight_1)

        bias_1 = self.b1_fc1(inputs_to_HyperNet)
        bias_1 = self.b1_act1(bias_1)
        bias_1 = self.b1_fc2(bias_1)

        weight_n = self.wn_fc1(inputs_to_HyperNet)
        weight_n = self.wn_act1(weight_n)
        weight_n = self.wn_fc2(weight_n)

        bias_n = self.bn_fc1(inputs_to_HyperNet)
        bias_n = self.bn_act1(bias_n)
        bias_n = self.bn_fc2(bias_n)

        Hyper_W_list = [weight_1]  # List of Weights : size (batch_size, in_feature * out_feature)
        Hyper_B_list = [bias_1]    # List of Biases : size (batch_size, out_feature)

        for layer_forward_i in range(self.n_layers_flow-1):
            weight_i = inputs_to_HyperNet
            for layer_forward_j in range(3):
                weight_i = self.weights_between[3*layer_forward_i + layer_forward_j](weight_i)
            Hyper_W_list.append(weight_i)

            bias_i = inputs_to_HyperNet
            for layer_forward_j in range(3):
                bias_i = self.biases_between[3*layer_forward_i + layer_forward_j](bias_i)
            Hyper_B_list.append(bias_i)

        Hyper_W_list.append(weight_n)
        Hyper_B_list.append(bias_n)

        return Hyper_W_list, Hyper_B_list


# LSTM
class LSTM_model(torch.nn.Module): # uses tanh as activation function. h is within the range of (-1,1)
    def __init__(self, input_size, dim_h_RNN, n_layers_RNN, b_first=True, device=torch.device("cuda")):
        super().__init__()
        # if batch_first = true, then input of lstm : (B, L, E), else, (L, B, E)
        self.device = device
        self.n_layers_RNN = n_layers_RNN
        self.dim_h_RNN = dim_h_RNN
        self.lstm = torch.nn.LSTM(input_size, dim_h_RNN, n_layers_RNN, batch_first=b_first)
    
    def forward(self, x): # x.size = [B, L, E]
        # Set initial hidden and cell states (size: [num_layers * num_directions, batch, hidden_size])
        #h0 = torch.zeros(self.n_layers_RNN, x.size(0), self.dim_h_RNN).to(self.device)
        #c0 = torch.zeros(self.n_layers_RNN, x.size(0), self.dim_h_RNN).to(self.device)
        h0 = torch.zeros(self.n_layers_RNN, x.size(0), self.dim_h_RNN).cuda()
        c0 = torch.zeros(self.n_layers_RNN, x.size(0), self.dim_h_RNN).cuda()

        # Forward propagate LSTM
        self.lstm.flatten_parameters()
        out, hidden = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size), hidden: the last hidden state (h,c) (2, batch_size, hidden_size)
    
        return out[:,-1,:]
  


class HyperNN_PRECOG_Carla_All(HyperNN_WB):
    """        
        Defines a hyper-network that takes conditions as input and returns weights and biases for the conditional AF.
        Ablation mode "All"
    """
    def __init__(self, args):
        super().__init__()
        self.is_WB_computed = False
        self.skip_HyperNet = False
        self.n_layers_flow, self.n_layers_RNN, self.dim_h_flow, self.dim_o = args.n_layers_flow, args.n_layers_RNN, args.dim_h_flow, args.dim_o
        self.dim_h_RNN, self.dim_sdv_hist, self.num_seq_input_sdv, self.temporal, self.dim_h_dt = args.dim_h_RNN, args.dim_sdv_hist, args.num_seq_input_sdv, args.temporal, args.dim_h_dt
        self.dim_h_Enc, self.Enc_norm_method, self.Enc_Model, self.path, self.device = args.dim_h_Enc, args.Enc_norm_method, args.Enc_Model, args.path, args.device
        self.HW_img, self.dim_c_EncInput, self.Pool_method = args.HW_img, args.dim_c_EncInput, args.Pool_method

        self.dim_h_traffic = args.dim_h_traffic
        self.dim_per_traffic = args.dim_per_traffic
        self.dim_traffic_hist = args.dim_traffic_hist
        self.max_num_actors = args.max_num_actors

        # if batch_first = true, then input of lstm : (B, L, E), else, (L, B, E)
        self.lstm_sdv_hist = LSTM_model(self.dim_sdv_hist//self.num_seq_input_sdv, self.dim_h_RNN, self.n_layers_RNN, b_first=True, device=self.device)

        # # Take the traffic states and go through a MLP
        self.lstm_actor1= LSTM_model(self.dim_traffic_hist//self.dim_per_traffic, self.dim_sdv_hist, self.n_layers_RNN, b_first=True, device=self.device)
        self.lstm_actor2= LSTM_model(self.dim_traffic_hist//self.dim_per_traffic, self.dim_sdv_hist, self.n_layers_RNN, b_first=True, device=self.device)
        self.lstm_actor3= LSTM_model(self.dim_traffic_hist//self.dim_per_traffic, self.dim_sdv_hist, self.n_layers_RNN, b_first=True, device=self.device)
        self.lstm_actor4= LSTM_model(self.dim_traffic_hist//self.dim_per_traffic, self.dim_sdv_hist, self.n_layers_RNN, b_first=True, device=self.device)

        self.all_actor_fc1 = torch.nn.Linear(self.dim_sdv_hist*self.max_num_actors, self.dim_sdv_hist*20)
        self.all_actor_act1 = torch.nn.Tanh()
        self.all_actor_fc2 = torch.nn.Linear(self.dim_sdv_hist*20, self.dim_h_traffic)

        # Take the map embeddings, and go through a MLP
        self.model_enc = define_encoder_2channel_lidar(self.Enc_Model, self.dim_h_Enc, self.path, self.device, self.Enc_norm_method, self.Pool_method)

        ## 1. Computes the hyperparameters (weights, biases)
        if self.temporal: # If time is part of context, then increase the dimension of the hypernet inputs by 1
            self.dim_i_HyperNet = self.dim_h_RNN + self.dim_h_traffic + self.dim_h_Enc + self.dim_h_dt
            self.wtime_fc1 = torch.nn.Linear(1, 64)
            self.wtime_act1 = torch.nn.ReLU()
            self.wtime_fc2 = torch.nn.Linear(64, 64)
            self.wtime_act2 = torch.nn.ReLU()
            self.wtime_fc3 = torch.nn.Linear(64, self.dim_h_dt)
        else:
            self.dim_i_HyperNet = self.dim_h_RNN + self.dim_h_traffic + self.dim_h_Enc

        super().define_WB()

    def forward(self, inputs):
        inputs, inputs_grid = inputs[0], inputs[1]                                           # inputs : [(B, dim_sdv_hist+dim_traffic+dim_o), (B, C, H, W)]
        #inputs = inputs.to(self.device)

        if self.skip_HyperNet == True and self.is_WB_computed == True:
            Hyper_W_list = self.Hyper_W_list
            Hyper_B_list = self.Hyper_B_list
        else:
            outputs_Enc = self.model_enc(inputs_grid).squeeze(-1).squeeze(-1)

            inputs_sdv_hist = inputs[:, :self.dim_sdv_hist]                                     # inputs_sdv_hist : (B, dim_sdv_hist)
            inputs_RNN = inputs_sdv_hist.view(-1, self.num_seq_input_sdv, self.dim_sdv_hist//self.num_seq_input_sdv)    # inputs_RNN : (B, seq_len, dim_sdv_hist/seq_len)
            outputs_RNN = self.lstm_sdv_hist(inputs_RNN)

            inputs_traffic = inputs[:, self.dim_sdv_hist : self.dim_sdv_hist+self.dim_traffic_hist*self.max_num_actors]    # inputs_traffic : (B, dim_traffic)
            inputs_traffic1 = inputs_traffic[:,                    :  self.dim_sdv_hist]
            inputs_traffic1 = inputs_traffic1.view(-1, self.num_seq_input_sdv, self.dim_sdv_hist//self.num_seq_input_sdv)   
            inputs_traffic2 = inputs_traffic[:,   self.dim_sdv_hist:2*self.dim_sdv_hist]
            inputs_traffic2 = inputs_traffic2.view(-1, self.num_seq_input_sdv, self.dim_sdv_hist//self.num_seq_input_sdv) 
            inputs_traffic3 = inputs_traffic[:, 2*self.dim_sdv_hist:3*self.dim_sdv_hist]
            inputs_traffic3 = inputs_traffic3.view(-1, self.num_seq_input_sdv, self.dim_sdv_hist//self.num_seq_input_sdv) 
            inputs_traffic4 = inputs_traffic[:, 3*self.dim_sdv_hist:4*self.dim_sdv_hist]
            inputs_traffic4 = inputs_traffic4.view(-1, self.num_seq_input_sdv, self.dim_sdv_hist//self.num_seq_input_sdv) 
            outputs_traffic1 = self.lstm_actor1(inputs_traffic1)
            outputs_traffic2 = self.lstm_actor1(inputs_traffic2)
            outputs_traffic3 = self.lstm_actor1(inputs_traffic3)
            outputs_traffic4 = self.lstm_actor1(inputs_traffic4)
            outputs_traffic = torch.cat((outputs_traffic1, outputs_traffic2, outputs_traffic3, outputs_traffic4),-1)

            outputs_traffic = self.all_actor_fc1(outputs_traffic)
            outputs_traffic = self.all_actor_act1(outputs_traffic)
            outputs_traffic = self.all_actor_fc2(outputs_traffic)       

            if self.temporal: # If time is part of context, then concatenate the RNN output with the time info.
                inputs_time = inputs[:, self.dim_sdv_hist+self.dim_traffic_hist*self.max_num_actors : self.dim_sdv_hist+self.dim_traffic_hist*self.max_num_actors+1] # inputs_time : (B, 1)
                # outputs_time = inputs_time.repeat(1, self.dim_h_dt)
                outputs_time = self.wtime_fc1(inputs_time)
                outputs_time = self.wtime_act1(outputs_time)
                outputs_time = self.wtime_fc2(outputs_time)
                outputs_time = self.wtime_act2(outputs_time)
                outputs_time = self.wtime_fc3(outputs_time)
                inputs_to_HyperNet = torch.cat((outputs_RNN, outputs_traffic, outputs_Enc, outputs_time), -1)
            else:
                inputs_to_HyperNet = torch.cat((outputs_RNN, outputs_traffic, outputs_Enc), -1)
            Hyper_W_list, Hyper_B_list = super().compute_WB(inputs_to_HyperNet)
            
            if self.skip_HyperNet == True:
                self.Hyper_W_list = Hyper_W_list
                self.Hyper_B_list = Hyper_B_list
                self.is_WB_computed = True

        if self.temporal:
            return inputs[:, self.dim_sdv_hist+self.dim_traffic_hist*self.max_num_actors+1:], [Hyper_W_list, Hyper_B_list]
        else:
            return inputs[:, self.dim_sdv_hist+self.dim_traffic_hist*self.max_num_actors:], [Hyper_W_list, Hyper_B_list]

class HyperNN_PRECOG_Carla_All_faster_temporal(HyperNN_WB):
    """        
        Defines a hyper-network that takes conditions as input and returns weights and biases for the conditional AF.
        Ablation mode "All", skip the computations up to HyperNN_WB, except the time-layer. 
    """
    def __init__(self, args):
        super().__init__()
        self.is_WB_computed = False
        self.skip_HyperNet = False
        self.n_layers_flow, self.n_layers_RNN, self.dim_h_flow, self.dim_o = args.n_layers_flow, args.n_layers_RNN, args.dim_h_flow, args.dim_o
        self.dim_h_RNN, self.dim_sdv_hist, self.num_seq_input_sdv, self.temporal, self.dim_h_dt = args.dim_h_RNN, args.dim_sdv_hist, args.num_seq_input_sdv, args.temporal, args.dim_h_dt
        self.dim_h_Enc, self.Enc_norm_method, self.Enc_Model, self.path, self.device = args.dim_h_Enc, args.Enc_norm_method, args.Enc_Model, args.path, args.device
        self.HW_img, self.dim_c_EncInput, self.Pool_method = args.HW_img, args.dim_c_EncInput, args.Pool_method

        self.dim_h_traffic = args.dim_h_traffic
        self.dim_per_traffic = args.dim_per_traffic
        self.dim_traffic_hist = args.dim_traffic_hist
        self.max_num_actors = args.max_num_actors

        # if batch_first = true, then input of lstm : (B, L, E), else, (L, B, E)
        self.lstm_sdv_hist = LSTM_model(self.dim_sdv_hist//self.num_seq_input_sdv, self.dim_h_RNN, self.n_layers_RNN, b_first=True, device=self.device)

        # Take the traffic states and go through a MLP
        self.lstm_actor1= LSTM_model(self.dim_traffic_hist//self.dim_per_traffic, self.dim_sdv_hist, self.n_layers_RNN, b_first=True, device=self.device)
        self.lstm_actor2= LSTM_model(self.dim_traffic_hist//self.dim_per_traffic, self.dim_sdv_hist, self.n_layers_RNN, b_first=True, device=self.device)
        self.lstm_actor3= LSTM_model(self.dim_traffic_hist//self.dim_per_traffic, self.dim_sdv_hist, self.n_layers_RNN, b_first=True, device=self.device)
        self.lstm_actor4= LSTM_model(self.dim_traffic_hist//self.dim_per_traffic, self.dim_sdv_hist, self.n_layers_RNN, b_first=True, device=self.device)

        self.all_actor_fc1 = torch.nn.Linear(self.dim_sdv_hist*self.max_num_actors, self.dim_sdv_hist*20)
        self.all_actor_act1 = torch.nn.Tanh()
        self.all_actor_fc2 = torch.nn.Linear(self.dim_sdv_hist*20, self.dim_h_traffic)

        # Take the map embeddings, and go through a MLP
        self.model_enc = define_encoder_2channel_lidar(self.Enc_Model, self.dim_h_Enc, self.path, self.device, self.Enc_norm_method, self.Pool_method)

        ## 1. Computes the hyperparameters (weights, biases)
        if self.temporal: # If time is part of context, then increase the dimension of the hypernet inputs by 1
            self.dim_i_HyperNet = self.dim_h_RNN + self.dim_h_traffic + self.dim_h_Enc + self.dim_h_dt
            self.wtime_fc1 = torch.nn.Linear(1, 64)
            self.wtime_act1 = torch.nn.ReLU()
            self.wtime_fc2 = torch.nn.Linear(64, 64)
            self.wtime_act2 = torch.nn.ReLU()
            self.wtime_fc3 = torch.nn.Linear(64, self.dim_h_dt)
        else:
            self.dim_i_HyperNet = self.dim_h_RNN + self.dim_h_traffic + self.dim_h_Enc

        super().define_WB()

    def forward(self, inputs):
        inputs, inputs_grid = inputs[0], inputs[1]                                           # inputs : [(B, dim_sdv_hist+dim_traffic+dim_o), (B, C, H, W)]
        #inputs = inputs.to(self.device)

        assert self.temporal == 1, 'This model requires temporal = 1'

        if self.skip_HyperNet == True and self.is_WB_computed == True:
            outputs_traffic = self.outputs_traffic
            outputs_RNN = self.outputs_RNN
            outputs_Enc = self.outputs_Enc
        else:
            self.outputs_traffic = None
            self.outputs_RNN = None
            self.outputs_Enc = None
            outputs_Enc = self.model_enc(inputs_grid).squeeze(-1).squeeze(-1)

            inputs_sdv_hist = inputs[:, :self.dim_sdv_hist]                                     # inputs_sdv_hist : (B, dim_sdv_hist)
            inputs_RNN = inputs_sdv_hist.view(-1, self.num_seq_input_sdv, self.dim_sdv_hist//self.num_seq_input_sdv)    # inputs_RNN : (B, seq_len, dim_sdv_hist/seq_len)
            outputs_RNN = self.lstm_sdv_hist(inputs_RNN)

            inputs_traffic = inputs[:, self.dim_sdv_hist : self.dim_sdv_hist+self.dim_traffic_hist*self.max_num_actors]    # inputs_traffic : (B, dim_traffic)
            inputs_traffic1 = inputs_traffic[:,                    :  self.dim_sdv_hist]
            inputs_traffic1 = inputs_traffic1.view(-1, self.num_seq_input_sdv, self.dim_sdv_hist//self.num_seq_input_sdv)   
            inputs_traffic2 = inputs_traffic[:,   self.dim_sdv_hist:2*self.dim_sdv_hist]
            inputs_traffic2 = inputs_traffic2.view(-1, self.num_seq_input_sdv, self.dim_sdv_hist//self.num_seq_input_sdv) 
            inputs_traffic3 = inputs_traffic[:, 2*self.dim_sdv_hist:3*self.dim_sdv_hist]
            inputs_traffic3 = inputs_traffic3.view(-1, self.num_seq_input_sdv, self.dim_sdv_hist//self.num_seq_input_sdv) 
            inputs_traffic4 = inputs_traffic[:, 3*self.dim_sdv_hist:4*self.dim_sdv_hist]
            inputs_traffic4 = inputs_traffic4.view(-1, self.num_seq_input_sdv, self.dim_sdv_hist//self.num_seq_input_sdv) 
            outputs_traffic1 = self.lstm_actor1(inputs_traffic1)
            outputs_traffic2 = self.lstm_actor1(inputs_traffic2)
            outputs_traffic3 = self.lstm_actor1(inputs_traffic3)
            outputs_traffic4 = self.lstm_actor1(inputs_traffic4)
            outputs_traffic = torch.cat((outputs_traffic1, outputs_traffic2, outputs_traffic3, outputs_traffic4),-1)

            outputs_traffic = self.all_actor_fc1(outputs_traffic)
            outputs_traffic = self.all_actor_act1(outputs_traffic)
            outputs_traffic = self.all_actor_fc2(outputs_traffic)  

            if self.skip_HyperNet == True:
                self.outputs_traffic = outputs_traffic
                self.outputs_RNN = outputs_RNN
                self.outputs_Enc = outputs_Enc
                self.is_WB_computed = True

        # Assumption: self.temporal == 1
        inputs_time = inputs[:, self.dim_sdv_hist+self.dim_traffic_hist*self.max_num_actors : self.dim_sdv_hist+self.dim_traffic_hist*self.max_num_actors+1] # inputs_time : (B, 1)
        outputs_time = self.wtime_fc1(inputs_time)
        outputs_time = self.wtime_act1(outputs_time)
        outputs_time = self.wtime_fc2(outputs_time)
        outputs_time = self.wtime_act2(outputs_time)
        outputs_time = self.wtime_fc3(outputs_time)
        inputs_to_HyperNet = torch.cat((outputs_RNN, outputs_traffic, outputs_Enc, outputs_time), -1)

        Hyper_W_list, Hyper_B_list = super().compute_WB(inputs_to_HyperNet)

        return inputs[:, self.dim_sdv_hist+self.dim_traffic_hist*self.max_num_actors+1:], [Hyper_W_list, Hyper_B_list]



class HyperNN_PRECOG_Carla_NoLidar(HyperNN_WB):
    """        
        Defines a hyper-network that takes conditions as input and returns weights and biases for the conditional AF.
        Ablation mode "No_Lidar"
    """
    def __init__(self, args):
        super().__init__()
        self.n_layers_flow, self.n_layers_RNN, self.dim_h_flow, self.dim_o = args.n_layers_flow, args.n_layers_RNN, args.dim_h_flow, args.dim_o
        self.dim_h_RNN, self.dim_sdv_hist, self.num_seq_input_sdv, self.temporal, self.dim_h_dt = args.dim_h_RNN, args.dim_sdv_hist, args.num_seq_input_sdv, args.temporal, args.dim_h_dt
        self.path, self.device = args.path, args.device
        self.dim_h_traffic = args.dim_h_traffic
        self.dim_per_traffic = args.dim_per_traffic
        self.dim_traffic_hist = args.dim_traffic_hist
        self.max_num_actors = args.max_num_actors

        # if batch_first = true, then input of lstm : (B, L, E), else, (L, B, E)
        self.lstm_sdv_hist = LSTM_model(self.dim_sdv_hist//self.num_seq_input_sdv, self.dim_h_RNN, self.n_layers_RNN, b_first=True, device=self.device)

        # # Take the traffic states and go through a MLP
        self.lstm_actor1= LSTM_model(self.dim_traffic_hist//self.dim_per_traffic, self.dim_sdv_hist, self.n_layers_RNN, b_first=True, device=self.device)
        self.lstm_actor2= LSTM_model(self.dim_traffic_hist//self.dim_per_traffic, self.dim_sdv_hist, self.n_layers_RNN, b_first=True, device=self.device)
        self.lstm_actor3= LSTM_model(self.dim_traffic_hist//self.dim_per_traffic, self.dim_sdv_hist, self.n_layers_RNN, b_first=True, device=self.device)
        self.lstm_actor4= LSTM_model(self.dim_traffic_hist//self.dim_per_traffic, self.dim_sdv_hist, self.n_layers_RNN, b_first=True, device=self.device)

        self.all_actor_fc1 = torch.nn.Linear(self.dim_sdv_hist*self.max_num_actors, self.dim_sdv_hist*20)
        self.all_actor_act1 = torch.nn.Tanh()
        self.all_actor_fc2 = torch.nn.Linear(self.dim_sdv_hist*20, self.dim_h_traffic)

        ## 1. Computes the hyperparameters (weights, biases)
        if self.temporal: # If time is part of context, then increase the dimension of the hypernet inputs by 1
            self.dim_i_HyperNet = self.dim_h_RNN + self.dim_h_traffic + self.dim_h_dt
            self.wtime_fc1 = torch.nn.Linear(1, 64)
            self.wtime_act1 = torch.nn.ReLU()
            self.wtime_fc2 = torch.nn.Linear(64, 64)
            self.wtime_act2 = torch.nn.ReLU()
            self.wtime_fc3 = torch.nn.Linear(64, self.dim_h_dt)
        else:
            self.dim_i_HyperNet = self.dim_h_RNN + self.dim_h_traffic

        super().define_WB()

    def forward(self, inputs):
        """
        Takes: inputs(conditions included)
        Returns: inputs(conditions excluded), Hyperparameters
        """
        #inputs = inputs.to(self.device)

        inputs_sdv_hist = inputs[:, :self.dim_sdv_hist]                                     # inputs_sdv_hist : (B, dim_sdv_hist)
        inputs_RNN = inputs_sdv_hist.view(-1, self.num_seq_input_sdv, self.dim_sdv_hist//self.num_seq_input_sdv)    # inputs_RNN : (B, seq_len, dim_sdv_hist/seq_len)
        outputs_RNN = self.lstm_sdv_hist(inputs_RNN)

        inputs_traffic = inputs[:, self.dim_sdv_hist : self.dim_sdv_hist+self.dim_traffic_hist*self.max_num_actors]    # inputs_traffic : (B, dim_traffic)
        inputs_traffic1 = inputs_traffic[:,                    :  self.dim_sdv_hist]
        inputs_traffic1 = inputs_traffic1.view(-1, self.num_seq_input_sdv, self.dim_sdv_hist//self.num_seq_input_sdv)   
        inputs_traffic2 = inputs_traffic[:,   self.dim_sdv_hist:2*self.dim_sdv_hist]
        inputs_traffic2 = inputs_traffic2.view(-1, self.num_seq_input_sdv, self.dim_sdv_hist//self.num_seq_input_sdv) 
        inputs_traffic3 = inputs_traffic[:, 2*self.dim_sdv_hist:3*self.dim_sdv_hist]
        inputs_traffic3 = inputs_traffic3.view(-1, self.num_seq_input_sdv, self.dim_sdv_hist//self.num_seq_input_sdv) 
        inputs_traffic4 = inputs_traffic[:, 3*self.dim_sdv_hist:4*self.dim_sdv_hist]
        inputs_traffic4 = inputs_traffic4.view(-1, self.num_seq_input_sdv, self.dim_sdv_hist//self.num_seq_input_sdv) 
        outputs_traffic1 = self.lstm_actor1(inputs_traffic1)
        outputs_traffic2 = self.lstm_actor1(inputs_traffic2)
        outputs_traffic3 = self.lstm_actor1(inputs_traffic3)
        outputs_traffic4 = self.lstm_actor1(inputs_traffic4)
        outputs_traffic = torch.cat((outputs_traffic1, outputs_traffic2, outputs_traffic3, outputs_traffic4),-1)

        outputs_traffic = self.all_actor_fc1(outputs_traffic)
        outputs_traffic = self.all_actor_act1(outputs_traffic)
        outputs_traffic = self.all_actor_fc2(outputs_traffic)  

        if self.temporal: # If time is part of context, then concatenate the RNN output with the time info.
            inputs_time = inputs[:, self.dim_sdv_hist+self.dim_traffic_hist*self.max_num_actors : self.dim_sdv_hist+self.dim_traffic_hist*self.max_num_actors+1] # inputs_time : (B, 1)
            outputs_time = self.wtime_fc1(inputs_time)
            outputs_time = self.wtime_act1(outputs_time)
            outputs_time = self.wtime_fc2(outputs_time)
            outputs_time = self.wtime_act2(outputs_time)
            outputs_time = self.wtime_fc3(outputs_time)
            inputs_to_HyperNet = torch.cat((outputs_RNN, outputs_traffic, outputs_time), -1)
        else:
            inputs_to_HyperNet = torch.cat((outputs_RNN, outputs_traffic), -1)

        Hyper_W_list, Hyper_B_list = super().compute_WB(inputs_to_HyperNet)

        if self.temporal:
            return inputs[:, self.dim_sdv_hist+self.dim_traffic_hist*self.max_num_actors+1:], [Hyper_W_list, Hyper_B_list]
        else:
            return inputs[:, self.dim_sdv_hist+self.dim_traffic_hist*self.max_num_actors:], [Hyper_W_list, Hyper_B_list]


class HyperNN_PRECOG_Carla_NoLidar_faster_temporal(HyperNN_WB):
    """        
        Defines a hyper-network that takes conditions as input and returns weights and biases for the conditional AF.
        Ablation mode "No_Lidar"
    """
    def __init__(self, args):
        super().__init__()
        self.is_WB_computed = False
        self.skip_HyperNet = False
        self.n_layers_flow, self.n_layers_RNN, self.dim_h_flow, self.dim_o = args.n_layers_flow, args.n_layers_RNN, args.dim_h_flow, args.dim_o
        self.dim_h_RNN, self.dim_sdv_hist, self.num_seq_input_sdv, self.temporal, self.dim_h_dt = args.dim_h_RNN, args.dim_sdv_hist, args.num_seq_input_sdv, args.temporal, args.dim_h_dt
        self.path, self.device = args.path, args.device
        self.dim_h_traffic = args.dim_h_traffic
        self.dim_per_traffic = args.dim_per_traffic
        self.dim_traffic_hist = args.dim_traffic_hist
        self.max_num_actors = args.max_num_actors

        # if batch_first = true, then input of lstm : (B, L, E), else, (L, B, E)
        self.lstm_sdv_hist = LSTM_model(self.dim_sdv_hist//self.num_seq_input_sdv, self.dim_h_RNN, self.n_layers_RNN, b_first=True, device=self.device)

        # # Take the traffic states and go through a MLP
        self.lstm_actor1= LSTM_model(self.dim_traffic_hist//self.dim_per_traffic, self.dim_sdv_hist, self.n_layers_RNN, b_first=True, device=self.device)
        self.lstm_actor2= LSTM_model(self.dim_traffic_hist//self.dim_per_traffic, self.dim_sdv_hist, self.n_layers_RNN, b_first=True, device=self.device)
        self.lstm_actor3= LSTM_model(self.dim_traffic_hist//self.dim_per_traffic, self.dim_sdv_hist, self.n_layers_RNN, b_first=True, device=self.device)
        self.lstm_actor4= LSTM_model(self.dim_traffic_hist//self.dim_per_traffic, self.dim_sdv_hist, self.n_layers_RNN, b_first=True, device=self.device)

        self.all_actor_fc1 = torch.nn.Linear(self.dim_sdv_hist*self.max_num_actors, self.dim_sdv_hist*20)
        self.all_actor_act1 = torch.nn.Tanh()
        self.all_actor_fc2 = torch.nn.Linear(self.dim_sdv_hist*20, self.dim_h_traffic)

        # 1. Computes the hyperparameters (weights, biases)
        if self.temporal: # If time is part of context, then increase the dimension of the hypernet inputs by 1
            self.dim_i_HyperNet = self.dim_h_RNN + self.dim_h_traffic + self.dim_h_dt
            self.wtime_fc1 = torch.nn.Linear(1, 64)
            self.wtime_act1 = torch.nn.ReLU()
            self.wtime_fc2 = torch.nn.Linear(64, 64)
            self.wtime_act2 = torch.nn.ReLU()
            self.wtime_fc3 = torch.nn.Linear(64, self.dim_h_dt)
        else:
            self.dim_i_HyperNet = self.dim_h_RNN + self.dim_h_traffic

        super().define_WB()

    def forward(self, inputs):
        assert self.temporal == 1, 'This model requires temporal = 1'

        if self.skip_HyperNet == True and self.is_WB_computed == True:
            outputs_traffic = self.outputs_traffic
            outputs_RNN = self.outputs_RNN
        else:
            inputs_sdv_hist = inputs[:, :self.dim_sdv_hist]                                     # inputs_sdv_hist : (B, dim_sdv_hist)
            inputs_RNN = inputs_sdv_hist.view(-1, self.num_seq_input_sdv, self.dim_sdv_hist//self.num_seq_input_sdv)    # inputs_RNN : (B, seq_len, dim_sdv_hist/seq_len)
            outputs_RNN = self.lstm_sdv_hist(inputs_RNN)

            inputs_traffic = inputs[:, self.dim_sdv_hist : self.dim_sdv_hist+self.dim_traffic_hist*self.max_num_actors]    # inputs_traffic : (B, dim_traffic)
            inputs_traffic1 = inputs_traffic[:,                    :  self.dim_sdv_hist]
            inputs_traffic1 = inputs_traffic1.view(-1, self.num_seq_input_sdv, self.dim_sdv_hist//self.num_seq_input_sdv)   
            inputs_traffic2 = inputs_traffic[:,   self.dim_sdv_hist:2*self.dim_sdv_hist]
            inputs_traffic2 = inputs_traffic2.view(-1, self.num_seq_input_sdv, self.dim_sdv_hist//self.num_seq_input_sdv) 
            inputs_traffic3 = inputs_traffic[:, 2*self.dim_sdv_hist:3*self.dim_sdv_hist]
            inputs_traffic3 = inputs_traffic3.view(-1, self.num_seq_input_sdv, self.dim_sdv_hist//self.num_seq_input_sdv) 
            inputs_traffic4 = inputs_traffic[:, 3*self.dim_sdv_hist:4*self.dim_sdv_hist]
            inputs_traffic4 = inputs_traffic4.view(-1, self.num_seq_input_sdv, self.dim_sdv_hist//self.num_seq_input_sdv) 
            outputs_traffic1 = self.lstm_actor1(inputs_traffic1)
            outputs_traffic2 = self.lstm_actor1(inputs_traffic2)
            outputs_traffic3 = self.lstm_actor1(inputs_traffic3)
            outputs_traffic4 = self.lstm_actor1(inputs_traffic4)
            outputs_traffic = torch.cat((outputs_traffic1, outputs_traffic2, outputs_traffic3, outputs_traffic4),-1)

            outputs_traffic = self.all_actor_fc1(outputs_traffic)
            outputs_traffic = self.all_actor_act1(outputs_traffic)
            outputs_traffic = self.all_actor_fc2(outputs_traffic)    

            if self.skip_HyperNet == True:
                self.outputs_traffic = outputs_traffic
                self.outputs_RNN = outputs_RNN
                self.is_WB_computed = True

        # Assumption: self.temporal == 1 
        inputs_time = inputs[:, self.dim_sdv_hist+self.dim_traffic_hist*self.max_num_actors : self.dim_sdv_hist+self.dim_traffic_hist*self.max_num_actors+1] # inputs_time : (B, 1)
        outputs_time = self.wtime_fc1(inputs_time)
        outputs_time = self.wtime_act1(outputs_time)
        outputs_time = self.wtime_fc2(outputs_time)
        outputs_time = self.wtime_act2(outputs_time)
        outputs_time = self.wtime_fc3(outputs_time)
        inputs_to_HyperNet = torch.cat((outputs_RNN, outputs_traffic, outputs_time), -1)

        Hyper_W_list, Hyper_B_list = super().compute_WB(inputs_to_HyperNet)

        if self.temporal:
            return inputs[:, self.dim_sdv_hist+self.dim_traffic_hist*self.max_num_actors+1:], [Hyper_W_list, Hyper_B_list]
        else:
            return inputs[:, self.dim_sdv_hist+self.dim_traffic_hist*self.max_num_actors:], [Hyper_W_list, Hyper_B_list]
