"""
# Author      : GS Oh
# Experiment  : The gaussian experiments presented in the HCNAF paper
# Note        : Define HCNAF (Hyper-conditioned neural autoregressive flow) customized to the gaussian experiments
"""
import numpy as np
import torch
import math
from exp_hcnaf_gaussians import sample_dataset   # Import hcnaf dataset


# Computing Loss function
def compute_log_p_x(model, data):
    z, log_det_j, HyperParam = model(data)
    log_p_z = torch.distributions.Normal(torch.zeros_like(z), torch.ones_like(z)).log_prob(z).sum(-1)
    return log_p_z + log_det_j

# Computing NLL, evaluation purpose
def compute_NLL(model, args, target_condition):
    val_loss = 0

    model.eval()
    with torch.no_grad():
        # mean of 640 samples
        for idx_val in range(1000):
            data = torch.from_numpy(sample_dataset(10, args, flag_sample_target_condition = True, target_condition = target_condition)).float().to(args.device)
            val_loss += - compute_log_p_x(model, data).mean()
        val_loss /= 1000
        val_loss = round(val_loss.item(), 4)
    print('Target_condition: {}, NLL: {}'.format(target_condition, val_loss))


# Create HCNAF model
def create_HCNAF(args):
    """
        Create a HCNAF model; (1) Hyper-network (2) Conditional Autoregressive Flow
    """
    # Define a hyper-network
    if args.hypernet_layers == '2s':
        HyperLayer = HyperNN_L2s(args)
    elif args.hypernet_layers == '2':
        HyperLayer = HyperNN_L2(args)
    elif args.hypernet_layers == '2b':
        HyperLayer = HyperNN_L2b(args)
    elif args.hypernet_layers == '3':
        HyperLayer = HyperNN_L3(args)

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
                   i_mask * self.H_f_in  : (i_mask + 1) * self.H_f_in] = 1

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


class HyperNN_L2s(torch.nn.Module):
    """
        Defines a hyper-network that takes conditions as input and returns weights and biases for the conditional AF.
        The hyper-network is a two-layers MLP with small number of parameters
    """
    def __init__(self, args):
        super().__init__()
        self.n_layers_flow, self.dim_h_flow, self.dim, self.dim_c, self.dev = args.n_layers_flow, args.dim_h_flow, args.dim_o, args.cond_dim, args.device
        
        # 1. Computes the hyperparameters (weights, biases)
        self.w1_fc1 = torch.nn.Linear(self.dim_c, self.dim_c*10)
        #self.w1_act1 = torch.nn.Tanh()
        self.w1_act1 = torch.nn.ReLU()
        self.w1_fc2 = torch.nn.Linear(self.dim_c*10, self.dim * (self.dim * self.dim_h_flow))

        self.b1_fc1 = torch.nn.Linear(self.dim_c, self.dim_c*50)
        #self.b1_act1 = torch.nn.Tanh()
        self.b1_act1 = torch.nn.ReLU()
        self.b1_fc2 = torch.nn.Linear(self.dim_c*50, (self.dim * self.dim_h_flow))

        self.wn_fc1 = torch.nn.Linear(self.dim_c, self.dim_c*10)
        #self.wn_act1 = torch.nn.Tanh()
        self.wn_act1 = torch.nn.ReLU()
        self.wn_fc2 = torch.nn.Linear(self.dim_c*10, (self.dim * self.dim_h_flow) * self.dim)

        self.bn_fc1 = torch.nn.Linear(self.dim_c, self.dim_c*50)
        #self.bn_act1 = torch.nn.Tanh()
        self.bn_act1 = torch.nn.ReLU()
        self.bn_fc2 = torch.nn.Linear(self.dim_c*50, self.dim)

        self.weights_between = torch.nn.ModuleList()
        self.biases_between = torch.nn.ModuleList()
        for _ in range(self.n_layers_flow-1):
            self.weights_between.append(torch.nn.Linear(self.dim_c, self.dim_c*10))
            #self.weights_between.append(torch.nn.Tanh())
            self.weights_between.append(torch.nn.ReLU())
            self.weights_between.append(torch.nn.Linear(self.dim_c*10, (self.dim * self.dim_h_flow) * (self.dim * self.dim_h_flow)))
            
            self.biases_between.append(torch.nn.Linear(self.dim_c, self.dim_c*50))
            #self.biases_between.append(torch.nn.Tanh())
            self.biases_between.append(torch.nn.ReLU())
            self.biases_between.append(torch.nn.Linear(self.dim_c*50, (self.dim * self.dim_h_flow)))
        

    def forward(self, inputs):
        inputs_c = inputs[:, :self.dim_c]  # Conditions
        
        weight_1 = self.w1_fc1(inputs_c)
        weight_1 = self.w1_act1(weight_1)
        weight_1 = self.w1_fc2(weight_1)

        bias_1 = self.b1_fc1(inputs_c)
        bias_1 = self.b1_act1(bias_1)
        bias_1 = self.b1_fc2(bias_1)

        weight_n = self.wn_fc1(inputs_c)
        weight_n = self.wn_act1(weight_n)
        weight_n = self.wn_fc2(weight_n)

        bias_n = self.bn_fc1(inputs_c)
        bias_n = self.bn_act1(bias_n)
        bias_n = self.bn_fc2(bias_n)

        Hyper_W_list = [weight_1]  # List of Weights : size (batch_size, in_feature * out_feature)
        Hyper_B_list = [bias_1]    # List of Biases : size (batch_size, out_feature)

        for layer_forward_i in range(self.n_layers_flow-1):
            weight_i = inputs_c
            for layer_forward_j in range(3):
                weight_i = self.weights_between[3*layer_forward_i + layer_forward_j](weight_i)
            Hyper_W_list.append(weight_i)

            bias_i = inputs_c
            for layer_forward_j in range(3):
                bias_i = self.biases_between[3*layer_forward_i + layer_forward_j](bias_i)
            Hyper_B_list.append(bias_i)

        Hyper_W_list.append(weight_n)
        Hyper_B_list.append(bias_n)

        return inputs[:, self.dim_c:], [Hyper_W_list, Hyper_B_list]
        

class HyperNN_L2(torch.nn.Module):
    """
        Defines a hyper-network that takes conditions as input and returns weights and biases for the conditional AF.
        The hyper-network is a two-layers MLP
    """
    def __init__(self, args):
        super().__init__()
        self.n_layers_flow, self.dim_h_flow, self.dim, self.dim_c, self.dev = args.n_layers_flow, args.dim_h_flow, args.dim_o, args.cond_dim, args.device
        
        # 1. Computes the hyperparameters (weights, biases)
        self.w1_fc1 = torch.nn.Linear(self.dim_c, self.dim_c*25)
        #self.w1_act1 = torch.nn.Tanh()
        self.w1_act1 = torch.nn.ReLU()
        self.w1_fc2 = torch.nn.Linear(self.dim_c*25, self.dim * (self.dim * self.dim_h_flow))

        self.b1_fc1 = torch.nn.Linear(self.dim_c, self.dim_c*50)
        #self.b1_act1 = torch.nn.Tanh()
        self.b1_act1 = torch.nn.ReLU()
        self.b1_fc2 = torch.nn.Linear(self.dim_c*50, (self.dim * self.dim_h_flow))

        self.wn_fc1 = torch.nn.Linear(self.dim_c, self.dim_c*25)
        #self.wn_act1 = torch.nn.Tanh()
        self.wn_act1 = torch.nn.ReLU()
        self.wn_fc2 = torch.nn.Linear(self.dim_c*25, (self.dim * self.dim_h_flow) * self.dim)

        self.bn_fc1 = torch.nn.Linear(self.dim_c, self.dim_c*50)
        #self.bn_act1 = torch.nn.Tanh()
        self.bn_act1 = torch.nn.ReLU()
        self.bn_fc2 = torch.nn.Linear(self.dim_c*50, self.dim)

        self.weights_between = torch.nn.ModuleList()
        self.biases_between = torch.nn.ModuleList()
        for _ in range(self.n_layers_flow-1):
            self.weights_between.append(torch.nn.Linear(self.dim_c, self.dim_c*30))
            #self.weights_between.append(torch.nn.Tanh())
            self.weights_between.append(torch.nn.ReLU())
            self.weights_between.append(torch.nn.Linear(self.dim_c*30, (self.dim * self.dim_h_flow) * (self.dim * self.dim_h_flow)))
            
            self.biases_between.append(torch.nn.Linear(self.dim_c, self.dim_c*50))
            #self.biases_between.append(torch.nn.Tanh())
            self.biases_between.append(torch.nn.ReLU())
            self.biases_between.append(torch.nn.Linear(self.dim_c*50, (self.dim * self.dim_h_flow)))
        

    def forward(self, inputs):
        inputs_c = inputs[:, :self.dim_c]  # Conditions
        
        weight_1 = self.w1_fc1(inputs_c)
        weight_1 = self.w1_act1(weight_1)
        weight_1 = self.w1_fc2(weight_1)

        bias_1 = self.b1_fc1(inputs_c)
        bias_1 = self.b1_act1(bias_1)
        bias_1 = self.b1_fc2(bias_1)

        weight_n = self.wn_fc1(inputs_c)
        weight_n = self.wn_act1(weight_n)
        weight_n = self.wn_fc2(weight_n)

        bias_n = self.bn_fc1(inputs_c)
        bias_n = self.bn_act1(bias_n)
        bias_n = self.bn_fc2(bias_n)

        Hyper_W_list = [weight_1]  # List of Weights : size (batch_size, in_feature * out_feature)
        Hyper_B_list = [bias_1]    # List of Biases : size (batch_size, out_feature)

        for layer_forward_i in range(self.n_layers_flow-1):
            weight_i = inputs_c
            for layer_forward_j in range(3):
                weight_i = self.weights_between[3*layer_forward_i + layer_forward_j](weight_i)
            Hyper_W_list.append(weight_i)

            bias_i = inputs_c
            for layer_forward_j in range(3):
                bias_i = self.biases_between[3*layer_forward_i + layer_forward_j](bias_i)
            Hyper_B_list.append(bias_i)

        Hyper_W_list.append(weight_n)
        Hyper_B_list.append(bias_n)

        return inputs[:, self.dim_c:], [Hyper_W_list, Hyper_B_list]


class HyperNN_L2b(torch.nn.Module):
    """
        Defines a hyper-network that takes conditions as input and returns weights and biases for the conditional AF.
        The hyper-network is a two-layers MLP with big number of parameters
    """
    def __init__(self, args):
        super().__init__()
        self.n_layers_flow, self.dim_h_flow, self.dim, self.dim_c, self.dev = args.n_layers_flow, args.dim_h_flow, args.dim_o, args.cond_dim, args.device
        
        # 1. Computes the hyperparameters (weights, biases)
        self.w1_fc1 = torch.nn.Linear(self.dim_c, self.dim_c*50)
        #self.w1_act1 = torch.nn.Tanh()
        self.w1_act1 = torch.nn.ReLU()
        self.w1_fc2 = torch.nn.Linear(self.dim_c*50, self.dim * (self.dim * self.dim_h_flow))

        self.b1_fc1 = torch.nn.Linear(self.dim_c, self.dim_c*100)
        #self.b1_act1 = torch.nn.Tanh()
        self.b1_act1 = torch.nn.ReLU()
        self.b1_fc2 = torch.nn.Linear(self.dim_c*100, (self.dim * self.dim_h_flow))

        self.wn_fc1 = torch.nn.Linear(self.dim_c, self.dim_c*50)
        #self.wn_act1 = torch.nn.Tanh()
        self.wn_act1 = torch.nn.ReLU()
        self.wn_fc2 = torch.nn.Linear(self.dim_c*50, (self.dim * self.dim_h_flow) * self.dim)

        self.bn_fc1 = torch.nn.Linear(self.dim_c, self.dim_c*100)
        #self.bn_act1 = torch.nn.Tanh()
        self.bn_act1 = torch.nn.ReLU()
        self.bn_fc2 = torch.nn.Linear(self.dim_c*100, self.dim)

        self.weights_between = torch.nn.ModuleList()
        self.biases_between = torch.nn.ModuleList()
        for _ in range(self.n_layers_flow-1):
            self.weights_between.append(torch.nn.Linear(self.dim_c, self.dim_c*60))
            #self.weights_between.append(torch.nn.Tanh())
            self.weights_between.append(torch.nn.ReLU())
            self.weights_between.append(torch.nn.Linear(self.dim_c*60, (self.dim * self.dim_h_flow) * (self.dim * self.dim_h_flow)))
            
            self.biases_between.append(torch.nn.Linear(self.dim_c, self.dim_c*100))
            #self.biases_between.append(torch.nn.Tanh())
            self.biases_between.append(torch.nn.ReLU())
            self.biases_between.append(torch.nn.Linear(self.dim_c*100, (self.dim * self.dim_h_flow)))
        

    def forward(self, inputs):
        inputs_c = inputs[:, :self.dim_c]  # Conditions
        
        weight_1 = self.w1_fc1(inputs_c)
        weight_1 = self.w1_act1(weight_1)
        weight_1 = self.w1_fc2(weight_1)

        bias_1 = self.b1_fc1(inputs_c)
        bias_1 = self.b1_act1(bias_1)
        bias_1 = self.b1_fc2(bias_1)

        weight_n = self.wn_fc1(inputs_c)
        weight_n = self.wn_act1(weight_n)
        weight_n = self.wn_fc2(weight_n)

        bias_n = self.bn_fc1(inputs_c)
        bias_n = self.bn_act1(bias_n)
        bias_n = self.bn_fc2(bias_n)

        Hyper_W_list = [weight_1]  # List of Weights : size (batch_size, in_feature * out_feature)
        Hyper_B_list = [bias_1]    # List of Biases : size (batch_size, out_feature)

        for layer_forward_i in range(self.n_layers_flow-1):
            weight_i = inputs_c
            for layer_forward_j in range(3):
                weight_i = self.weights_between[3*layer_forward_i + layer_forward_j](weight_i)
            Hyper_W_list.append(weight_i)

            bias_i = inputs_c
            for layer_forward_j in range(3):
                bias_i = self.biases_between[3*layer_forward_i + layer_forward_j](bias_i)
            Hyper_B_list.append(bias_i)

        Hyper_W_list.append(weight_n)
        Hyper_B_list.append(bias_n)

        return inputs[:, self.dim_c:], [Hyper_W_list, Hyper_B_list]


class HyperNN_L3(torch.nn.Module):
    """
        Defines a hyper-network that takes conditions as input and returns weights and biases for the conditional AF.
        The hyper-network is a three-layers MLP
    """
    def __init__(self, args):
        super().__init__()
        self.n_layers_flow, self.dim_h_flow, self.dim, self.dim_c, self.dev = args.n_layers_flow, args.dim_h_flow, args.dim_o, args.cond_dim, args.device

        # 1. Computes the hyperparameters (weights, biases)
        self.w1_fc1 = torch.nn.Linear(self.dim_c, self.dim_c*50)
        #self.w1_act1 = torch.nn.Tanh()
        self.w1_act1 = torch.nn.ReLU()
        self.w1_fc2 = torch.nn.Linear(self.dim_c*50, self.dim_c*1000)
        self.w1_act2 = torch.nn.ReLU()
        self.w1_fc3 = torch.nn.Linear(self.dim_c*1000, self.dim * (self.dim * self.dim_h_flow))

        self.b1_fc1 = torch.nn.Linear(self.dim_c, self.dim_c*10)
        #self.b1_act1 = torch.nn.Tanh()
        self.b1_act1 = torch.nn.ReLU()
        self.b1_fc2 = torch.nn.Linear(self.dim_c*10, self.dim_c*100)
        self.b1_act2 = torch.nn.ReLU()
        self.b1_fc3 = torch.nn.Linear(self.dim_c*100, (self.dim * self.dim_h_flow))

        self.wn_fc1 = torch.nn.Linear(self.dim_c, self.dim_c*50)
        #self.wn_act1 = torch.nn.Tanh()
        self.wn_act1 = torch.nn.ReLU()
        self.wn_fc2 = torch.nn.Linear(self.dim_c*50, self.dim_c*1000)
        self.wn_act2 = torch.nn.ReLU()
        self.wn_fc3 = torch.nn.Linear(self.dim_c*1000, (self.dim * self.dim_h_flow) * self.dim)

        self.bn_fc1 = torch.nn.Linear(self.dim_c, self.dim_c*10)
        #self.bn_act1 = torch.nn.Tanh()
        self.bn_act1 = torch.nn.ReLU()
        self.bn_fc2 = torch.nn.Linear(self.dim_c*10, self.dim_c*100)
        self.bn_act2 = torch.nn.ReLU()
        self.bn_fc3 = torch.nn.Linear(self.dim_c*100, self.dim)

        self.weights_between = torch.nn.ModuleList()
        self.biases_between = torch.nn.ModuleList()
        for _ in range(self.n_layers_flow-1):
            self.weights_between.append(torch.nn.Linear(self.dim_c, self.dim_c*50))
            #self.weights_between.append(torch.nn.Tanh())
            self.weights_between.append(torch.nn.ReLU())
            self.weights_between.append(torch.nn.Linear(self.dim_c*50, self.dim_c*2000))
            self.weights_between.append(torch.nn.ReLU())
            self.weights_between.append(torch.nn.Linear(self.dim_c*2000, (self.dim * self.dim_h_flow) * (self.dim * self.dim_h_flow)))
            
            self.biases_between.append(torch.nn.Linear(self.dim_c, self.dim_c*10))
            #self.biases_between.append(torch.nn.Tanh())
            self.biases_between.append(torch.nn.ReLU())
            self.biases_between.append(torch.nn.Linear(self.dim_c*10, self.dim_c*100))
            self.biases_between.append(torch.nn.ReLU())
            self.biases_between.append(torch.nn.Linear(self.dim_c*100, (self.dim * self.dim_h_flow)))
        

    def forward(self, inputs):
        inputs_c = inputs[:, :self.dim_c]  # Conditions
        
        weight_1 = self.w1_fc1(inputs_c)
        weight_1 = self.w1_act1(weight_1)
        weight_1 = self.w1_fc2(weight_1)
        weight_1 = self.w1_act2(weight_1)
        weight_1 = self.w1_fc3(weight_1)

        bias_1 = self.b1_fc1(inputs_c)
        bias_1 = self.b1_act1(bias_1)
        bias_1 = self.b1_fc2(bias_1)
        bias_1 = self.b1_act2(bias_1)
        bias_1 = self.b1_fc3(bias_1)

        weight_n = self.wn_fc1(inputs_c)
        weight_n = self.wn_act1(weight_n)
        weight_n = self.wn_fc2(weight_n)
        weight_n = self.wn_act2(weight_n)
        weight_n = self.wn_fc3(weight_n)

        bias_n = self.bn_fc1(inputs_c)
        bias_n = self.bn_act1(bias_n)
        bias_n = self.bn_fc2(bias_n)
        bias_n = self.bn_act2(bias_n)
        bias_n = self.bn_fc3(bias_n)

        Hyper_W_list = [weight_1]  # List of Weights : size (batch_size, in_feature * out_feature)
        Hyper_B_list = [bias_1]    # List of Biases : size (batch_size, out_feature)

        for layer_forward_i in range(self.n_layers_flow-1):
            weight_i = inputs_c
            for layer_forward_j in range(5):
                weight_i = self.weights_between[5*layer_forward_i + layer_forward_j](weight_i)
            Hyper_W_list.append(weight_i)

            bias_i = inputs_c
            for layer_forward_j in range(5):
                bias_i = self.biases_between[5*layer_forward_i + layer_forward_j](bias_i)
            Hyper_B_list.append(bias_i)

        Hyper_W_list.append(weight_n)
        Hyper_B_list.append(bias_n)

        return inputs[:, self.dim_c:], [Hyper_W_list, Hyper_B_list]