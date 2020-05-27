"""
# Author      : GS Oh
# Experiment  : PRECOG_Carla
# Note        : Definitions of Encoders (ConvBlock (w residual connection), CoordConvBlock (w residual connection)) for PRECOG_Carla exp, block modules were modified from https://github.com/rajatvd/AutoencoderAnim
"""
import torch
from torch import nn
from torch import optim

import numpy as np
import os


# Encoder Base Module 1 : ConvBlock
class ConvBlock(nn.Module):
    """
    A block of 2d convolutions with residual connections
    5 different normalization layers used: BN, GN, GN2, LN, IN
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, norm_method='BN'):
        super().__init__()
        
        assert kernel_size%2==1, "kernel_size should be odd number"
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding = kernel_size//2)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding = kernel_size//2)

        if norm_method == 'BN':
            self.bn1_1 = nn.BatchNorm2d(in_channels)
            self.bn1_2 = nn.BatchNorm2d(out_channels)
            self.bn2_1 = nn.BatchNorm2d(out_channels)
            self.bn2_2 = nn.BatchNorm2d(out_channels)
        elif norm_method == 'GN':
            # the first arg of GroupNorm (num_groups should be able to divide the channel.)
            num_groups = max(out_channels // 4, 1)
            self.bn1_1 = nn.GroupNorm(in_channels, in_channels)
            self.bn1_2 = nn.GroupNorm(num_groups, out_channels)
            self.bn2_1 = nn.GroupNorm(num_groups, out_channels)
            self.bn2_2 = nn.GroupNorm(num_groups, out_channels)
        elif norm_method == 'GN2':
            # the first arg of GroupNorm (num_groups should be able to divide the channel.)
            num_groups = max(out_channels // 8, 1)
            self.bn1_1 = nn.GroupNorm(1, in_channels)
            self.bn1_2 = nn.GroupNorm(num_groups, out_channels)
            self.bn2_1 = nn.GroupNorm(num_groups, out_channels)
            self.bn2_2 = nn.GroupNorm(num_groups, out_channels)
        elif norm_method == 'IN':
            self.bn1_1 = nn.GroupNorm(in_channels, in_channels)
            self.bn1_2 = nn.GroupNorm(out_channels, out_channels)
            self.bn2_1 = nn.GroupNorm(out_channels, out_channels)
            self.bn2_2 = nn.GroupNorm(out_channels, out_channels)
        elif norm_method == 'LN':
            self.bn1_1 = nn.GroupNorm(1, in_channels)
            self.bn1_2 = nn.GroupNorm(1, out_channels)
            self.bn2_1 = nn.GroupNorm(1, out_channels)
            self.bn2_2 = nn.GroupNorm(1, out_channels)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        if self.out_channels != self.in_channels:
            self.res_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if norm_method == 'BN':
                self.res_bn = nn.BatchNorm2d(out_channels)
            elif norm_method in ['GN', 'GN2']:
                self.res_bn = nn.GroupNorm(num_groups, out_channels)
            elif norm_method == 'IN':
                self.res_bn = nn.GroupNorm(out_channels, out_channels)
            elif norm_method == 'LN':
                self.res_bn = nn.GroupNorm(1, out_channels)
        
    def forward(self, x):
        residual = self.bn1_1(x)
        
        out = self.conv1(x)
        out = self.bn1_2(out)
        out = nn.ReLU()(out)
        
        out = self.bn2_1(out)
        out = self.conv2(out)
        out = self.bn2_2(out)
        
        if self.out_channels != self.in_channels:
            residual = self.res_conv(residual)
            residual = self.res_bn(residual)
        
        out = nn.ReLU()(out+residual)
            
        return out


# Coordinate Convolution
class CoordConv(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=0):
        super().__init__()
        
        self.conv = nn.Conv2d(in_channels+2, out_channels, kernel_size=kernel_size, padding=padding)
        
    def forward(self, input):
        h = input.shape[2]
        w = input.shape[3]
        
        i_linspace = np.linspace(-1,1,h)
        j_linspace = np.linspace(-1,1,w)
        
        ii, jj = np.meshgrid(i_linspace, j_linspace)
        ii = torch.tensor(ii, dtype=input.dtype).to(input.device).repeat((input.shape[0],1,1,1))
        jj = torch.tensor(jj, dtype=input.dtype).to(input.device).repeat((input.shape[0],1,1,1))
        
        inp = torch.cat([input, ii, jj], dim=1)
        
        out = self.conv(inp)
        
        return out


# Encoder Base Module 2 : CoordConvBlock
class CoordConvBlock(nn.Module):
    """
    A block of 2d coordinate convolutions with residual connections
    5 different normalization layers used: BN, GN, GN2, LN, IN
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, norm_method='BN'):
        super().__init__()
        
        assert kernel_size%2==1, "kernel_size should be odd number"
        
        self.cconv1 = CoordConv(in_channels, out_channels, kernel_size, padding = kernel_size//2)
        self.cconv2 = CoordConv(out_channels, out_channels, kernel_size, padding = kernel_size//2)
    
        if norm_method == 'BN':
            self.bn1_1 = nn.BatchNorm2d(in_channels)
            self.bn1_2 = nn.BatchNorm2d(out_channels)
            self.bn2_1 = nn.BatchNorm2d(out_channels)
            self.bn2_2 = nn.BatchNorm2d(out_channels)
        elif norm_method == 'GN':
            # the first arg of GroupNorm (num_groups should be able to divide the channel.)
            num_groups = max(out_channels // 4, 1)
            self.bn1_1 = nn.GroupNorm(in_channels, in_channels)
            self.bn1_2 = nn.GroupNorm(num_groups, out_channels)
            self.bn2_1 = nn.GroupNorm(num_groups, out_channels)
            self.bn2_2 = nn.GroupNorm(num_groups, out_channels)
        elif norm_method == 'GN2':
            # the first arg of GroupNorm (num_groups should be able to divide the channel.)
            num_groups = max(out_channels // 8, 1)
            self.bn1_1 = nn.GroupNorm(1, in_channels)
            self.bn1_2 = nn.GroupNorm(num_groups, out_channels)
            self.bn2_1 = nn.GroupNorm(num_groups, out_channels)
            self.bn2_2 = nn.GroupNorm(num_groups, out_channels)
        elif norm_method == 'IN':
            self.bn1_1 = nn.GroupNorm(in_channels, in_channels)
            self.bn1_2 = nn.GroupNorm(out_channels, out_channels)
            self.bn2_1 = nn.GroupNorm(out_channels, out_channels)
            self.bn2_2 = nn.GroupNorm(out_channels, out_channels)
        elif norm_method == 'LN':
            self.bn1_1 = nn.GroupNorm(1, in_channels)
            self.bn1_2 = nn.GroupNorm(1, out_channels)
            self.bn2_1 = nn.GroupNorm(1, out_channels)
            self.bn2_2 = nn.GroupNorm(1, out_channels)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        if self.out_channels != self.in_channels:
            self.res_cconv = CoordConv(in_channels, out_channels, kernel_size=1)
            if norm_method == 'BN':
                self.res_bn = nn.BatchNorm2d(out_channels)
            elif norm_method in ['GN', 'GN2']:
                self.res_bn = nn.GroupNorm(num_groups, out_channels)
            elif norm_method == 'IN':
                self.res_bn = nn.GroupNorm(out_channels, out_channels)
            elif norm_method == 'LN':
                self.res_bn = nn.GroupNorm(1, out_channels)
        
    def forward(self, x):
        residual = self.bn1_1(x)
        
        out = self.cconv1(x)
        out = self.bn1_2(out)
        out = nn.ReLU()(out)
        
        out = self.bn2_1(out)
        out = self.cconv2(out)
        out = self.bn2_2(out)
        
        if self.out_channels != self.in_channels:
            residual = self.res_cconv(residual)
            residual = self.res_bn(residual)
        
        out = nn.ReLU()(out+residual)
            
        return out


class BlockSet(nn.Module):
    """
    Cascades a set of given blocks. The first block maps in_channels to 
    out_channels, and remaining blocks map out_channels to out_channels.
    """
    def __init__(self, block, in_channels, out_channels, block_count, kernel_size=3, norm_method='BN'):
        super().__init__()
        
        block1 = block(in_channels, out_channels, kernel_size=kernel_size, norm_method=norm_method)
        blocks = [block(out_channels, out_channels, kernel_size=kernel_size, norm_method=norm_method) for _ in range(block_count-1)]
        
        self.blocks = nn.Sequential(block1, *blocks)
            
    def forward(self, input):
        out = self.blocks(input)
        #print("is blockset training?:", self.training)
        #print("blockset output:", out.squeeze())
        
        return out

class BlockNet(nn.Module):
    """
    Cascades multiple BlockSets to form a complete network. One BlockSet is used for each element in
    the channel_sequence, and size scaling is done between blocks based on size_sequence. 
    A decrease in size is done using fractional max pooling.
    An increase in size is done by bilinear upsampling.
    
    A final 1x1 block is added after all the BlockSets.
    """
    def __init__(self, block, channel_sequence, size_sequence, block_count, kernel_size=3, norm_method='BN', use_block_for_last=False, pool_layer='fmp'):
        super().__init__()
        
        assert len(channel_sequence)==len(size_sequence), "channel and size sequences should have same length"
        
        old_channels, old_size = channel_sequence[0], size_sequence[0]
        
        layers = []
        for channels, size in zip(channel_sequence[1:], size_sequence[1:]):
            layers.append(BlockSet(block, 
                                   in_channels=old_channels,
                                   out_channels=channels,
                                   block_count=block_count,
                                   kernel_size=kernel_size, 
                                   norm_method=norm_method))
            if size<old_size:
                if pool_layer == 'fmp': # Fractional Max Pooling
                    layers.append(nn.FractionalMaxPool2d(kernel_size=kernel_size, output_size=size))
                elif pool_layer == 'mp': # Max Pooling, padding = 0, stride := kernel_size, dilation = 1, 
                    layers.append(nn.MaxPool2d(kernel_size=old_size//size))
            elif size>old_size:
                layers.append(nn.Upsample(size=(size,size), mode='bilinear', align_corners=True))
            
            old_channels, old_size = channels, size
        
        if use_block_for_last:
            layers.append(block(channels, channels, kernel_size=1, norm_method=norm_method))
        else:
            layers.append(nn.Conv2d(channels, channels, kernel_size=1))
        
        self.layers = nn.Sequential(*layers)
            
    def forward(self, input):
        out = self.layers(input)
        # print("is blockNet training?:", self.training)
        # print("blockNet output:", out.squeeze())
        return out


class NormalizeModule(nn.Module):
    """Returns (input-mean)/std"""
    def __init__(self, mean, std):
        super().__init__()
        self.mean = mean
        self.std = std
        
    def forward(self, input):
        return (input-self.mean)/self.std


def define_enc_n_save_spec(DEVICE, ModelName, SAVE_DIR, **kwargs):
    print("Device used :", DEVICE)

    # Save specs for encoder
    encoder_spec = "[Encoder specification]"
    encoder_spec += ''.join((
    "\nchannel_sequence_enc : ", repr(kwargs["channel_seq_enc"]),
    "\nsize_sequence_enc : ", repr(kwargs["size_seq_enc"]),
    "\nblock_count_enc : ", repr(kwargs["block_count_enc"]),
    "\nkernel_size_enc : ", repr(kwargs["kernel_size_enc"]),
    "\nnormalization_method : ", repr(kwargs["norm_method"]),
    "\nimg_size : ", repr(kwargs["img_size"]),
    "\npooling_method : ", repr(kwargs["pool_layer"])
    ))

    # Define encoder
    if ModelName=="ConvBlock":   
        encoder = BlockNet(ConvBlock, 
            channel_sequence=kwargs["channel_seq_enc"], size_sequence=kwargs["size_seq_enc"], 
            block_count=kwargs["block_count_enc"], kernel_size=kwargs["kernel_size_enc"], 
            norm_method=kwargs["norm_method"], use_block_for_last=kwargs["use_block_for_last_enc"], pool_layer=kwargs["pool_layer"]).to(DEVICE)

        encoder_spec += ''.join(("\nuse_block_for_last_enc : ", repr(kwargs["use_block_for_last_enc"])))

    elif ModelName=="CoordConvBlock":
        encoder = nn.Sequential(
            NormalizeModule(0.5,0.5),
            BlockNet(CoordConvBlock,  
                channel_sequence=kwargs["channel_seq_enc"], size_sequence=kwargs["size_seq_enc"], 
                block_count=kwargs["block_count_enc"], kernel_size=kwargs["kernel_size_enc"], 
                norm_method=kwargs["norm_method"], pool_layer=kwargs["pool_layer"])).to(DEVICE)

    encp = nn.utils.parameters_to_vector(encoder.parameters()).shape[0]
    print("Encoder has {} params".format(encp))

    SAVE_DIR_enc_spec = os.path.join(SAVE_DIR, "Encoder_spec_summary.txt")
    with open(SAVE_DIR_enc_spec, 'w') as f:
        f.write(encoder_spec)

    return encoder



####################### Coordconv Models with lidar (2 channels), HW200 ##########################

def coordconv_E32_HW200_lidar_simple(DEVICE, SAVE_DIR, norm_method='BN', pool_layer='fmp'):
    channel_seq_enc=[2,8,16,32]
    size_seq_enc=[200,100,25,1]
    block_count_enc=3
    kernel_size_enc=3

    ModelName = 'CoordConvBlock'
    kwargs = {"channel_seq_enc":channel_seq_enc, "size_seq_enc":size_seq_enc, "block_count_enc":block_count_enc, "kernel_size_enc":kernel_size_enc, \
                "norm_method":norm_method, "img_size":200, 'pool_layer':pool_layer}
                
    # Define encoder & Save specs for encoder and decoder as a txt file
    encoder = define_enc_n_save_spec(DEVICE, ModelName, SAVE_DIR, **kwargs)
    
    return encoder

def coordconv_E64_HW200_lidar_simple(DEVICE, SAVE_DIR, norm_method='BN', pool_layer='fmp'):
    channel_seq_enc=[2,8,32,64]
    size_seq_enc=[200,50,10,1]
    block_count_enc=3
    kernel_size_enc=3

    ModelName = 'CoordConvBlock'
    kwargs = {"channel_seq_enc":channel_seq_enc, "size_seq_enc":size_seq_enc, "block_count_enc":block_count_enc, "kernel_size_enc":kernel_size_enc, \
                "norm_method":norm_method, "img_size":200, 'pool_layer':pool_layer}
                
    # Define encoder & Save specs for encoder and decoder as a txt file
    encoder = define_enc_n_save_spec(DEVICE, ModelName, SAVE_DIR, **kwargs)
    
    return encoder

def coordconv_E64_HW200_lidar_complex(DEVICE, SAVE_DIR, norm_method='BN', pool_layer='fmp'):
    channel_seq_enc=[2,8,16,32,32,64,64]
    size_seq_enc=[200,100,50,50,20,5,1]
    block_count_enc=3
    kernel_size_enc=3

    ModelName = 'CoordConvBlock'
    kwargs = {"channel_seq_enc":channel_seq_enc, "size_seq_enc":size_seq_enc, "block_count_enc":block_count_enc, "kernel_size_enc":kernel_size_enc, \
                "norm_method":norm_method, "img_size":200, 'pool_layer':pool_layer}
                
    # Define encoder & Save specs for encoder and decoder as a txt file
    encoder = define_enc_n_save_spec(DEVICE, ModelName, SAVE_DIR, **kwargs)
    
    return encoder

def coordconv_E96_HW200_lidar_simple(DEVICE, SAVE_DIR, norm_method='BN', pool_layer='fmp'):
    channel_seq_enc=[2,8,32,64,96]
    size_seq_enc=[200,50,20,5,1]
    block_count_enc=3
    kernel_size_enc=3

    ModelName = 'CoordConvBlock'
    kwargs = {"channel_seq_enc":channel_seq_enc, "size_seq_enc":size_seq_enc, "block_count_enc":block_count_enc, "kernel_size_enc":kernel_size_enc, \
                "norm_method":norm_method, "img_size":200, 'pool_layer':pool_layer}
                
    # Define encoder & Save specs for encoder and decoder as a txt file
    encoder = define_enc_n_save_spec(DEVICE, ModelName, SAVE_DIR, **kwargs)
    
    return encoder

