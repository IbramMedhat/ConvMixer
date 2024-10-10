import torch
import torch.nn as nn
from models.Rev_in import RevIN

class Model(nn.Module):
    """
    Patches-Conv based LTSF model
    This model is based on the patching an input time series into equal sized contexts (patches)
    Then apply a mixing-based process on 3 different dimesnsions (Channels, Inter-patches, and Intra-patches)
    The mixing is done through convolutional operators
    """
    def __init__(self, configs):
        super(Model, self).__init__()

        # Patching related parameters and layers with a set of non-overlapping patches
        self.patch_size = configs.patch_size
        self.stride = configs.stride
        self.seq_len = configs.seq_len
        self.channels = configs.enc_in
        self.rev_norm = RevIN(self.channels, configs.affine)
        self.activation = configs.activation
        self.dropout = configs.dropout
        self.hidden_size = configs.hidden_size
        
        assert self.seq_len % self.patch_size == 0, "Sequence length should be divisble patch size"
        assert self.patch_size >= self.stride, "Stride should be less than patch length to make sure of not missing completely parts of the input"

        self.num_patches = int((self.seq_len - self.patch_size) / self.stride) + 1
        # Mixing related parameters and layers
        self.mixer_block = MixerBlock(self.channels, self.num_patches, self.patch_size, self.stride, 
                                       self.activation, self.dropout, self.hidden_size)
        self.pred_len = configs.pred_len
        self.num_blocks = configs.num_blocks
        self.flatten_dimension = self.num_patches * self.patch_size
        self.projection_layer = nn.Linear(self.flatten_dimension, self.pred_len)


    def forward(self, x):
        x = self.rev_norm(x, "norm")
        x = torch.swapaxes(x, 1, 2)
        # Input to mixer block has dimensions of [batch size, channels, sequence length]
        for _ in range(self.num_blocks) :
            x = self.mixer_block(x) 
        y = torch.zeros([x.size(0), self.channels, self.pred_len],dtype=x.dtype).to(x.device)
        y = self.projection_layer(x.clone())
        y = torch.swapaxes(y, 1, 2)
        y = self.rev_norm(y, "denorm") # Finally denormalizing to finalize the RevIN operation
        return y
        
class ConvChannelMixer(nn.Module):
    """Conv block for channel mixing"""
    def __init__(self, channels, activation, dropout_factor, hidden_size):
        super(ConvChannelMixer, self).__init__()
        self.conv_layer = nn.Conv2d(channels, hidden_size, kernel_size=1)
        if activation=="gelu" :
            self.activation_layer = nn.GELU()
        elif activation=="relu" :
            self.activation_layer = nn.ReLU()
        else :
            self.activation_layer = None
        self.conv_layer2 = nn.Conv2d(hidden_size, channels, kernel_size=1)
        self.dropout_layer = nn.Dropout(dropout_factor)

    def forward(self, x) :
        # Apply mixing through conv layer
        y = self.conv_layer(x)
        if self.activation_layer is not None :
            y = self.activation_layer(y)
        y = self.conv_layer2(y)
        y = self.dropout_layer(y)
        return x + y
    
class ConvInterPatchMixer(nn.Module):
    """Conv block for inter-patch mixing"""
    def __init__(self, num_patches, activation, dropout_factor, hidden_size):
        super(ConvInterPatchMixer, self).__init__()
        self.conv_layer = nn.Conv2d(num_patches, hidden_size, kernel_size=1)        
        if activation=="gelu" :
            self.activation_layer = nn.GELU()
        elif activation=="relu" :
            self.activation_layer = nn.ReLU()
        else :
            self.activation_layer = None
        self.dropout_layer = nn.Dropout(dropout_factor)
        self.conv_layer2 = nn.Conv2d(hidden_size, num_patches, kernel_size=1)

    def forward(self, x) :
        # [batch size, num of patches, channels, patch size]
        y = self.conv_layer(x)
        if self.activation_layer is not None :
            y = self.activation_layer(y)
        y = self.dropout_layer(y)
        y = self.conv_layer2(y)
        return x + y    
    
class ConvIntraPatchMixer(nn.Module):
    """
    Conv block for Intra-patch mixing
    does the patching process as well as the patching embedding
    """
    def __init__(self, patch_size, stride, activation, dropout_factor):
        super(ConvIntraPatchMixer, self).__init__()
        # This layer is expected to do patching and inside patch mixing through conv operator
        kernel_size = (1, patch_size)
        channels_in = 1
        channels_out = patch_size
        stride = (1, stride)
        self.conv_layer = nn.Conv2d(channels_in, channels_out, kernel_size=kernel_size, stride=stride)        
        if activation=="gelu" :
            self.activation_layer = nn.GELU()
        elif activation=="relu" :
            self.activation_layer = nn.ReLU()
        else :
            self.activation_layer = None
        self.dropout_layer = nn.Dropout(dropout_factor)


    def forward(self, x) :
        # [batch_size, channels, sequence length]
        y = torch.unsqueeze(x, 1) # outputs [batch size, 1, channels, sequence length]
        y = self.conv_layer(y) # outputs [batch size, patch size, channels, num patches]
        if self.activation_layer is not None :
            y = self.activation_layer(y)
        y = self.dropout_layer(y)
        return y  

class MixerBlock(nn.Module):
    """Mixer block layer mixing over a chosen subset of {channels, inter patches, intra patches}"""
    def __init__(self, channels, num_patches, patch_size, stride, activation, dropout_factor, hidden_size) :
        super(MixerBlock, self).__init__()
        self.channels = channels
        self.patch_size = patch_size
        self.stride = stride
        self.num_patches = num_patches
        self.hidden_size = hidden_size

        self.intra_patch_mixer = ConvIntraPatchMixer(self.patch_size, self.stride, activation, dropout_factor)

        self.inter_patch_mixer = ConvInterPatchMixer(num_patches, activation, dropout_factor, self.hidden_size)
        self.normalization_layer_inter_patches = nn.BatchNorm2d(self.num_patches)
        
        self.channels_mixer    = ConvChannelMixer(self.channels, activation, dropout_factor, self.hidden_size)
        self.normalization_layer_inter_channels = nn.BatchNorm2d(self.channels)

    def forward(self, x) :
        y = self.intra_patch_mixer(x) # outputs [batch size, patch size, channels, number of patches]
        y = torch.swapaxes(y, 1, 3) # to [batch size, number of patches, channels, patch size]
        y = self.normalization_layer_inter_patches(y)
        y = self.inter_patch_mixer(y) 
        y = torch.swapaxes(y, 1, 2) # to [batch size, channels, number of patches, patch size]
        y = self.normalization_layer_inter_channels(y)
        y = self.channels_mixer(y)
        y = torch.reshape(y, 
                          (y.size(0),
                            self.channels, 
                           self.num_patches*self.patch_size)) # Collapse to [batch_size, channels, sequence length]
        return y