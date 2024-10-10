import torch
import torch.nn as nn

from models.Rev_in import RevIN

class Model(nn.Module):
    """
    Patches-TSMixer based LTSF model mixing across channels and patches
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        # Patching related parameters and layers
        self.patch_size = configs.patch_size
        self.seq_len = configs.seq_len
        self.channels = configs.enc_in
        self.stride = configs.patch_size
        self.hidden_size = configs.hidden_size
        self.activation = configs.activation
        self.dropout = configs.dropout
        self.excluded_component = configs.excluded_component


        # assert self.seq_len % self.patch_size == 0, "Sequence length should be divisble patch size"

        self.num_patches = int(self.seq_len/self.stride)
        self.rev_norm = RevIN(self.channels, configs.affine)

        # Mixing related parameters and layers
        self.mixer_block = MixerBlock(self.channels, self.hidden_size, self.num_patches, self.patch_size,
                                       self.activation, self.dropout, self.excluded_component)
        self.pred_len = configs.pred_len
        self.num_blocks = configs.num_blocks
        self.output_linear_layers = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x):
        #  x : [batch size, channels, timesteps]
        x = self.rev_norm(x, "norm")
        x = torch.swapaxes(x, 1, 2)
        # Patching
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride) # reshaping to [batch size, channels, num_patches, patch_size]
        for _ in range(self.num_blocks) :
            x = self.mixer_block(x)
        x = torch.reshape(x, 
                          (x.size(0),
                            self.channels,
                            self.num_patches*self.patch_size)) # Collapsing patches together to shape [batch size, channels, sequence len]
        # Preparing tensor output with the correct prediction length
        # Output tensor shape of [batch size, channel, pred_len]
        y = torch.zeros([x.size(0), self.channels, self.pred_len],dtype=x.dtype).to(x.device)
        y = self.output_linear_layers(x.clone())
        y = torch.swapaxes(y, 1, 2)
        y = self.rev_norm(y, "denorm")
        return y
        
class MlpBlockFeatures(nn.Module):
    """MLP for features"""
    def __init__(self, channels, hidden_size, activation, dropout_factor):
        super(MlpBlockFeatures, self).__init__()
        self.linear_layer = nn.Linear(channels, hidden_size)
        self.dropout_layer = nn.Dropout(dropout_factor)
        if activation=="gelu" :
            self.activation_layer = nn.GELU()
        elif activation=="relu" :
            self.activation_layer = nn.ReLU()
        else :
            self.activation_layer = None
        self.linear_layer2 = nn.Linear(hidden_size, channels)

    def forward(self, x) :
        # x : [batch_size, channels, num of patches, patch size]
        y = self.linear_layer(x)
        if self.activation_layer is not None :
            y = self.activation_layer(y)
        y = self.linear_layer2(y)
        y = self.dropout_layer(y)
        return x + y
    
class MlpBlockPatches(nn.Module):
    """MLP for patches"""
    def __init__(self, num_patches, hidden_size, activation, dropout_factor):
        super(MlpBlockPatches, self).__init__()
        self.linear_layer = nn.Linear(num_patches, hidden_size)
        self.dropout_layer = nn.Dropout(dropout_factor)
        if activation=="gelu" :
            self.activation_layer = nn.GELU()
        elif activation=="relu" :
            self.activation_layer = nn.ReLU()
        else :
            self.activation_layer = None
        self.linear_layer2 = nn.Linear(hidden_size, num_patches)


    def forward(self, x) :
        # [batch_size, num of patches, channels, patch size]
        y = self.linear_layer(x)
        if self.activation_layer is not None :
            y = self.activation_layer(y)
        y = self.dropout_layer(y)
        y = self.linear_layer2(y)
        return x + y    
    
class MlpBlockPatchSize(nn.Module):
    """MLP for num_patches"""
    def __init__(self, patch_size, activation, dropout_factor):
        super(MlpBlockPatchSize, self).__init__()
        self.linear_layer = nn.Linear(patch_size, patch_size)
        self.dropout_layer = nn.Dropout(dropout_factor)
        if activation=="gelu" :
            self.activation_layer = nn.GELU()
        elif activation=="relu" :
            self.activation_layer = nn.ReLU()
        else :
            self.activation_layer = None


    def forward(self, x) :
        # [batch_size, patch size, channels, num of patches]
        y = self.linear_layer(x)
        if self.activation_layer is not None :
            y = self.activation_layer(y)
        y = self.dropout_layer(y)
        return y  

class MixerBlock(nn.Module):
    """Mixer block layer only mixing channels in this model"""
    def __init__(self, channels,hidden_size, num_patches, patch_size, activation, dropout_factor, excluded_component) :
        super(MixerBlock, self).__init__()

        self.channels = channels
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.hidden_size = hidden_size
        self.activation = activation
        self.dropout_factor = dropout_factor
        self.excluded_component = excluded_component

        self.intra_patch_mixer = MlpBlockPatchSize(self.patch_size, self.activation, self.dropout_factor)
    
        self.normalization_layer_inter_patches = nn.BatchNorm2d(self.num_patches)
        self.inter_patch_mixer = MlpBlockPatches(self.num_patches, self.hidden_size, self.activation, self.dropout_factor)
    
        self.normalization_layer_inter_channels = nn.BatchNorm2d(self.channels)    
        self.channels_mixer = MlpBlockFeatures(self.channels, self.hidden_size, self.activation, self.dropout_factor)
        
    def forward(self, x) :

        # Excluding intra patch mixing component
        if (self.excluded_component == 1) :
            y = torch.swapaxes(x, 1, 2) # to [batch size, number of patches, channels, patch size]
            y = self.normalization_layer_inter_patches(y)
            y = torch.swapaxes(y, 1, 3) # to [batch size, patch size, channels, number of patches]
            y = self.inter_patch_mixer(y)
            y = torch.swapaxes(y, 1, 2) # to [batch size, channels, patch size, number of patches]
            y = self.normalization_layer_inter_channels(y)
            y = torch.swapaxes(y, 1, 3) # to [batch size, number of patches, patch size, channels]
            y = self.channels_mixer(y)
            y = torch.swapaxes(y, 2, 3)
            y = torch.swapaxes(y, 1, 2) # reshape to [batch_size, channels, number of patches, patch size]

        # Excluding inter patch mixing component
        elif (self.excluded_component == 2) :
            y = self.intra_patch_mixer(x) # outputs [batch size, channels, number of patches, patch size]
            y = self.normalization_layer_inter_channels(y)
            y = torch.swapaxes(y, 1, 3) # to [batch size, patch size, number of patches, channels]
            y = self.channels_mixer(y)
            y = torch.swapaxes(y, 1, 3) # reshape to [batch_size, channels, number of patches, patch size]

        # Excluding inter channel mixing (channel independence assumption)
        elif (self.excluded_component == 3) :
            y = self.intra_patch_mixer(x) # outputs [batch size, channels, number of patches, patch size]
            y = torch.swapaxes(y, 1, 2) # to [batch size, number of patches, channels, patch size]
            y = self.normalization_layer_inter_patches(y)
            y = torch.swapaxes(y, 1, 3) # to [batch size, patch size, channels, number of patches]
            y = self.inter_patch_mixer(y)
            y = torch.swapaxes(y, 2, 3) # reshape to [batch_size, patch size, number of patches, channels]
            y = torch.swapaxes(y, 1, 3) # reshape to [batch_size, channels, number of patches, patch size]
        
        # Original model includes all components
        else :
            y = self.intra_patch_mixer(x) # outputs [batch size, channels, number of patches, patch size]
            y = torch.swapaxes(y, 1, 2) # to [batch size, number of patches, channels, patch size]
            y = self.normalization_layer_inter_patches(y)
            y = torch.swapaxes(y, 1, 3) # to [batch size, patch size, channels, number of patches]
            y = self.inter_patch_mixer(y)
            y = torch.swapaxes(y, 1, 2) # to [batch size, channels, patch size, number of patches]
            y = self.normalization_layer_inter_channels(y)
            y = torch.swapaxes(y, 1, 3) # to [batch size, number of patches, patch size, channels]
            y = self.channels_mixer(y)
            y = torch.swapaxes(y, 2, 3)
            y = torch.swapaxes(y, 1, 2) # reshape to [batch_size, channels, number of patches, patch size]
        return y

