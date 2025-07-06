import torch
import torch.nn as nn
import math
from mmcv.cnn import build_norm_layer
from mmengine.model.weight_init import trunc_normal_
from mmdet.registry import MODELS
from .attention import PositionalEncoding, TransformerEncoder, EncoderBlock


@MODELS.register_module()
class TemporalTransformer(nn.Module):
    def __init__(self, 
                 input_dim=128, 
                 model_dim=128, 
                 num_heads=4, 
                 num_layers=4, 
                 dropout=0.0,
                 input_dropout=0.0
                 ):
        super(TemporalTransformer, self).__init__()
       
        # Input dim -> Model dim
        self.input_net = nn.Sequential(
            nn.Dropout(input_dropout),
            nn.Linear(input_dim, model_dim)
        )
        # Positional encoding for sequences
        self.positional_encoding = PositionalEncoding(d_model=model_dim)
        # Transformer
        self.transformer = TransformerEncoder(num_layers=num_layers,
                                              input_dim=model_dim,
                                              dim_feedforward=2 * model_dim,
                                              num_heads=num_heads,
                                              dropout=dropout)
        

    def forward(self, x, mask=None, spatial_shape=None, add_positional_encoding=True):
        """
        Inputs:
            x - Input features of shape [Batch, SeqLen, input_dim]
            mask - Mask to apply on the attention outputs (optional)
            add_positional_encoding - If True, we add the positional encoding to the input.
                                      Might not be desired for some tasks.
        """
        x = self.input_net(x)
        if add_positional_encoding:
            x = self.positional_encoding(x)
        x = self.transformer(x, mask=mask)
        x = x[:, (x.shape[1]-1)//2, :]
        # print("transformer output shape: {}.".format(x.shape))
        # x = self.output_net(x)
        return x
        

@MODELS.register_module()
class TimeSformerTransformer(nn.Module):
    """TimeSformer with exact same interface as TemporalTransformer but using divided space-time attention."""
    
    def __init__(self,
                 input_dim=128, 
                 model_dim=128, 
                 num_heads=4, 
                 num_layers=4,
                 dropout=0.0,
                 input_dropout=0.0):
        super().__init__()
        
        # Input projection 
        self.input_net = nn.Sequential(
            nn.Dropout(input_dropout),
            nn.Linear(input_dim, model_dim)
        )
        
        # Positional encoding 
        self.positional_encoding = PositionalEncoding(d_model=model_dim)
        
        # Divided space-time attention layers
        self.space_layers = nn.ModuleList([
            EncoderBlock(
                input_dim=model_dim,
                num_heads=num_heads,
                dim_feedforward=2*model_dim,
                dropout=dropout
            ) for _ in range(num_layers)
        ])
        
        self.time_layers = nn.ModuleList([
            EncoderBlock(
                input_dim=model_dim,
                num_heads=num_heads,
                dim_feedforward=2*model_dim,
                dropout=dropout
            ) for _ in range(num_layers)
        ])

    def forward(self, x, mask=None, spatial_shape=None, add_positional_encoding=True):
        """
        Identical interface to TemporalTransformer:
        Input: [Batch, SeqLen, input_dim]
        Output: [Batch, model_dim] (features from center frame)
        """
        # Apply input projection and positional encoding
        x = self.input_net(x)
        if add_positional_encoding:
            x = self.positional_encoding(x)
        
        BHW, T, D = x.shape
        
        # Get spatial dimensions
        if spatial_shape is None:
            # Auto-detect square dimensions (fallback - shouldn't happen!)
            HW = BHW // B  
            H = W = int(HW**0.5)
        else:
            H, W = spatial_shape  # Explicit shape from FPN (correct path)
            

        B = BHW // (H * W)
        x = x.view(B, H, W, T, D)
        
        # Process through divided attention layers
        for space_layer, time_layer in zip(self.space_layers, self.time_layers):
            # Spatial attention (within each frame)
            x_space = x.reshape(B*T, H*W, D)
            x_space = space_layer(x_space, mask=mask)
            x_space = x_space.view(B, T, H, W, D)
            
            # Temporal attention (within each spatial location)
            x_time = x.reshape(B*H*W, T, D)
            x_time = time_layer(x_time, mask=mask)
            x_time = x_time.view(B, H, W, T, D)
            x_time = x_time.permute(0, 3, 1, 2, 4)  # [B, H, W, T, D] -> [B, T, H, W, D]
            
            # Combine with residual
            x = x_space + x_time
        
        # Take center frame's features (no pooling)
        center_idx = (x.shape[3]-1)//2  # Temporal center
        x = x[:, :, :, center_idx, :]   # [B, H, W, D]
        
        return x
