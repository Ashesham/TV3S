import os
import random
import warnings
from functools import partial
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from einops import rearrange
from IPython import embed
from timm.models.layers import DropPath

from mmcv.cnn import ConvModule
from mmseg.ops import resize
from mmseg.models.builder import HEADS
from mmseg.models.utils import *
from .decode_head import BaseDecodeHead_clips_flow, BaseDecodeHead_clips_flow_city

from mamba_ssm import Mamba
try:
    from mamba_ssm import Mamba2
except ImportError:
    Mamba2 = None
    warnings.warn("Mamba2 not available, hence is deactivated")

from mamba_ssm.utils.generation import InferenceParams
try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn

'''Model Types and Validation Types for Mamba'''
class Model_type: 
    Normal = 0          # Normal model with no bi and no add/concat as default
    Bi = 1              # Bi-directional model with concat
    Bi_Embed = 2        # Bi-directional model with embedding
    Add = 3             # Normal model with add
    Concat = 4          # Normal model with concat
    Alt_Residual = 5    # Perform alternate residual connection all going to even blocks

class Val_type:
    '''Note >0 = mamba mode [1Frame Processing]'''
    Normal = 0      # Normal validation = 4 Frame no sharing 
    Frame_1 = 1     # 1 Frame, no sharing
    Frame_1_greedy_seq = 2 # 1 Frame, sharing of states but token-by-token processing
    Frame_1_greedy = 3 # 1 Frame, sharing of states and frame-by-frame processing (After CUDA Changes)
    Frame_4_seq = -1 # 4 Frame, sharing of states only across the 4 frames but token-by-token processing
    Frame_n_seq = Frame_4_seq # The same as Frame_4_seq and is generalizable.
    
torch.tensor([10]).stride()
'''Helper Classes'''

class MLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x

class mambaSequential(nn.Sequential):
    # Allow for multiple inputs in sequential model
    def forward(self, x, inference_params=None):
        for module in self._modules.values():
            inputs = module(x, inference_params=inference_params)
        return inputs

class VideoMambaBlock(nn.Module):
    '''
        Taken from: https://github.com/OpenGVLab/VideoMamba/blob/main/videomamba/image_sm/models/videomamba.py class Block()
    '''
    def __init__(
        self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False,drop_path=0.,
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(
        self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        if not self.fused_add_norm:
            residual = (residual + self.drop_path(hidden_states)) if residual is not None else hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            hidden_states, residual = fused_add_norm_fn(
                hidden_states if residual is None else self.drop_path(hidden_states),
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
            )
        hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
    

def create_block(
    d_model,
    ssm_cfg=None,
    norm_epsilon=1e-5,
    drop_path=0.,
    rms_norm=True,
    residual_in_fp32=True,
    fused_add_norm=True,
    layer_idx=None,
    bimamba=True,
    mamba2=False,
    device=None,
    dtype=None,
):
    '''
        Taken from: https://github.com/OpenGVLab/VideoMamba/blob/main/videomamba/image_sm/models/videomamba.py
    '''

    factory_kwargs = {"device": device, "dtype": dtype}
    if ssm_cfg is None:
        ssm_cfg = {}
    # import ipdb;ipdb.set_trace()
    if mamba2:
        mixer_cls = partial(Mamba2, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
    elif bimamba:
        mixer_cls = partial(Mamba, layer_idx=layer_idx, bimamba=bimamba, **ssm_cfg, **factory_kwargs)
    else:
        mixer_cls = partial(Mamba, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
    
    norm_cls = partial(nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon)
    block = VideoMambaBlock(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        drop_path=drop_path,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block

class MambaBlocks(nn.Module):

    def __init__(   self,
                    embed_dim = 192,
                    ssm_cfg=None,
                    norm_epsilon=1e-5,
                    rms_norm=True,
                    depth=24,
                    fused_add_norm=True,
                    residual_in_fp32=True,
                    bimamba=True,
                    mamba2=False,
                    drop_path_rate=0.1,
                    device=None,
                    dtype=None):
        
        super().__init__()

        factory_kwargs = {"device": device, "dtype": dtype}

        self.fused_add_norm = fused_add_norm
        self.residual_in_fp32 = residual_in_fp32

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        inter_dpr = [0.0] + dpr
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        # mamba blocks
        self.layers = nn.ModuleList(
            [
                create_block(
                    embed_dim,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    bimamba=bimamba,
                    mamba2 = mamba2,
                    drop_path=inter_dpr[i],
                    **factory_kwargs,
                )
                for i in range(depth)
            ]
        )

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(embed_dim, eps=norm_epsilon, **factory_kwargs)

    
    def forward(self, x, inference_params=None):
        # mamba impl
        residual = None
        hidden_states = x
        for layer in self.layers:
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=inference_params
            )

        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                self.drop_path(hidden_states),
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )

        return hidden_states

class ShiftedMambaBlocks(nn.Module):

    def __init__(   self,
                    embed_dim = 192,
                    ssm_cfg=None,
                    norm_epsilon=1e-5,
                    rms_norm=True,
                    depth=24,
                    fused_add_norm=True,
                    residual_in_fp32=True,
                    bimamba=False,
                    mamba2=False,
                    window_w1=30,
                    window_h1=30,
                    window_w2=30,
                    window_h2=30,
                    shift_size=None,
                    real_shift=False,
                    drop_path_rate=0.1,
                    device=None,
                    dtype=None,
                    val_mode=0,
                    model_mode=Model_type.Normal,
                    zigzag=False,
                    quad_scan=False):
        
        super().__init__()

        factory_kwargs = {"device": device, "dtype": dtype}

        self.fused_add_norm = fused_add_norm
        self.residual_in_fp32 = residual_in_fp32

        # Shifted mamba parameters
        self.embed_dim = embed_dim
        self.window_w1 = window_w1
        self.window_h1 = window_h1
        self.window_w2 = window_w2
        self.window_h2 = window_h2
        self.shift_size = shift_size
        self.real_shift = real_shift
        self.val_mode = val_mode
        self.model_mode = model_mode

        if model_mode==Model_type.Alt_Residual:
            print("*"*20, "Alternate Residual Connection", "*"*20)

        # Check the shift size if it's greater than the window size for real shift
        if self.real_shift and self.shift_size>=self.window_h1:
            # Don't Support as it doesn't make sense
            raise ValueError("Real shift with shift size greater than window size is not supported")
            assert self.window_h1 == self.window_h2 and self.window_w1 == self.window_w2, "Real shift with shift more than window size is only supported for same window size"
            self.shift_size = self.window_h1
            warnings.warn("Real shift is enabled, make sure the shift size is less than the window size, currently the shift size is set to window size: {}".format(self.shift_size))


        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        inter_dpr = [0.0] + dpr
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        # mamba blocks
        self.layers = nn.ModuleList(
            [
                create_block(
                    embed_dim,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    bimamba=bimamba,
                    mamba2 = mamba2,
                    drop_path=inter_dpr[i],
                    **factory_kwargs,
                )
                for i in range(depth)
            ]
        )

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(embed_dim, eps=norm_epsilon, **factory_kwargs)
        
        self.is_zigzag = zigzag
        if self.is_zigzag:
            assert self.window_h1 == self.window_h2 and self.window_w1 == self.window_w2, "Zigzag only works with same window size"
            self.zigzag_indices = {}
            self.zigzag_indices[str(self.window_h1)+'_'+str(self.window_w1)] = self.zigzag_indices_generator(self.window_h1, self.window_w1)
        else: 
            self.zigzag_indices = None

        self.is_quad_scan = quad_scan
        

    def zigzag_indices_generator(self, height, width):
        """Generate zigzag order indices for an (H, W) matrix."""
        indices = np.empty((height * width, 2), dtype=int)
        idx = 0

        for s in range(height + width - 1):
            if s % 2 == 0:
                # Even diagonal: top-right to bottom-left
                for i in range(max(0, s - width + 1), min(height, s + 1)):
                    indices[idx] = [i, s - i]
                    idx += 1
            else:
                # Odd diagonal: bottom-left to top-right
                for i in range(max(0, s - width + 1), min(height, s + 1))[::-1]:
                    indices[idx] = [i, s - i]
                    idx += 1
        return indices 

    def flatten(self, x):

        if not self.is_zigzag:
            x = rearrange(x, 'b t c i h j w -> (b i j) (t h w) c')
        else: 
            _, _, _, _, h, _, w = x.shape
            if f"{h}_{w}" not in self.zigzag_indices:
                self.zigzag_indices[f"{h}_{w}"] = self.zigzag_indices_generator(h, w)
            
            zigzag_indices = self.zigzag_indices[f"{h}_{w}"]
            x = rearrange(x, 'b t c i h j w -> (b i j) t c h w')
            x = x[:, :, :, zigzag_indices[:, 0], zigzag_indices[:, 1]]
            x = rearrange(x, 'b t c s -> b (t s) c', s=x.shape[-1])
        return x
    
    def unflatten(self, x, i, j, h=None, w=None):
        if h is None: 
            h = self.window_h1
        if w is None:
            w = self.window_w1


        if not self.is_zigzag:
            x = rearrange(x, '(b i j) (t h w) c -> b t c (i h) (j w)', i=i, j=j, h=h, w=w)
        else:
            assert f"{h}_{w}" in self.zigzag_indices, "Zigzag indices not found, has it been initialized?"
            zigzag_indices = self.zigzag_indices[f"{h}_{w}"]
            x = rearrange(x, 'b (t h w) c -> b t c (h w)', h=h, w=w)
            _x = torch.empty(x.shape[0], x.shape[1], x.shape[2], h, w, device=x.device, dtype=x.dtype)
            _x[:, :, :, zigzag_indices[:, 0], zigzag_indices[:, 1]] = x
            x = rearrange(_x, '(b i j) t c h w -> b t c (i h) (j w)', i=i, j=j, h=h, w=w)
        return x


    def forward_feats(self, layer, hidden_state, residual=None, inference_params=None):
        # Implementation of inference
        if residual is None:
            res = torch.zeros_like(hidden_state)
            
        if inference_params is not None:
            if self.val_mode==Val_type.Frame_1_greedy_seq or self.val_mode==Val_type.Frame_4_seq:
                for idx in range(hidden_state.shape[1]):
                    if residual is None:
                        out = layer(
                            hidden_state[:,idx:idx+1], residual, inference_params=inference_params
                        )
                        res[:,idx:idx+1] = out[1]
                        hidden_state[:,idx:idx+1] = out[0]
                    else: 
                        out = layer(
                            hidden_state[:,idx:idx+1], residual[:,idx:idx+1], inference_params=inference_params
                        )
                        residual[:,idx:idx+1] = out[1]
                        hidden_state[:,idx:idx+1] = out[0]
                    inference_params.seqlen_offset += 1
                if self.val_mode == Val_type.Frame_4_seq:
                    inference_params.seqlen_offset = 0 
            elif self.val_mode == Val_type.Frame_1_greedy:
                hidden_state, residual = layer(
                    hidden_state, residual, inference_params=inference_params
                    )
                inference_params.seqlen_offset += 1
            else:
                hidden_state, residual = layer(
                    hidden_state, residual, inference_params=inference_params
                )
        else:   # Train and val for non sequential frame-by-frame processing
            hidden_state, residual = layer(
                hidden_state, residual, inference_params=inference_params
            )

        if residual is None:
            residual = res

        return hidden_state, residual

    def forward(self, x, inference_params=None):
        # mamba impl
        residual = None
        hidden_states = x
        for idx, layer in enumerate(self.layers):
            if idx%2==0:
                if self.is_quad_scan:
                    hidden_states = torch.rot90(hidden_states,1,(-1,-2))
                infer_param = inference_params
                if inference_params is not None:
                    infer_param = inference_params[0]
                hidden_states, residual = self.unshifted_process(layer, hidden_states, residual, infer_param)
            else:
                infer_param = inference_params
                if inference_params is not None:
                    infer_param = inference_params[1]
                if self.model_mode==Model_type.Alt_Residual:
                    hidden_states, _ = self.unshifted_process(layer, hidden_states, None, infer_param)
                else:
                    hidden_states, residual = self.shifted_process(layer, hidden_states, residual, infer_param)

        B, T, C, H, W = hidden_states.shape
        h1, w1 = H//self.window_h1, W//self.window_w1

        hidden_states = hidden_states.reshape(B, T, self.embed_dim, h1, self.window_h1, w1, self.window_w1)
        # hidden_states = rearrange(hidden_states, 'b t c i h j w -> (b i j) (t h w) c')
        hidden_states = self.flatten(hidden_states)
        if residual is not None:
            residual = residual.reshape(B, T, self.embed_dim, h1, self.window_h1, w1, self.window_w1)
            # residual = rearrange(residual, 'b t c i h j w -> (b i j) (t h w) c')
            residual = self.flatten(residual)

        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                self.drop_path(hidden_states),
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )

        if residual is not None:
            # residual = rearrange(residual, '(b i j) (t h w) c -> b t c (i h) (j w)', i=h1,    j=w1, h=self.window_h1 , w=self.window_w1)  
            residual = self.unflatten(residual, h1, w1)
        # hidden_states = rearrange(hidden_states, '(b i j) (t h w) c -> b t c (i h) (j w)', i=h1,    j=w1, h=self.window_h1 , w=self.window_w1)  
        hidden_states = self.unflatten(hidden_states, h1, w1)

        if inference_params is not None:
            self.residual = residual

        return hidden_states
    
    def unshifted_process(self, layer, hidden_state, residual=None, inference_params=None):

        B, T, C, H, W = hidden_state.shape
                
        h1, w1 = H//self.window_h1, W//self.window_w1

        # Updated statespace
        # [2, 4, 256, 60, 60] -> [2, 256, 6, 10, 6, 10]
        hidden_state = hidden_state.reshape(B, T, self.embed_dim, h1, self.window_h1, w1, self.window_w1)

        # hidden_state = rearrange(hidden_state, 'b t c i h j w -> (b i j) (t h w) c')
        # import ipdb;ipdb.set_trace()
        hidden_state = self.flatten(hidden_state)
        if residual is not None:
            residual = residual.reshape(B, T, self.embed_dim, h1, self.window_h1, w1, self.window_w1)
            # residual = rearrange(residual, 'b t c i h j w -> (b i j) (t h w) c')
            residual = self.flatten(residual)
        
        # Implementation of inference
        hidden_state, residual = self.forward_feats(layer, hidden_state, residual, inference_params)
        
        if residual is not None:
            # residual = rearrange(residual, '(b i j) (t h w) c -> b t c (i h) (j w)', i=h1,    j=w1, h=self.window_h1 , w=self.window_w1)  
            residual = self.unflatten(residual, h1, w1, h=self.window_h1 , w=self.window_w1)
        # hidden_state = rearrange(hidden_state, '(b i j) (t h w) c -> b t c (i h) (j w)', i=h1,    j=w1, h=self.window_h1 , w=self.window_w1)  
        hidden_state = self.unflatten(hidden_state, h1, w1, h=self.window_h1 , w=self.window_w1)

        return hidden_state, residual
    
    def shifted_process(self, layer, hidden_state, residual=None, inference_params=None):

        B, T, C, H, W = hidden_state.shape
        h1, w1 = H//self.window_h1, W//self.window_w1

    
        if not self.real_shift:
            # Update with shifted mamba
            h2, w2 = H//self.window_h2, W//self.window_w2

            hidden_state = hidden_state.reshape(B, T, self.embed_dim, h2, self.window_h2, w2, self.window_w2)

            # hidden_state = rearrange(hidden_state, 'b t c i h j w -> (b i j) (t h w) c')
            hidden_state = self.flatten(hidden_state)
            if residual is not None:
                residual = residual.reshape(B, T, self.embed_dim, h2, self.window_h2, w2, self.window_w2)
                # residual = rearrange(residual, 'b t c i h j w -> (b i j) (t h w) c')
                residual = self.flatten(residual)
                 
            # Implementation of inference
            hidden_state, residual = self.forward_feats(layer, hidden_state, residual, inference_params)

            # hidden_state = rearrange(hidden_state, '(b i j) (t h w) c -> b t c (i h) (j w)', i=h2, j=w2, h=self.window_h2 , w=self.window_w2)
            hidden_state = self.unflatten(hidden_state, h2, w2)
            if residual is not None:
                # residual = rearrange(residual, '(b i j) (t h w) c -> b t c (i h) (j w)', i=h2, j=w2, h=self.window_h2 , w=self.window_w2)
                residual = self.unflatten(residual, h2, w2)
        
        else:
            assert self.window_h1==self.window_h2 and self.window_w1==self.window_w2, "Real shift only works with same window size"
            assert self.shift_size is not None, "Shift size must be provided for real shift"
            # assert residual is None, "Residual for real shift is not implemented and should be None"
            # Update with shifted mamba
            h2, w2 = H//self.window_h2, W//self.window_w2

            
            hidden_state = torch.roll(hidden_state, shifts=(-self.shift_size, -self.shift_size), dims=(-2, -1))
            _hidden_state = torch.zeros_like(hidden_state)

            # Program section to handle shift size greater than or equal to window size
            if self.shift_size==self.window_h1:
                hidden_state = hidden_state.reshape(B, T, self.embed_dim, h2, self.window_h2, w2, self.window_w2)
                # hidden_state = rearrange(hidden_state, 'b t c i h j w -> (b i j) (t h w) c')
                hidden_state = self.flatten(hidden_state)

                if residual is not None:
                    residual = torch.roll(residual, shifts=(-self.shift_size, -self.shift_size), dims=(-2, -1))
                    residual = residual.reshape(B, T, self.embed_dim, h2, self.window_h2, w2, self.window_w2)
                    # residual = rearrange(residual, 'b t c i h j w -> (b i j) (t h w) c')
                    residual = self.flatten(residual)

                # Implementation of inference
                hidden_state, residual = self.forward_feats(layer, hidden_state, residual, inference_params)

                # hidden_state = rearrange(hidden_state, '(b i j) (t h w) c -> b t c (i h) (j w)', i=h2, j=w2, h=self.window_h2 , w=self.window_w2)
                hidden_state = self.unflatten(hidden_state, h2, w2)
                if residual is not None:
                    # residual = rearrange(residual, '(b i j) (t h w) c -> b t c (i h) (j w)', i=h2, j=w2, h=self.window_h2 , w=self.window_w2)
                    residual = self.unflatten(residual, h2, w2)
                    residual = torch.roll(residual, shifts=(self.shift_size, self.shift_size), dims=(-2, -1))

                hidden_state = torch.roll(hidden_state, shifts=(self.shift_size, self.shift_size), dims=(-2, -1))

                return hidden_state, residual


            # Split out the different dimensions at the last row and column to send them individually through the shifted mamba
            a1 = hidden_state[:,:,:,   -self.shift_size:,                  -self.shift_size:]
            b3 = hidden_state[:,:,:,   -self.window_h2:-self.shift_size,   -self.shift_size:]
            c3 = hidden_state[:,:,:,   -self.shift_size:,                  -self.window_w2:-self.shift_size]
            d = hidden_state[:,:,:,    -self.window_h2:-self.shift_size,   -self.window_w2:-self.shift_size]

            b1 = hidden_state[:,:,:,   :-self.window_h2,                   -self.shift_size:]
            b2 = hidden_state[:,:,:,   :-self.window_h2,                   -self.window_w2:-self.shift_size] 
            c1 = hidden_state[:,:,:,   -self.shift_size:,                  :-self.window_w2]
            c2 = hidden_state[:,:,:,   -self.window_h2:-self.shift_size,   :-self.window_w2]

            hidden_state = hidden_state[:,:,:,    :-self.window_h2,                   :-self.window_w2]

            # Reshape the split out dimensions to send them as windows parallely
            hidden_state = hidden_state.reshape   (B,     T,  self.embed_dim,     h2-1,   self.window_h2,                     w2-1,   self.window_w2                  )
            
            a1 = a1.reshape (B,     T,  self.embed_dim,     1,      self.shift_size,                    1,      self.shift_size                 )
            b3 = b3.reshape (B,     T,  self.embed_dim,     1,      self.window_h2-self.shift_size,     1,      self.shift_size                 )
            c3 = c3.reshape (B,     T,  self.embed_dim,     1,      self.shift_size,                    1,      self.window_w2-self.shift_size  )
            d = d.reshape   (B,     T,  self.embed_dim,     1,      self.window_h2-self.shift_size,     1,      self.window_w2-self.shift_size  )

            b1 = b1.reshape (B,     T,  self.embed_dim,     h2-1,   self.window_h2,                     1,      self.shift_size                 )
            b2 = b2.reshape (B,     T,  self.embed_dim,     h2-1,   self.window_h2,                     1,      self.window_w2-self.shift_size  )
            c1 = c1.reshape (B,     T,  self.embed_dim,     1,      self.shift_size,                    w2-1,   self.window_w2                  )
            c2 = c2.reshape (B,     T,  self.embed_dim,     1,      self.window_h2-self.shift_size,     w2-1,   self.window_w2                  )

            # Tackle for residual 
            

            if residual is not None:
                residual = torch.roll(residual, shifts=(-self.shift_size, -self.shift_size), dims=(-2, -1))
                _residual = torch.zeros_like(residual)

                # Split out the different dimensions at the last row and column to send them individually through the shifted mamba
                r_a1 = residual[:,:,:,   -self.shift_size:,                  -self.shift_size:]
                r_b3 = residual[:,:,:,   -self.window_h2:-self.shift_size,   -self.shift_size:]
                r_c3 = residual[:,:,:,   -self.shift_size:,                  -self.window_w2:-self.shift_size]
                r_d = residual[:,:,:,    -self.window_h2:-self.shift_size,   -self.window_w2:-self.shift_size]

                r_b1 = residual[:,:,:,   :-self.window_h2,                   -self.shift_size:]
                r_b2 = residual[:,:,:,   :-self.window_h2,                   -self.window_w2:-self.shift_size] 
                r_c1 = residual[:,:,:,   -self.shift_size:,                  :-self.window_w2]
                r_c2 = residual[:,:,:,   -self.window_h2:-self.shift_size,   :-self.window_w2]

                residual = residual[:,:,:,    :-self.window_h2,                   :-self.window_w2]

                
                # Reshape the split out dimensions to send them as windows parallely
                residual = residual.reshape   (B,     T,  self.embed_dim,     h2-1,   self.window_h2,                     w2-1,   self.window_w2)
                
                r_a1 = r_a1.reshape (B,     T,  self.embed_dim,     1,      self.shift_size,                    1,      self.shift_size                 )
                r_b3 = r_b3.reshape (B,     T,  self.embed_dim,     1,      self.window_h2-self.shift_size,     1,      self.shift_size                 )
                r_c3 = r_c3.reshape (B,     T,  self.embed_dim,     1,      self.shift_size,                    1,      self.window_w2-self.shift_size  )
                r_d  = r_d.reshape  (B,     T,  self.embed_dim,     1,      self.window_h2-self.shift_size,     1,      self.window_w2-self.shift_size  )

                r_b1 = r_b1.reshape (B,     T,  self.embed_dim,     h2-1,   self.window_h2,                     1,      self.shift_size                 )
                r_b2 = r_b2.reshape (B,     T,  self.embed_dim,     h2-1,   self.window_h2,                     1,      self.window_w2-self.shift_size  )
                r_c1 = r_c1.reshape (B,     T,  self.embed_dim,     1,      self.shift_size,                    w2-1,   self.window_w2                  )
                r_c2 = r_c2.reshape (B,     T,  self.embed_dim,     1,      self.window_h2-self.shift_size,     w2-1,   self.window_w2                  )
            else:
                r_a1, r_b3, r_c3, r_d, r_b1, r_b2, r_c1, r_c2 = None, None, None, None, None, None, None, None

            if inference_params is None:
                inference_params = [None]*9

            # Send the split out dimensions through the shifted mamba
            hidden_state, residual = self.shift_send(layer, hidden_state,      h2-1,   self.window_h2,                     w2-1,   self.window_w2                  , residual, inference_params[0])
            
            a1, r_a1 = self.shift_send(layer, a1,    1,      self.shift_size,                    1,      self.shift_size                 , r_a1, inference_params[1])
            b3, r_b3 = self.shift_send(layer, b3,    1,      self.window_h2-self.shift_size,     1,      self.shift_size                 , r_b3, inference_params[2])
            c3, r_c3 = self.shift_send(layer, c3,    1,      self.shift_size,                    1,      self.window_w2-self.shift_size  , r_c3, inference_params[3])
            d , r_d = self.shift_send(layer, d,      1,      self.window_h2-self.shift_size,     1,      self.window_w2-self.shift_size  , r_d,  inference_params[4])

            b1, r_b1 = self.shift_send(layer, b1,    h2-1,   self.window_h2,                     1,      self.shift_size                 , r_b1, inference_params[5])
            b2, r_b2 = self.shift_send(layer, b2,    h2-1,   self.window_h2,                     1,      self.window_w2-self.shift_size  , r_b2, inference_params[6])
            c1, r_c1 = self.shift_send(layer, c1,    1,      self.shift_size,                    w2-1,   self.window_w2                  , r_c1, inference_params[7])
            c2, r_c2 = self.shift_send(layer, c2,    1,      self.window_h2-self.shift_size,     w2-1,   self.window_w2                  , r_c2, inference_params[8])

            # Combine the outputs from the shifted mamba back to the original shape
            _hidden_state[:,:,:,   :-self.window_h2,                   :-self.window_w2                ] = hidden_state

            _hidden_state[:,:,:,   -self.shift_size:,                  -self.shift_size:               ] = a1  
            _hidden_state[:,:,:,   -self.window_h2:-self.shift_size,   -self.shift_size:               ] = b3
            _hidden_state[:,:,:,   -self.shift_size:,                  -self.window_w2:-self.shift_size] = c3
            _hidden_state[:,:,:,   -self.window_h2:-self.shift_size,   -self.window_w2:-self.shift_size] = d

            _hidden_state[:,:,:,   :-self.window_h2,                   -self.shift_size:               ] = b1
            _hidden_state[:,:,:,   :-self.window_h2,                   -self.window_w2:-self.shift_size] = b2
            _hidden_state[:,:,:,   -self.shift_size:,                  :-self.window_w2                ] = c1
            _hidden_state[:,:,:,   -self.window_h2:-self.shift_size,   :-self.window_w2                ] = c2
                

            hidden_state = torch.roll(_hidden_state, shifts=(self.shift_size, self.shift_size), dims=(-2, -1))    

            if residual is not None:
                _residual[:,:,:,   :-self.window_h2,                   :-self.window_w2                ] = residual

                _residual[:,:,:,   -self.shift_size:,                  -self.shift_size:               ] = r_a1  
                _residual[:,:,:,   -self.window_h2:-self.shift_size,   -self.shift_size:               ] = r_b3
                _residual[:,:,:,   -self.shift_size:,                  -self.window_w2:-self.shift_size] = r_c3
                _residual[:,:,:,   -self.window_h2:-self.shift_size,   -self.window_w2:-self.shift_size] = r_d

                _residual[:,:,:,   :-self.window_h2,                   -self.shift_size:               ] = r_b1
                _residual[:,:,:,   :-self.window_h2,                   -self.window_w2:-self.shift_size] = r_b2
                _residual[:,:,:,   -self.shift_size:,                  :-self.window_w2                ] = r_c1
                _residual[:,:,:,   -self.window_h2:-self.shift_size,   :-self.window_w2                ] = r_c2

                residual = torch.roll(_residual, shifts=(self.shift_size, self.shift_size), dims=(-2, -1))

        return hidden_state, residual

    def shift_send(self, layer, hidden_state, h2, window_h, w2, window_w, residual=None, inference_params=None):
        # hidden_state = rearrange(hidden_state, 'b t c i h j w -> (b i j) (t h w) c')
        hidden_state = self.flatten(hidden_state)
        if residual is not None: 
            # residual = rearrange(residual, 'b t c i h j w -> (b i j) (t h w) c')
            residual = self.flatten(residual)

        # hidden_state, residual = layer(
        #     hidden_state, residual, inference_params=inference_params
        # )
        hidden_state, residual = self.forward_feats(layer, hidden_state, residual, inference_params)


        # hidden_state = rearrange(hidden_state, '(b i j) (t h w) c -> b t c (i h) (j w)', i=h2, j=w2, h=window_h , w=window_w)
        hidden_state = self.unflatten(hidden_state, h2, w2, h=window_h, w=window_w)
        if residual is not None:
            # residual = rearrange(residual, '(b i j) (t h w) c -> b t c (i h) (j w)', i=h2, j=w2, h=window_h , w=window_w)
            residual = self.unflatten(residual, h2, w2, h=window_h, w=window_w)
                
        return hidden_state, residual

'''Modules'''

@HEADS.register_module()
class TV3SHead_shift(BaseDecodeHead_clips_flow):
    '''
    Note: Updated with mamba blocks from videomamba
    '''
    def __init__(self, feature_strides, **kwargs):
        super(TV3SHead_shift, self).__init__(input_transform='multiple_select', **kwargs)
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        decoder_params = kwargs['decoder_params']
        embedding_dim = decoder_params['embed_dim']
        self.emmbedding_dim = embedding_dim

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.window_w = decoder_params['window_w']
        self.window_h = decoder_params['window_h']
        shift_size = decoder_params['shift_size']

        self.greedy_inference = False
        self.inference_params = None
        self.test_mode = False
        self.val_mode = Val_type.Normal
        self.use_emmbedding = False
        self.prev_class = None
        self.add_mode = False
        self.concat_mode = False
        
        if 'zigzag' in decoder_params:
            self.zigzag = decoder_params['zigzag']
        else:
            self.zigzag = False

        if 'quad_scan' in decoder_params:
            self.quad_scan = decoder_params['quad_scan']
        else:
            self.quad_scan = False
        
        if 'model_type' in decoder_params:
            self.model_type = decoder_params['model_type']
        else:
            self.model_type = Model_type.Normal

        if not 'n_mambas' in decoder_params:
            total_mambas = 8                   # Note the default is 8 for shifted mambas
        else:
            total_mambas = decoder_params['n_mambas']

        if 'real_shift' in decoder_params:
            self.real_shift= decoder_params['real_shift']
        else:
            self.real_shift = False     # Note defaulted to Expand

        if not 'mamba2' in decoder_params:
            mamba2 = False
        else:
            mamba2 = decoder_params['mamba2']

        if 'test_mode' in decoder_params:
            self.test_mode = decoder_params['test_mode']
            if self.test_mode:
                if 'max_seqlen' in decoder_params and 'max_batch_size' in decoder_params:
                    NotImplementedError("Inference params not implemented for user params")
                else:
                    print("Inference params not provided, using default of 1000*1000 for seqlen and 10 for batch size")
                    self.max_seqlen = 1000*1000
                    self.max_batch_size = 10
                    if self.real_shift:     # Note: Need to handle for real_shift and Expand
                        self.inference_params = [InferenceParams(max_seqlen=self.max_seqlen,max_batch_size=self.max_batch_size), [InferenceParams(max_seqlen=int(self.max_seqlen/10),max_batch_size=self.max_batch_size) for _ in range(9)]]
                    else:
                        self.inference_params = [InferenceParams(max_seqlen=self.max_seqlen,max_batch_size=self.max_batch_size), InferenceParams(max_seqlen=self.max_seqlen,max_batch_size=self.max_batch_size)]
                
                if 'val_mode' in decoder_params:
                    self.val_mode = decoder_params['val_mode']

                    if self.val_mode == Val_type.Frame_1_greedy or self.val_mode == Val_type.Frame_1_greedy_seq:
                        self.greedy_inference = True
                        print("*"*20, "Running Greedy Infer!!", "*"*20)
                    elif self.val_mode == Val_type.Frame_4_seq:
                        # self.greedy_inference = True
                        print("*"*20, "Running Seq Infer with F4!!", "*"*20)
                else:
                    self.val_mode = Val_type.Normal

        if self.model_type == Model_type.Add:
            self.add_mode = True
        elif self.model_type == Model_type.Concat:
            self.concat_mode = True
            
        if self.model_type == Model_type.Bi or self.model_type == Model_type.Bi_Embed:
            bimamba = True
        else:
            bimamba = False

        if bimamba:
            if 'im_size' in decoder_params:
                self.h, self.w = decoder_params['im_size']
                if 'n_frames' in decoder_params:
                    num_frames = decoder_params['n_frames']
                else:
                    warnings.warn("n_frames not provided, using default of 512")
                    num_frames = 512
            else:
                warnings.warn("im_size not provided, using default of 60x60")
                self.h, self.w = 60, 60
                warnings.warn("n_frames not provided, using default of 512")
                num_frames = 512

            if self.model_type == Model_type.Bi_Embed:
                self.use_emmbedding = True
                self.temporal_pos_embedding = nn.Parameter(torch.zeros(1, num_frames, self.emmbedding_dim, 1, 1, 1, 1))
                self.pos_embed = nn.Parameter(torch.zeros(1, 1, self.emmbedding_dim, (self.h//self.window_h), 1, (self.w//self.window_w), 1))

            # If testing
            if self.test_mode:
                self.t_count = 0

        # Handle the shift size
        if not self.real_shift:
            self.window_w2 = self.window_w + shift_size
            self.window_h2 = self.window_h + shift_size
        else:
            self.window_w2 = self.window_w
            self.window_h2 = self.window_h
        
        # self.vs = Mamba(d_model=embedding_dim, d_state=16, d_conv=4, expand=2)
        # self.vs = mambaSequential()
        self.vs = ShiftedMambaBlocks(
                    embed_dim=embedding_dim,
                    depth=total_mambas,
                    window_w1=self.window_w, 
                    window_h1=self.window_h, 
                    window_w2=self.window_w2, 
                    window_h2=self.window_h2,
                    real_shift=self.real_shift,
                    shift_size=shift_size, 
                    bimamba=bimamba,
                    mamba2=mamba2,
                    val_mode=self.val_mode,
                    model_mode=self.model_type,
                    zigzag=self.zigzag,
                    quad_scan=self.quad_scan
                )


        self.linear_fuse = ConvModule(
            in_channels=embedding_dim*4,
            out_channels=embedding_dim,
            kernel_size=1,
            norm_cfg=dict(type='SyncBN', requires_grad=True)
        )

        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)
        
        # self.linear_predvs = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)

        if self.concat_mode:
            self.linear_pred2 = nn.Conv2d(embedding_dim*2, self.num_classes, kernel_size=1)
        else:
            self.linear_pred2 = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)
        
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'temporal_pos_embedding', 'pos_embed'}
    
    def forward(self, inputs, batch_size=None, num_clips=None, imgs=None, img_metas=None):
        if self.training:
            assert self.num_clips==num_clips, f"{self.num_clips} and {num_clips}"

        # Assert for img_metas is not None during inference
        if self.test_mode and self.inference_params is not None:
            assert img_metas is not None, "Require img_metas during inference to properly clear the statespace"
            assert len(img_metas) == 1, "Only single image inference supported" 

            # Support only single GPU inference
            cur_class = os.path.split(os.path.split(os.path.split(img_metas[0]['filename'])[0])[0])[1]           
            if self.prev_class is not None and self.prev_class != cur_class:
                if self.real_shift:
                    self.inference_params = [InferenceParams(max_seqlen=self.max_seqlen,max_batch_size=self.max_batch_size), [InferenceParams(max_seqlen=int(self.max_seqlen/10),max_batch_size=self.max_batch_size) for _ in range(9)]]
                else:
                    self.inference_params = [InferenceParams(max_seqlen=self.max_seqlen,max_batch_size=self.max_batch_size), InferenceParams(max_seqlen=self.max_seqlen,max_batch_size=self.max_batch_size)]
                
                '''Changed here as was not previously reset'''
                # self.inference_params.seqlen_offset = 0 
                self.inference_params[0].seqlen_offset = 0 
                self.vs.residual = None
                
                if type(self.inference_params[1]) is list:
                    for i in range(len(self.inference_params[1])):
                        self.inference_params[1][i].seqlen_offset = 0
                else:    
                    self.inference_params[1].seqlen_offset = 0

                if self.use_emmbedding:
                    self.t_count = 0

                print(cur_class)

            self.prev_class = cur_class
            
        x = self._transform_inputs(inputs)  # len=4, 1/4,1/8,1/16,1/32 # No Chg
        # [[8, 64, 120, 120], [8, 128, 60, 60], [8, 320, 30, 30], [8, 512, 15, 15]]
        c1, c2, c3, c4 = x
        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape

        _c4 = self.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3]) # [8, 512, 15, 15] -> [8, 256, 15, 15]
        _c4 = resize(_c4, size=c1.size()[2:],mode='bilinear',align_corners=False) # [8, 256, 15, 15] -> [8, 256, 120, 120]

        _c3 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3]) # [8, 320, 30, 30] -> [8, 256, 30, 30]
        _c3 = resize(_c3, size=c1.size()[2:],mode='bilinear',align_corners=False) # [8, 256, 30, 30] -> [8, 256, 120, 120]

        _c2 = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3]) # [8, 128, 60, 60] -> [8, 256, 60, 60]
        _c2 = resize(_c2, size=c1.size()[2:],mode='bilinear',align_corners=False) # [8, 256, 60, 60] -> [8, 256, 120, 120]

        _c1 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3]) # [8, 64, 120, 120] -> [8, 256, 120, 120]

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1)) # [8, 256, 120, 120] 

        _, _, h, w=_c.shape
        '''x = self.dropout(_c) # [8, 256, 120, 120]
        x = self.linear_pred(x) # [8, 256, 120, 120] -> [8, 124, 120, 120]
        x = x.reshape(batch_size, num_clips, -1, h, w) # [8, 124, 120, 120] -> [2, 4, 124, 120, 120]
	'''
        if not self.test_mode:
            x = self.dropout(_c) # [8, 256, 120, 120]
            x = self.linear_pred(x) # [8, 256, 120, 120] -> [8, 124, 120, 120]
            x = x.reshape(batch_size, num_clips, -1, h, w) # [8, 124, 120, 120] -> [2, 4, 124, 120, 120]
        else:
            x = self.linear_pred(_c) # [8, 256, 120, 120] -> [8, 124, 120, 120]
            x = x.reshape(batch_size, num_clips, -1, h, w) # [8, 124, 120, 120] -> [2, 4, 124, 120, 120]
            # x = torch.zeros(batch_size, num_clips, self.num_classes, h, w).to(_c.device)

        # print("_c.shape: ", _c.shape)
        # print("Checking....")
        if not self.training and num_clips!=self.num_clips:
        # if not self.training:
            return x[:,-1]
        # if n ==1: 
        #     return x[:,-1]
        # print("Not Skipped!!!!")

        # Make h2 and w2 divisible by window size
        h2 = int(h/2) # 60
        w2 = int(w/2) # 60
        
        _wcheck = np.lcm(self.window_w, self.window_w2)
        _hcheck = np.lcm(self.window_h, self.window_h2)
        h2, w2 = np.ceil(h2/_hcheck).astype(int)*_hcheck, np.ceil(w2/_wcheck).astype(int)*_wcheck
        # h2, w2 = (h//self.window_h)*self.window_h, (w//self.window_w)*self.window_w
        
        _c = resize(_c, size=(h2,w2),mode='bilinear',align_corners=False) # [8, 256, 120, 120] -> [8, 256, 60, 60]

        _c_further=_c.reshape(batch_size, num_clips, -1, h2, w2) # [8, 256, 60, 60] -> [2, 4, 256, 60, 60]
        _c2 = _c_further.clone()    
        
        # Note the shape change
        if self.use_emmbedding:
            # #pass through the statespace model
            h3, w3 = h2//self.window_h, w2//self.window_w
            # Updated statespace
            # [2, 4, 256, 60, 60] -> [2, 4, 256, 6, 10, 6, 10]
            _c2 = _c2.reshape(batch_size, num_clips, self.emmbedding_dim, h3, self.window_h, w3, self.window_w)

            # Interpolate the positional embeddings
            if len(self.pos_embed.shape) > 4 and not (self.pos_embed.shape[-4], self.pos_embed.shape[-2]) == (h3, w3): 
                pos_embed = rearrange(self.pos_embed, 'b t c i h j w -> (b t) c (h i) (w j)')
                pos_embed = nn.functional.interpolate(pos_embed, size=(h3,w3), mode='bilinear', align_corners=False)
                self.pos_embed =  nn.Parameter(pos_embed.reshape(1, 1, self.emmbedding_dim, h3, 1, w3, 1))
            
            elif len(self.pos_embed.shape) == 4 and not self.pos_embed.shape[-2:] == (h3, w3):
                pos_embed = nn.functional.interpolate(self.pos_embed, size=(h3,w3), mode='bilinear', align_corners=False)
                self.pos_embed =  nn.Parameter(pos_embed.reshape(1, 1, self.emmbedding_dim, h3, 1, w3, 1))
            
            if self.test_mode:
                temporal_pos_embed = nn.Parameter(self.temporal_pos_embedding[:,self.t_count:self.t_count+num_clips])
                self.t_count +=1   
            else:
                # Random choose start point for temporal pos embedding to avoid overfitting
                rand_start = random.randint(0, self.temporal_pos_embedding.shape[1]-num_clips)
                temporal_pos_embed = nn.Parameter(self.temporal_pos_embedding[:,rand_start:rand_start+num_clips])    

            _c2 = _c2 + self.pos_embed
            _c2 = _c2 + temporal_pos_embed
            _c2 = rearrange(_c2, 'b t c i h j w -> (b) t c (i h) (j w)')
        # import ipdb;ipdb.set_trace()    
        if self.inference_params is not None and not self.greedy_inference:
            self.inference_params[0].seqlen_offset = 0 
            self.vs.residual = None
            
            if type(self.inference_params[1]) is list:
                for i in range(len(self.inference_params[1])):
                    self.inference_params[1][i].seqlen_offset = 0
            else:    
                self.inference_params[1].seqlen_offset = 0
                
            _c2 = self.vs(_c2, self.inference_params)
        else:
            _c2 = self.vs(_c2, self.inference_params)

        assert _c_further.shape==_c2.shape

        if self.concat_mode:
            _c_further2=torch.cat([_c_further[:,-1], _c2[:,-1]],1) # [2, 256, 60, 60] + [2, 256, 60, 60] -> [2, 512, 60, 60]
        elif self.add_mode:
            _c_further2 = _c_further[:,-1] + _c2[:,-1]
        else:
            _c_further2 = _c2[:,-1]

        x2 = self.dropout(_c_further2) # [2, 512, 60, 60]
        x2 = self.linear_pred2(x2) # [2, 512, 60, 60] -> [2, 124, 60, 60]
        x2=resize(x2, size=(h,w),mode='bilinear',align_corners=False) # [2, 124, 60, 60] -> [2, 124, 120, 120]
        x2=x2.unsqueeze(1) # [2, 124, 120, 120] -> [2, 1, 124, 120, 120]

        x=torch.cat([x,x2],1)   ## b*(k+1)*124*h*w [2, 5, 124, 120, 120]

        if not self.training:
            return x2.squeeze(1)

        return x
    
@HEADS.register_module()
class TV3SHead_shift_city(BaseDecodeHead_clips_flow_city):
    '''
    Note: Updated with mamba blocks from videomamba
    '''
    def __init__(self, feature_strides, **kwargs):
        super(TV3SHead_shift_city, self).__init__(input_transform='multiple_select', **kwargs)
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        decoder_params = kwargs['decoder_params']
        embedding_dim = decoder_params['embed_dim']
        self.emmbedding_dim = embedding_dim

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.window_w = decoder_params['window_w']
        self.window_h = decoder_params['window_h']
        shift_size = decoder_params['shift_size']

        self.greedy_inference = False
        self.inference_params = None
        self.test_mode = False
        self.val_mode = Val_type.Normal
        self.use_emmbedding = False
        self.prev_class = None
        self.add_mode = False
        self.concat_mode = False
        
        if 'model_type' in decoder_params:
            self.model_type = decoder_params['model_type']
        else:
            self.model_type = Model_type.Normal

        if not 'n_mambas' in decoder_params:
            total_mambas = 8                   # Note the default is 8 for shifted mambas
        else:
            total_mambas = decoder_params['n_mambas']

        if 'real_shift' in decoder_params:
            self.real_shift= decoder_params['real_shift']
        else:
            self.real_shift = False     # Note defaulted to Expand

        if not 'mamba2' in decoder_params:
            mamba2 = False
        else:
            mamba2 = decoder_params['mamba2']

        if 'test_mode' in decoder_params:
            self.test_mode = decoder_params['test_mode']
            if self.test_mode:
                if 'max_seqlen' in decoder_params and 'max_batch_size' in decoder_params:
                    NotImplementedError("Inference params not implemented for user params")
                else:
                    print("Inference params not provided, using default of 1000*1000 for seqlen and 10 for batch size")
                    self.max_seqlen = 1000*1000
                    self.max_batch_size = 10
                    if self.real_shift:     # Note: Need to handle for real_shift and Expand
                        self.inference_params = [InferenceParams(max_seqlen=self.max_seqlen,max_batch_size=self.max_batch_size), [InferenceParams(max_seqlen=int(self.max_seqlen/10),max_batch_size=self.max_batch_size) for _ in range(9)]]
                    else:
                        self.inference_params = [InferenceParams(max_seqlen=self.max_seqlen,max_batch_size=self.max_batch_size), InferenceParams(max_seqlen=self.max_seqlen,max_batch_size=self.max_batch_size)]
                
                if 'val_mode' in decoder_params:
                    self.val_mode = decoder_params['val_mode']

                    if self.val_mode == Val_type.Frame_1_greedy or self.val_mode == Val_type.Frame_1_greedy_seq:
                        self.greedy_inference = True
                        print("*"*20, "Running Greedy Infer!!", "*"*20)
                    elif self.val_mode == Val_type.Frame_4_seq:
                        # self.greedy_inference = True
                        print("*"*20, "Running Seq Infer with F4!!", "*"*20)
                else:
                    self.val_mode = Val_type.Normal

        if self.model_type == Model_type.Add:
            self.add_mode = True
        elif self.model_type == Model_type.Concat:
            self.concat_mode = True
            
        if self.model_type == Model_type.Bi or self.model_type == Model_type.Bi_Embed:
            bimamba = True
        else:
            bimamba = False

        if bimamba:
            if 'im_size' in decoder_params:
                self.h, self.w = decoder_params['im_size']
                if 'n_frames' in decoder_params:
                    num_frames = decoder_params['n_frames']
                else:
                    warnings.warn("n_frames not provided, using default of 512")
                    num_frames = 512
            else:
                warnings.warn("im_size not provided, using default of 60x60")
                self.h, self.w = 60, 60
                warnings.warn("n_frames not provided, using default of 512")
                num_frames = 512

            if self.model_type == Model_type.Bi_Embed:
                self.use_emmbedding = True
                self.temporal_pos_embedding = nn.Parameter(torch.zeros(1, num_frames, self.emmbedding_dim, 1, 1, 1, 1))
                self.pos_embed = nn.Parameter(torch.zeros(1, 1, self.emmbedding_dim, (self.h//self.window_h), 1, (self.w//self.window_w), 1))

            # If testing
            if self.test_mode:
                self.t_count = 0

        # Handle the shift size
        if not self.real_shift:
            self.window_w2 = self.window_w + shift_size
            self.window_h2 = self.window_h + shift_size
        else:
            self.window_w2 = self.window_w
            self.window_h2 = self.window_h
        
        # self.vs = Mamba(d_model=embedding_dim, d_state=16, d_conv=4, expand=2)
        # self.vs = mambaSequential()
        self.vs = ShiftedMambaBlocks(
                    embed_dim=embedding_dim,
                    depth=total_mambas,
                    window_w1=self.window_w, 
                    window_h1=self.window_h, 
                    window_w2=self.window_w2, 
                    window_h2=self.window_h2,
                    real_shift=self.real_shift,
                    shift_size=shift_size, 
                    bimamba=bimamba,
                    mamba2=mamba2,
                    val_mode=self.val_mode,
                    model_mode=self.model_type
                )


        self.linear_fuse = ConvModule(
            in_channels=embedding_dim*4,
            out_channels=embedding_dim,
            kernel_size=1,
            norm_cfg=dict(type='SyncBN', requires_grad=True)
        )

        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)
        
        # self.linear_predvs = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)

        if self.concat_mode:
            self.linear_pred2 = nn.Conv2d(embedding_dim*2, self.num_classes, kernel_size=1)
        else:
            self.linear_pred2 = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)
        
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'temporal_pos_embedding', 'pos_embed'}
    
    def forward(self, inputs, batch_size=None, num_clips=None, imgs=None, img_metas=None):
        if self.training:
            assert self.num_clips==num_clips, f"{self.num_clips} and {num_clips}"

        # Assert for img_metas is not None during inference
        if self.test_mode and self.inference_params is not None:
            assert img_metas is not None, "Require img_metas during inference to properly clear the statespace"
            assert len(img_metas) == 1, "Only single image inference supported" 

            # Support only single GPU inference
            cur_class = os.path.split(os.path.split(os.path.split(img_metas[0]['filename'])[0])[0])[1]           
            if self.prev_class is not None and self.prev_class != cur_class:
                if self.real_shift:
                    self.inference_params = [InferenceParams(max_seqlen=self.max_seqlen,max_batch_size=self.max_batch_size), [InferenceParams(max_seqlen=int(self.max_seqlen/10),max_batch_size=self.max_batch_size) for _ in range(9)]]
                else:
                    self.inference_params = [InferenceParams(max_seqlen=self.max_seqlen,max_batch_size=self.max_batch_size), InferenceParams(max_seqlen=self.max_seqlen,max_batch_size=self.max_batch_size)]
                
                '''Changed here as was not previously reset'''
                # self.inference_params.seqlen_offset = 0 
                self.inference_params[0].seqlen_offset = 0 
                self.vs.residual = None
                
                if type(self.inference_params[1]) is list:
                    for i in range(len(self.inference_params[1])):
                        self.inference_params[1][i].seqlen_offset = 0
                else:    
                    self.inference_params[1].seqlen_offset = 0

                if self.use_emmbedding:
                    self.t_count = 0

                print(cur_class)

            self.prev_class = cur_class
            
        x = self._transform_inputs(inputs)  # len=4, 1/4,1/8,1/16,1/32 # No Chg
        # [[8, 64, 120, 120], [8, 128, 60, 60], [8, 320, 30, 30], [8, 512, 15, 15]]
        c1, c2, c3, c4 = x
        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape

        _c4 = self.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3]) # [8, 512, 15, 15] -> [8, 256, 15, 15]
        _c4 = resize(_c4, size=c1.size()[2:],mode='bilinear',align_corners=False) # [8, 256, 15, 15] -> [8, 256, 120, 120]

        _c3 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3]) # [8, 320, 30, 30] -> [8, 256, 30, 30]
        _c3 = resize(_c3, size=c1.size()[2:],mode='bilinear',align_corners=False) # [8, 256, 30, 30] -> [8, 256, 120, 120]

        _c2 = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3]) # [8, 128, 60, 60] -> [8, 256, 60, 60]
        _c2 = resize(_c2, size=c1.size()[2:],mode='bilinear',align_corners=False) # [8, 256, 60, 60] -> [8, 256, 120, 120]

        _c1 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3]) # [8, 64, 120, 120] -> [8, 256, 120, 120]

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1)) # [8, 256, 120, 120] 

        _, _, h, w=_c.shape
        '''x = self.dropout(_c) # [8, 256, 120, 120]
        x = self.linear_pred(x) # [8, 256, 120, 120] -> [8, 124, 120, 120]
        x = x.reshape(batch_size, num_clips, -1, h, w) # [8, 124, 120, 120] -> [2, 4, 124, 120, 120]
	'''
        if not self.test_mode:
            x = self.dropout(_c) # [8, 256, 120, 120]
            x = self.linear_pred(x) # [8, 256, 120, 120] -> [8, 124, 120, 120]
            x = x.reshape(batch_size, num_clips, -1, h, w) # [8, 124, 120, 120] -> [2, 4, 124, 120, 120]
        else:
            x = torch.zeros(batch_size, num_clips, self.num_classes, h, w).to(_c.device)

        # print("_c.shape: ", _c.shape)
        if not self.training and num_clips!=self.num_clips:
        # if not self.training:
            return x[:,-1]

        # Make h2 and w2 divisible by window size
        h2 = int(h/2) # 60
        w2 = int(w/2) # 60
        
        _wcheck = np.lcm(self.window_w, self.window_w2)
        _hcheck = np.lcm(self.window_h, self.window_h2)
        h2, w2 = np.ceil(h2/_hcheck).astype(int)*_hcheck, np.ceil(w2/_wcheck).astype(int)*_wcheck
        # h2, w2 = (h//self.window_h)*self.window_h, (w//self.window_w)*self.window_w
        
        _c = resize(_c, size=(h2,w2),mode='bilinear',align_corners=False) # [8, 256, 120, 120] -> [8, 256, 60, 60]

        _c_further=_c.reshape(batch_size, num_clips, -1, h2, w2) # [8, 256, 60, 60] -> [2, 4, 256, 60, 60]
        _c2 = _c_further.clone()    
        
        # Note the shape change
        if self.use_emmbedding:
            # #pass through the statespace model
            h3, w3 = h2//self.window_h, w2//self.window_w
            # Updated statespace
            # [2, 4, 256, 60, 60] -> [2, 4, 256, 6, 10, 6, 10]
            _c2 = _c2.reshape(batch_size, num_clips, self.emmbedding_dim, h3, self.window_h, w3, self.window_w)

            # Interpolate the positional embeddings
            if len(self.pos_embed.shape) > 4 and not (self.pos_embed.shape[-4], self.pos_embed.shape[-2]) == (h3, w3): 
                pos_embed = rearrange(self.pos_embed, 'b t c i h j w -> (b t) c (h i) (w j)')
                pos_embed = nn.functional.interpolate(pos_embed, size=(h3,w3), mode='bilinear', align_corners=False)
                self.pos_embed =  nn.Parameter(pos_embed.reshape(1, 1, self.emmbedding_dim, h3, 1, w3, 1))
            
            elif len(self.pos_embed.shape) == 4 and not self.pos_embed.shape[-2:] == (h3, w3):
                pos_embed = nn.functional.interpolate(self.pos_embed, size=(h3,w3), mode='bilinear', align_corners=False)
                self.pos_embed =  nn.Parameter(pos_embed.reshape(1, 1, self.emmbedding_dim, h3, 1, w3, 1))
            
            if self.test_mode:
                temporal_pos_embed = nn.Parameter(self.temporal_pos_embedding[:,self.t_count:self.t_count+num_clips])
                self.t_count +=1   
            else:
                # Random choose start point for temporal pos embedding to avoid overfitting
                rand_start = random.randint(0, self.temporal_pos_embedding.shape[1]-num_clips)
                temporal_pos_embed = nn.Parameter(self.temporal_pos_embedding[:,rand_start:rand_start+num_clips])    

            _c2 = _c2 + self.pos_embed
            _c2 = _c2 + temporal_pos_embed
            _c2 = rearrange(_c2, 'b t c i h j w -> (b) t c (i h) (j w)')
        # import ipdb;ipdb.set_trace()    
        if self.inference_params is not None and not self.greedy_inference:
            self.inference_params[0].seqlen_offset = 0 
            self.vs.residual = None
            
            if type(self.inference_params[1]) is list:
                for i in range(len(self.inference_params[1])):
                    self.inference_params[1][i].seqlen_offset = 0
            else:    
                self.inference_params[1].seqlen_offset = 0
                
            _c2 = self.vs(_c2, self.inference_params)
        else:
            _c2 = self.vs(_c2, self.inference_params)

        assert _c_further.shape==_c2.shape

        if self.concat_mode:
            _c_further2=torch.cat([_c_further[:,-1], _c2[:,-1]],1) # [2, 256, 60, 60] + [2, 256, 60, 60] -> [2, 512, 60, 60]
        elif self.add_mode:
            _c_further2 = _c_further[:,-1] + _c2[:,-1]
        else:
            _c_further2 = _c2[:,-1]

        x2 = self.dropout(_c_further2) # [2, 512, 60, 60]
        x2 = self.linear_pred2(x2) # [2, 512, 60, 60] -> [2, 124, 60, 60]
        x2=resize(x2, size=(h,w),mode='bilinear',align_corners=False) # [2, 124, 60, 60] -> [2, 124, 120, 120]
        x2=x2.unsqueeze(1) # [2, 124, 120, 120] -> [2, 1, 124, 120, 120]

        x=torch.cat([x,x2],1)   ## b*(k+1)*124*h*w [2, 5, 124, 120, 120]

        if not self.training:
            return x2.squeeze(1)

        return x
