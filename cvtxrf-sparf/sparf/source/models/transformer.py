import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as torch_F

from typing import Callable, Sequence, List, Mapping, MutableMapping, Tuple, Union, Dict
from typing import Any, Optional

from source.models.frequency_nerf import FrequencyEmbedder

class Transformer(torch.nn.Module): 
    def __init__(self, opt: Dict[str, Any]): 
        super().__init__()
        self.opt = opt
        self.define_network(opt)

        self.embedder_pts = FrequencyEmbedder(self.opt)
        self.embedder_view = FrequencyEmbedder(self.opt)
        if opt.barf_c2f is None:
            # will be useless but just to be on the safe side. 
            # that corresponds to high frequency positional encoding
            self.progress = torch.nn.Parameter(torch.tensor(1.)) 
        else:
            self.progress = torch.nn.Parameter(torch.tensor(0.)) 

    def define_network(self, opt: Dict[str, Any]): 
        if opt.arch.layers_feat_fine is not None: 
            self.alph_embed_dim = opt.arch.layers_feat_fine[-1]
        else: 
            self.alph_embed_dim = opt.arch.layers_feat[-1]
        self.alph_fc = nn.Sequential(
            nn.Linear(self.alph_embed_dim, 1), 
            # nn.ReLU(), 
            getattr(nn, opt.transformer.density_activ)() # relu_,abs_,sigmoid_,exp_....
        )

        self.rgb_embed_dim = opt.arch.layers_rgb[-2] # 128
        self.rgb_proj = nn.Sequential(
            nn.Linear(self.alph_embed_dim, self.rgb_embed_dim), 
            nn.ReLU(), 
        )
        self.rgb_fc = nn.Sequential(
            nn.Linear(self.rgb_embed_dim, 3),
            nn.Sigmoid(),  
        )

        self.encoder = nn.Sequential(*nn.ModuleList(
            [nn.TransformerEncoderLayer(
                d_model=self.alph_embed_dim,  
                nhead=opt.transformer.n_heads, 
                dim_feedforward=opt.transformer.dim_ff, 
                batch_first=True) for _ in range(opt.transformer.n_encoder_layers)]
        ))

        decoder_layer = nn.TransformerDecoderLayer(d_model=self.alph_embed_dim, nhead=opt.transformer.n_heads)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=opt.transformer.n_decoder_layers)
        
        # define the input dim
        input_3D_dim = 0
        if opt.arch.posenc.add_raw_3D_points:
            input_3D_dim += 3
        input_3D_dim += 6*opt.arch.posenc.L_3D if opt.arch.posenc.L_3D>0 else 0
        assert input_3D_dim > 0

        assert opt.nerf.view_dep
        input_view_dim = 0
        if opt.arch.posenc.add_raw_rays:
            input_view_dim += 3
        input_view_dim += 6*opt.arch.posenc.L_view if opt.arch.posenc.L_view>0 else 0
        assert input_view_dim > 0

        self.decoder_in_fc = nn.Sequential(
            nn.Linear(input_3D_dim+input_view_dim, self.alph_embed_dim), 
            nn.ReLU(), 
        )

        self.n_sur_pts = opt.voxel_sampler.n_sur_pts
        self.n_nxt_ray_pts = opt.voxel_sampler.n_nxt_ray_pts

        print("[Transformer Config]: alph_embed_dim: {}. rgb_embed_dim: {}. density_activ: {}. ".format(
            self.alph_embed_dim, self.rgb_embed_dim, opt.transformer.density_activ
        ))
        
    def __merge_enc_outputs(self, x): 
        return x

    def __surround_embedding(self, x): 
        return torch.max(x, dim=1)[0]
    
    def positional_encoding(self, opt: Dict[str, Any], input: torch.Tensor, 
                            embedder_fn: Callable[[Dict[str, Any], torch.Tensor, int], torch.Tensor], 
                            L: int) -> torch.Tensor: # [B,...,N]
        """Apply the coarse-to-fine positional encoding strategy of BARF. 

        Args:
            opt (edict): settings
            input (torch.Tensor): shaps is (B, ..., C) where C is channel dimension
            embedder_fn (function): positional encoding function
            L (int): Number of frequency basis
        returns:
            positional encoding
        """
        shape = input.shape
        input_enc = embedder_fn(opt, input, L) # [B,...,2NL]

        # [B,...,2NL]
        # coarse-to-fine: smoothly mask positional encoding for BARF
        # the progress could also be fixed but then we finetune without cf_pe, then it is just not updated 
        if opt.barf_c2f is not None:
            # set weights for different frequency bands
            start,end = opt.barf_c2f
            alpha = (self.progress.data-start)/(end-start)*L
            k = torch.arange(L,dtype=torch.float32,device=input.device)
            weight = (1-(alpha-k).clamp_(min=0,max=1).mul_(np.pi).cos_())/2

            # apply weights
            shape = input_enc.shape
            input_enc = (input_enc.view(-1,L)*weight).view(*shape)
        return input_enc
    
    def forward(self, opt: Dict[str, Any], 
                sur_pts: torch.Tensor, alph: torch.Tensor, alph_embed: torch.Tensor, 
                rgb: torch.Tensor, rgb_embed: torch.Tensor, viewdirs: torch.Tensor, 
                embedder_pts: Callable[[Dict[str, Any], torch.Tensor, int], torch.Tensor],
                embedder_view: Callable[[Dict[str, Any], torch.Tensor, int], torch.Tensor], 
                mode: str = None) -> Dict[str, Any]: # [B,...,3])

        # [N_rays, 1+n_sur_pts+n_nxt_ray_pts, ...]

        N_rays = sur_pts.shape[0]
        
        sur_embed_alph = alph_embed[:, 1:(1+self.n_sur_pts)] # [N_rays, n_sur_pts, alph_embed_dim]
        sur_embed = sur_embed_alph

        # input to the decoder
        ref_xyz = sur_pts[:, [0]] # [N_rays, 1, 3]
        nxt_ray_pts_xyz = sur_pts[:, (1+self.n_sur_pts):] # [N_rays, n_nxt_ray_pts, 3]
        dec_in_xyz = torch.cat([ref_xyz, nxt_ray_pts_xyz], dim=1) # [N_rays, 1+n_nxt_ray_pts, 3]
        dec_in_xyz_flat = torch.reshape(dec_in_xyz, [-1, dec_in_xyz.shape[-1]])
        if opt.arch.posenc.L_3D > 0: 
            points_3D_enc = self.positional_encoding(opt, dec_in_xyz_flat, embedder_fn=embedder_pts, 
                                                     L=opt.arch.posenc.L_3D)
            if opt.arch.posenc.add_raw_3D_points:
                points_3D_enc = torch.cat([dec_in_xyz_flat, points_3D_enc],dim=-1) # [B,...,6L+3]
        dec_in_xyz_flat = points_3D_enc

        ref_viewdirs = viewdirs[:, [0]]
        nxt_ray_pts_viewdirs = viewdirs[:, (1+self.n_sur_pts):]
        dec_in_viewdirs = torch.cat([ref_viewdirs, nxt_ray_pts_viewdirs], dim=1)
        dec_in_viewdirs_flat = torch.reshape(dec_in_viewdirs, [-1, dec_in_viewdirs.shape[-1]])
        if opt.arch.posenc.L_view > 0: 
            ray_enc = self.positional_encoding(opt, dec_in_viewdirs_flat, embedder_fn=embedder_view, 
                                                   L=opt.arch.posenc.L_view)
            if opt.arch.posenc.add_raw_rays:
                ray_enc = torch.cat([dec_in_viewdirs_flat, ray_enc],dim=-1) # [B,...,6L+3]
        dec_in_viewdirs_flat = ray_enc

        decoder_inputs = torch.cat([
            torch.reshape(dec_in_xyz_flat, [N_rays, 1+self.n_nxt_ray_pts, -1]), 
            torch.reshape(dec_in_viewdirs_flat, [N_rays, 1+self.n_nxt_ray_pts, -1])], dim=-1)
        decoder_inputs = self.decoder_in_fc(decoder_inputs)


        # encoder output
        encoder_outputs = self.encoder(sur_embed) # [N_rays, n_sur_pts, embed_dim]
        surround_embed = self.__surround_embedding(encoder_outputs) # [N_rays, embed_dim]
        
        # encoder prediction
        encoder_alph = self.alph_fc(encoder_outputs) # actually useless
        encoder_rgb_proj = self.rgb_proj(encoder_outputs)
        encoder_rgb = self.rgb_fc(encoder_rgb_proj)

        # decoder output
        decoder_outputs = self.decoder(torch.transpose(decoder_inputs, 0, 1), torch.transpose(encoder_outputs, 0, 1))
        decoder_outputs = torch.transpose(decoder_outputs, 0, 1) # [N_rays, 1+n_nxt_ray_pts, 256]
        
        # decoder prediction
        decoder_alph = self.alph_fc(decoder_outputs) # [N_rays, 1+n_nxt_ray_pts, 1]
        decoder_rgb_proj = self.rgb_proj(decoder_outputs) # [N_rays, 1+n_nxt_ray_pts, 128]
        decoder_rgb = self.rgb_fc(decoder_rgb_proj) # [N_rays, 1+n_nxt_ray_pts, 3]

        alph_proj = torch.cat([decoder_outputs[:, [0]], encoder_outputs[:, :self.n_sur_pts], decoder_outputs[:, 1:]], dim=1)
        alph = torch.cat([decoder_alph[:, [0]], encoder_alph[:, :self.n_sur_pts], decoder_alph[:, 1:]], dim=1)
        rgb_proj = torch.cat([decoder_rgb_proj[:, [0]], encoder_rgb_proj[:, :self.n_sur_pts], decoder_rgb_proj[:, 1:]], dim=1)
        rgb = torch.cat([decoder_rgb[:, [0]], encoder_rgb[:, :self.n_sur_pts], decoder_rgb[:, 1:]], dim=1)

        return {
            "alph": alph, # [N_rays, 1+n_sur_pts+n_nxt_ray_pts, 1]
            "alph_proj": alph_proj, # [N_rays, 1+n_sur_pts+n_nxt_ray_pts, 256]
            "rgb": rgb, # [N_rays, 1+n_sur_pts+n_nxt_ray_pts, 3]
            "rgb_proj": rgb_proj, # [N_rays, 1+n_sur_pts+n_nxt_ray_pts, 128]
            "surround_embed": surround_embed, # [N_rays, embed_dim]
        }
