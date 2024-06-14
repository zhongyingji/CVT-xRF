import torch
# torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Misc
img2mse = lambda x, y : torch.mean((x - y) ** 2)
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)

# Positional encoding (section 5.1)
class Embedder(nn.Module):
    def __init__(self, **kwargs):
        super(Embedder, self).__init__()
        self.kwargs = kwargs
        self.create_embedding_fn()
        self.progress = torch.nn.Parameter(torch.tensor(0.)) # actually useless
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)

def get_embedder(multires, i=0):
    if i == -1:
        return nn.Identity(), 3
    
    embed_kwargs = {
                'include_input' : True,
                'input_dims' : 3,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }
    
    embedder_obj = Embedder(**embed_kwargs)
    # embed = lambda x, eo=embedder_obj : eo.embed(x)
    # return embed, embedder_obj.out_dim
    return embedder_obj

class Embedder_BARF(nn.Module):
    def __init__(self, **kwargs):
        super(Embedder_BARF, self).__init__()
        self.kwargs = kwargs
        self.create_embedding_fn()
        self.progress = torch.nn.Parameter(torch.tensor(0.))
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0

        if self.kwargs['include_input']:
            # embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            if self.kwargs['include_pi_in_posenc']: 
                freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs) * np.pi
            else: 
                freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
        # [x, y, z, sin(x), sin(y), sin(z), cos(x), cos(y), cos(z), sin(2x), sin(2y), sin(2z), ...]
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        input_enc = torch.cat([fn(inputs) for fn in self.embed_fns], -1)
        # input_enc: [N, L*6]
        start, end = self.kwargs['barf_c2f']
        L = self.kwargs['num_freqs']

        input_enc = input_enc.reshape(-1, L, 6) # [N, L, 6]

        if end < 0: 
            input_enc = input_enc.reshape(-1, L*6) 
            return torch.cat([inputs, input_enc], dim=-1)

        alpha = (self.progress.data-start)/(end-start)*L
        k = torch.arange(L, dtype=torch.float32) # [L, ]
        weight = (1-(alpha-k).clamp_(min=0,max=1).mul_(np.pi).cos_())/2
        input_enc = (input_enc*weight[:, None]).reshape(-1, L*6)
        return torch.cat([inputs, input_enc], dim=-1)

def get_embedder_barf(multires, i=0, include_pi_in_posenc=True, barf_c2f=[0.1, 0.5]):
    if i == -1:
        return nn.Identity(), 3
    
    embed_kwargs = {
                'include_input' : True,
                'input_dims' : 3,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
                'include_pi_in_posenc': include_pi_in_posenc, 
                'barf_c2f': barf_c2f, 
    }
    
    embedder_obj = Embedder_BARF(**embed_kwargs)
    # embed = lambda x, eo=embedder_obj : eo.embed(x)
    # return embed, embedder_obj.out_dim
    return embedder_obj

class SelfAttentionEncoder(nn.Module):
    def __init__(self, pe_dim, n_heads=1, dim_ff=256, n_encoder_layers=2, 
        n_decoder_layers=2, 
        n_sur_pts=9, 
        n_nxt_ray_pts=9, 
        alph_embed_dim=256, 
        rgb_embed_dim=128, 
        pe_multires=4, 
        pe_multires_views=3, 
        encoder_input_method="concat", 
        use_barf_pe=False, 
        include_pi_in_posenc=False, 
        barf_c2f=[0.1, 0.5]): 
        # learnable_param_dim=32, 
        # enc_input_cat_coarse=False, # whether to concat the alpha feature from coarse mlp
        # dec_input_mlp="coarse", # should be coarse prediction, since the decoder has to predict the density & color rather than from mlp
        # dec_input_mlp_detach=True, # detach the decoder input of mlp prediction
        # dec_input_xyzview_pe=False, # whether to use positional encoding for decoder input
        # dec_input_cat_coarse=True, # whether to concat the coarse mlp feature in decoder input
        # ):
        
        super(SelfAttentionEncoder, self).__init__()
        
        self.encoder_input_method = encoder_input_method
        if encoder_input_method == "concat":
            self.rgb_in_fc = nn.Identity()
            self.encoder_in_fc = nn.Sequential(
                nn.Linear(alph_embed_dim+rgb_embed_dim, alph_embed_dim),
                nn.ReLU())
        elif encoder_input_method == "flatten":
            self.rgb_in_fc = nn.Sequential(
                nn.Linear(rgb_embed_dim, alph_embed_dim),
                nn.ReLU())
            self.encoder_in_fc = nn.Identity()
        elif encoder_input_method == "rgb_embed":
            self.rgb_in_fc = nn.Sequential(
                nn.Linear(rgb_embed_dim, alph_embed_dim),
                nn.ReLU())
            self.encoder_in_fc = nn.Identity()
        elif encoder_input_method == "alph_embed":
            self.rgb_in_fc = nn.Identity()
            self.encoder_in_fc = nn.Identity()
        else: 
            raiseNotImplementedError("encoder_input_method: [concat, flatten, rgb_embed, alph_embed]")

        self.encoder = nn.Sequential(*nn.ModuleList(
            [nn.TransformerEncoderLayer(
                d_model=alph_embed_dim,
                nhead=n_heads,
                dim_feedforward=dim_ff,
                batch_first=True) for _ in range(n_encoder_layers)]))

        # projection head
        self.alph_fc = nn.Linear(alph_embed_dim, 1)
        self.rgb_proj = nn.Sequential(
            nn.Linear(alph_embed_dim, rgb_embed_dim), 
            nn.ReLU())
        self.rgb_fc = nn.Linear(rgb_embed_dim, 3)

        self.n_sur_pts = n_sur_pts
        self.n_nxt_ray_pts = n_nxt_ray_pts
        self.alph_embed_dim = alph_embed_dim
        self.rgb_embed_dim = rgb_embed_dim

        decoder_layer = nn.TransformerDecoderLayer(d_model=self.alph_embed_dim, nhead=n_heads)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_decoder_layers)

        if use_barf_pe:
            pe_embedder = lambda x: self.get_pe_embedder_barf(
                                x, include_pi_in_posenc=include_pi_in_posenc, barf_c2f=barf_c2f)
        else:
            pe_embedder = self.get_pe_embedder

        self.xyz_embedder = pe_embedder(pe_multires)
        self.xyz_embedder_fn, self.xyz_embed_dim = self.xyz_embedder.embed, self.xyz_embedder.out_dim
        self.view_embedder = pe_embedder(pe_multires_views)
        self.view_embedder_fn, self.view_embed_dim = self.view_embedder.embed, self.view_embedder.out_dim

        self.decoder_in_fc = nn.Sequential(
            nn.Linear(self.xyz_embed_dim+self.view_embed_dim, alph_embed_dim),
            nn.ReLU())

    def get_pe_embedder_barf(self, multires, i=0, include_pi_in_posenc=True, barf_c2f=[0.1, 0.5]):
        if i == -1:
            return nn.Identity(), 3
        
        embed_kwargs = {
                    'include_input' : True,
                    'input_dims' : 3,
                    'max_freq_log2' : multires-1,
                    'num_freqs' : multires,
                    'log_sampling' : True,
                    'periodic_fns' : [torch.sin, torch.cos],
                    'include_pi_in_posenc': include_pi_in_posenc, 
                    'barf_c2f': barf_c2f, 
        }
        
        embedder_obj = Embedder_BARF(**embed_kwargs)
        return embedder_obj
        
    def get_pe_embedder(self, multires, i=0): 
        if i == -1:
            return nn.Identity(), 3
        
        embed_kwargs = {
                    'include_input' : False,
                    'input_dims' : 3,
                    'max_freq_log2' : multires-1,
                    'num_freqs' : multires,
                    'log_sampling' : True,
                    'periodic_fns' : [torch.sin, torch.cos],
        }

        embedder_obj = Embedder(**embed_kwargs)
        return embedder_obj

    def _surround_embedding(self, x):
        # x: (N_rays, n_sur_pts, embed_dim)
        # return torch.mean(x, dim=1) # [N_rays, embed_dim]
        return torch.max(x, dim=1)[0]
    
    def forward(self, x):
        # x: (N_rays*(1+n_sur_pts+n_nxt_ray_pts), 4+embed_dim+3+embed_dim/2+3)
        x = torch.reshape(x, [x.shape[0]//(1+self.n_sur_pts+self.n_nxt_ray_pts), \
                                1+self.n_sur_pts+self.n_nxt_ray_pts, \
                                4+self.alph_embed_dim+3+self.rgb_embed_dim+3+self.alph_embed_dim])
        sur_pts, alph, alph_embed, rgb, rgb_embed, viewdirs, coarse_alph_embed = torch.split(x, \
                                [3, 1, self.alph_embed_dim, 3, self.rgb_embed_dim, 3, self.alph_embed_dim], dim=-1)
        # (N_rays, 1+n_sur_pts+n_nxt_ray_pts, ...)
        N_rays = sur_pts.shape[0]

        sur_embed_alph = alph_embed[:, 1:(1+self.n_sur_pts)] # (N_rays, n_sur_pts, alph_embed_dim)
        sur_embed_rgb = rgb_embed[:, 1:(1+self.n_sur_pts)] # (N_rays, n_sur_pts, rgb_embed_dim)
        
        if self.encoder_input_method == "concat":
            sur_embed = torch.cat([sur_embed_alph, sur_embed_rgb], dim=-1) 
            # (N_rays, n_sur_pts, alph_embed_dim+rgb_embed_dim)
        elif self.encoder_input_method == "flatten":
            sur_embed = torch.cat([sur_embed_alph, sur_embed_rgb], dim=1)
            # (N_rays, 2*n_sur_pts, alph_embed_dim)
        elif self.encoder_input_method == "rgb_embed": 
            sur_embed = sur_embed_rgb
        elif self.encoder_input_method == "alph_embed":
            sur_embed = sur_embed_alph
        sur_embed = self.encoder_in_fc(self.rgb_in_fc(sur_embed))
        # (N_rays, n_sur_pts, alph_embed_dim)

        ref_xyz = sur_pts[:, [0]] # (N_rays, 1, 3)
        nxt_ray_pts_xyz = sur_pts[:, (1+self.n_sur_pts):] # (N_rays, n_nxt_ray_pts, 3)
        dec_in_xyz = torch.cat([ref_xyz, nxt_ray_pts_xyz], dim=1) # (N_rays, 1+n_nxt_ray_pts, 3)
        dec_in_xyz_flat = torch.reshape(dec_in_xyz, [-1, dec_in_xyz.shape[-1]])
        dec_in_xyz_flat = self.xyz_embedder_fn(dec_in_xyz_flat)

        ref_viewdirs = viewdirs[:, [0]]
        nxt_ray_pts_viewdirs = viewdirs[:, (1+self.n_sur_pts):]
        dec_in_viewdirs = torch.cat([ref_viewdirs, nxt_ray_pts_viewdirs], dim=1)
        dec_in_viewdirs_flat = torch.reshape(dec_in_viewdirs, [-1, dec_in_viewdirs.shape[-1]])
        dec_in_viewdirs_flat = self.view_embedder_fn(dec_in_viewdirs_flat)

        decoder_inputs = torch.cat([
            torch.reshape(dec_in_xyz_flat, [N_rays, 1+self.n_nxt_ray_pts, -1]), 
            torch.reshape(dec_in_viewdirs_flat, [N_rays, 1+self.n_nxt_ray_pts, -1])], dim=-1)
        decoder_inputs = self.decoder_in_fc(decoder_inputs)

        encoder_outputs = self.encoder(sur_embed) # (N_rays, n_sur_pts, embed_dim)
        surround_embed = self._surround_embedding(encoder_outputs) # (N_rays, embed_dim)
        # not correct if encoder_input_method is flatten
       
        # placeholder, actually useless on encoder side
        encoder_alph = self.alph_fc(encoder_outputs) 
        encoder_rgb_proj = self.rgb_proj(encoder_outputs)
        encoder_rgb = self.rgb_fc(encoder_rgb_proj)

        decoder_outputs = self.decoder(torch.transpose(decoder_inputs, 0, 1), torch.transpose(encoder_outputs, 0, 1))
        decoder_outputs = torch.transpose(decoder_outputs, 0, 1) # (N_rays, 1+n_nxt_ray_pts, 256)
        
        decoder_alph = self.alph_fc(decoder_outputs) # (N_rays, 1+n_nxt_ray_pts, 1)
        decoder_rgb_proj = self.rgb_proj(decoder_outputs) # (N_rays, 1+n_nxt_ray_pts, 128)
        decoder_rgb = self.rgb_fc(decoder_rgb_proj) # (N_rays, 1+n_nxt_ray_pts, 3)
        
        # reorganize into the order of "ref_pts, sur_pts, nxt_ray_pts"
        alph_proj = torch.cat([decoder_outputs[:, [0]], encoder_outputs[:, :self.n_sur_pts], decoder_outputs[:, 1:]], dim=1)
        alph = torch.cat([decoder_alph[:, [0]], encoder_alph[:, :self.n_sur_pts], decoder_alph[:, 1:]], dim=1)
        rgb_proj = torch.cat([decoder_rgb_proj[:, [0]], encoder_rgb_proj[:, :self.n_sur_pts], decoder_rgb_proj[:, 1:]], dim=1)
        rgb = torch.cat([decoder_rgb[:, [0]], encoder_rgb[:, :self.n_sur_pts], decoder_rgb[:, 1:]], dim=1)

        return alph, alph_proj, rgb, rgb_proj, surround_embed
        
# Model
class NeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=False):
        """ 
        """
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        
        # self.pts_linears = nn.ModuleList(
        #     [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)])
        
        input_linear = nn.Linear(input_ch, W)
        self.tensorflow_init_weights(None, input_linear, None)
        self.pts_linears = [input_linear]
        for i in range(D-1): 
            linear = nn.Linear(W, W) if i not in self.skips else nn.Linear(W+input_ch, W)
            self.tensorflow_init_weights(None, linear, None)
            self.pts_linears.append(linear)
        
        self.pts_linears = nn.ModuleList(self.pts_linears)

        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        # self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W//2)])
        views_input_linear = nn.Linear(input_ch_views + W, W//2)
        self.tensorflow_init_weights(None, views_input_linear, None)
        self.views_linears = nn.ModuleList([views_input_linear])

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])
        
        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.tensorflow_init_weights(None, self.feature_linear, None)

            self.alpha_linear = nn.Linear(W, 1)
            self.tensorflow_init_weights(None, self.alpha_linear, "first")

            self.rgb_linear = nn.Linear(W//2, 3)
            self.tensorflow_init_weights(None, self.rgb_linear, "all")
        else:
            self.output_linear = nn.Linear(W, output_ch)
    
    # From SPARF, source/models/frequency_nerf.py
    def tensorflow_init_weights(self, opt, linear, out):
        # use Xavier init instead of Kaiming init
        relu_gain = torch.nn.init.calculate_gain("relu") # sqrt(2)
        if out=="all":
            torch.nn.init.xavier_uniform_(linear.weight)
        elif out=="first":
            torch.nn.init.xavier_uniform_(linear.weight[:1])
            torch.nn.init.xavier_uniform_(linear.weight[1:],gain=relu_gain)
        else:
            torch.nn.init.xavier_uniform_(linear.weight,gain=relu_gain)
        torch.nn.init.zeros_(linear.bias)
        return

    def forward(self, x):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)
        # TODO: dirty code
        feature_alph = h
        feature_rgb = h
        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)
        
            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)
            feature_rgb = h
            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        return outputs, feature_alph, feature_rgb

    def load_weights_from_keras(self, weights):
        assert self.use_viewdirs, "Not implemented if use_viewdirs=False"
        
        # Load pts_linears
        for i in range(self.D):
            idx_pts_linears = 2 * i
            self.pts_linears[i].weight.data = torch.from_numpy(np.transpose(weights[idx_pts_linears]))    
            self.pts_linears[i].bias.data = torch.from_numpy(np.transpose(weights[idx_pts_linears+1]))
        
        # Load feature_linear
        idx_feature_linear = 2 * self.D
        self.feature_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_feature_linear]))
        self.feature_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_feature_linear+1]))

        # Load views_linears
        idx_views_linears = 2 * self.D + 2
        self.views_linears[0].weight.data = torch.from_numpy(np.transpose(weights[idx_views_linears]))
        self.views_linears[0].bias.data = torch.from_numpy(np.transpose(weights[idx_views_linears+1]))

        # Load rgb_linear
        idx_rbg_linear = 2 * self.D + 4
        self.rgb_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear]))
        self.rgb_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear+1]))

        # Load alpha_linear
        idx_alpha_linear = 2 * self.D + 6
        self.alpha_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear]))
        self.alpha_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear+1]))

# Ray helpers
def get_rays(H, W, K, c2w):
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    return rays_o, rays_d

def get_rays_np(H, W, K, c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))
    return rays_o, rays_d

def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[...,2]) / rays_d[...,2]
    rays_o = rays_o + t[...,None] * rays_d
    
    # Projection
    o0 = -1./(W/(2.*focal)) * rays_o[...,0] / rays_o[...,2]
    o1 = -1./(H/(2.*focal)) * rays_o[...,1] / rays_o[...,2]
    o2 = 1. + 2. * near / rays_o[...,2]

    d0 = -1./(W/(2.*focal)) * (rays_d[...,0]/rays_d[...,2] - rays_o[...,0]/rays_o[...,2])
    d1 = -1./(H/(2.*focal)) * (rays_d[...,1]/rays_d[...,2] - rays_o[...,1]/rays_o[...,2])
    d2 = -2. * near / rays_o[...,2]
    
    rays_o = torch.stack([o0,o1,o2], -1)
    rays_d = torch.stack([d0,d1,d2], -1)
    
    return rays_o, rays_d

# Hierarchical sampling (section 5.2)
def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    # Get pdf
    weights = weights + 1e-5 # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

    # Pytest, overwrite u with numpy's fixed random numbers
    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0., 1., N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = torch.Tensor(u)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[...,1]-cdf_g[...,0])
    denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[...,0])/denom
    samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])

    return samples

