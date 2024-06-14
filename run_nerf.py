import os, sys
import numpy as np
import imageio
import json
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange
import matplotlib.pyplot as plt

from dataloader.dtu_loader import DTUDatasetSPARF, DTUDatasetDefault
from dataloader.blender_loader import BlenderDatasetDefault
from dataloader.sampler import VoxelSampler0, VoxelSamplerDefault

from run_nerf_helpers import *
from utils.loss_helper import ContrastiveLoss, TripletLoss
from utils import camera
from utils.eval import psnr_fn, ssim_fn, lpips_fn, psnr_mask_fn 
from utils.save_ply import save_point_cloud_in_ply

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEBUG = False

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True
    
# set_seed(20230114)

def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn
    # def ret(inputs):
    #     return torch.cat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
    # return ret
    def ret_with_alph_embed(inputs):
        outputs = [fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)]
        return torch.cat([output[0] for output in outputs], 0), \
                    torch.cat([output[1] for output in outputs], 0), \
                    torch.cat([output[2] for output in outputs], 0)
    return ret_with_alph_embed

def batchify_att(fn, chunk):
    if chunk is None:
        return fn
    # def ret(inputs):
    #     return torch.cat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
    # return ret
    def ret_with_atten_embed(inputs):
        outputs = [fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)]
        return torch.cat([output[0] for output in outputs], 0), \
                    torch.cat([output[1] for output in outputs], 0), \
                    torch.cat([output[2] for output in outputs], 0), \
                    torch.cat([output[3] for output in outputs], 0), \
                    torch.cat([output[4] for output in outputs], 0)
    return ret_with_atten_embed

def run_self_attention(
    sur_pts, alph, alph_embed, 
    rgb, rgb_embed, viewdirs, 
    coarse_alph_embed, 
    fn, netchunk=1024*64):
    """Args: 
       sur_pts: (N_rays, 1+n_sur_pts, 3). It is used to get position encoding. 
       alph: (N_rays, 1+n_sur_pts, 1). 
       alph_embed: (N_rays, 1+n_sur_pts, embed_dim). 
       rgb: (N_rays, 1+n_sur_pts, 3). 
       rgb_embed: (N_rays, 1+n_sur_pts, embed_dim/2). 
       viewdirs: (N_rays, 3). 
       coarse_alph_embed: (N_rays, 1+n_sur_pts, embed_dim). 
       fn: self attention module.
       outputs: (N_rays, 1). The predicted alpha value.  
    """
    viewdirs = viewdirs[:, None].expand(sur_pts.shape)
    inputs = torch.cat([sur_pts, alph, alph_embed, \
                        rgb, rgb_embed, viewdirs, coarse_alph_embed], -1) 
                        # (N_rays, 1+n_sur_pts, 4+embed_dim+3+embed_dim/2+3+embed_dim)
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]]) # [all_points, 4+embed_dim+3+embed_dim/2+3]
    alph, alph_proj, rgb, rgb_proj, surround_embed = \
                batchify_att(fn, netchunk)(inputs_flat) 
    # alph: (N_rays, (1+n_sur_pts), 1)
    # alph_proj: (N_rays, (1+n_sur_pts), 256)
    return alph, alph_proj, rgb, rgb_proj, surround_embed

def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024*64):
    """Prepares inputs and applies network 'fn'.
    """
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    embedded = embed_fn(inputs_flat)

    if viewdirs is not None:
        input_dirs = viewdirs[:,None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat([embedded, embedded_dirs], -1)

    rgba_flat, alph_embed_flat, rgb_embed_flat = batchify(fn, netchunk)(embedded)
    rgba = torch.reshape(rgba_flat, list(inputs.shape[:-1]) + [rgba_flat.shape[-1]])
    alph_embed = torch.reshape(alph_embed_flat, list(inputs.shape[:-1]) + [alph_embed_flat.shape[-1]])
    rgb_embed = torch.reshape(rgb_embed_flat, list(inputs.shape[:-1]) + [rgb_embed_flat.shape[-1]])
    return rgba, alph_embed, rgb_embed
    
def batchify_rays(rays_flat, chunk=1024*32, **kwargs):
    """Render rays in smaller minibatches to avoid OOM.
    """
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(rays_flat[i:i+chunk], **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret

def render(H, W, K, chunk=1024*32, rays=None, c2w=None, ndc=True,
                  near=0., far=1.,
                  use_viewdirs=False, c2w_staticcam=None, 
                  surround=None, 
                  **kwargs):
    """Render rays
    Args:
      H: int. Height of image in pixels.
      W: int. Width of image in pixels.
      focal: float. Focal length of pinhole camera.
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
      rays: array of shape [2, batch_size, 3]. Ray origin and direction for
        each example in batch.
      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
      ndc: bool. If True, represent ray origin, direction in NDC coordinates.
      near: float or array of shape [batch_size]. Nearest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.
      use_viewdirs: bool. If True, use viewing direction of a point in space in model.
      c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for 
       camera while using other c2w argument for viewing directions.
      surround: array of shape [batch_size, 4+3*(n_sur_pts+n_nxt_ray_pts)+n_nxt_ray_pts+1]. 
        For the ray sampled from a voxel, mid points of in and out coord, its t value, surrounding 
        points, sampled points of next ray, their t values, and the vox_end flag. 
    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything returned by render_rays().
    """
    if c2w is not None:   
        # c2w is an indicator of testing
        # the rays_o and rays_d are generated in render_path
        if rays is not None: 
            rays_o, rays_d = rays
        else: 
            rays_o, rays_d = get_rays(H, W, K, c2w)
    else:
        rays_o, rays_d = rays

    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays(H, W, K, c2w_staticcam)
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1,3]).float()

    sh = rays_d.shape # [..., 3]
    # if ndc:
    # Before training, the rays have been converted into ndc rays
    # in dataset; During testing, the rays have not been converted
    if ndc and (c2w is not None):
        # for forward facing scenes
        rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1,3]).float()
    rays_d = torch.reshape(rays_d, [-1,3]).float()

    near, far = near * torch.ones_like(rays_d[...,:1]), far * torch.ones_like(rays_d[...,:1])
    rays = torch.cat([rays_o, rays_d, near, far], -1)
    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)

    if surround is None:
        # placeholder. actually useless during testing
        # the shape of surround points is: (N_rays, 4+3*n_sur_pts)
        n_sur_pts = kwargs["n_sur_pts"]
        n_nxt_ray_pts = kwargs["n_nxt_ray_pts"]
        surround = torch.randn(rays.shape[0], 4+3*(n_sur_pts+n_nxt_ray_pts)+n_nxt_ray_pts+1)
    rays = torch.cat([rays, surround], -1)

    # Render and reshape
    all_ret = batchify_rays(rays, chunk, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map', 'disp_map', 'acc_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k : all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]

def render_path(render_poses, hwf, K, chunk, render_kwargs, 
                gt_imgs=None, savedir=None, render_factor=0, 
                w2c=None, intr=None, 
                point_cloud_vis=False, points_sample_skip=1):

    H, W, focal = hwf

    if render_factor!=0:
        # Render downsampled for speed
        H = H//render_factor
        W = W//render_factor
        focal = focal/render_factor
    
    if w2c is not None: 
        if len(intr.shape) == 2:
            intr = intr.unsqueeze(0)
        rays_o, rays_d = camera.get_center_and_ray(
                pose_w2c=w2c, H=H, W=W, intr=intr)
        rays_o = rays_o.reshape(-1, H, W, 3)
        rays_d = rays_d.reshape(-1, H, W, 3)

    rgbs = []
    disps = []
    accum_psnr = []
    # t = time.time()
    for i, c2w in enumerate(tqdm(render_poses)):
        # print(i, time.time() - t)
        # t = time.time()
        if w2c is not None: 
            rgb, disp, acc, extras = render(H, W, K, chunk=chunk, c2w=c2w[:3,:4], 
                                            rays=[rays_o[i], rays_d[i]], 
                                            retraw=True, **render_kwargs)
        else: 
            rgb, disp, acc, extras = render(H, W, K, chunk=chunk, c2w=c2w[:3,:4], 
                                            retraw=True, **render_kwargs)
        rgbs.append(rgb.cpu().numpy())
        disps.append(disp.cpu().numpy())

        if i==0:
            print(rgb.shape, disp.shape)
        
        # if gt_imgs is not None and render_factor==0:
        #     p = -10. * np.log10(np.mean(np.square(rgb.cpu().numpy() - gt_imgs[i].cpu().numpy())))
        #     accum_psnr.append(p)
        #     print(p)
        
        if savedir is not None:
            rgb8 = to8b(rgbs[-1])
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)
            if point_cloud_vis: 
                save_point_cloud_in_ply(extras, H, W, savedir, '{:03d}'.format(i), points_sample_skip)
            
    # print("The average psnr of {} poses is: {}".format(len(render_poses), np.mean(accum_psnr)))
    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)
    return rgbs, disps

def create_nerf(args):
    """Instantiate NeRF's MLP model.
    """
    if args.use_barf_pe: 
        embedder = get_embedder_barf(args.multires, args.i_embed, 
                include_pi_in_posenc=not args.not_include_pi_in_posenc, 
                barf_c2f=[args.barf_pe_coarse, args.barf_pe_fine])
    else: 
        embedder = get_embedder(args.multires, args.i_embed)
    embed_fn, input_ch = embedder.embed, embedder.out_dim

    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        if args.use_barf_pe: 
            embedder_views = get_embedder_barf(args.multires_views, args.i_embed, 
                        include_pi_in_posenc=not args.not_include_pi_in_posenc, 
                        barf_c2f=[args.barf_pe_coarse, args.barf_pe_fine])
        else: 
            embedder_views = get_embedder(args.multires_views, args.i_embed)
        embeddirs_fn, input_ch_views = embedder_views.embed, embedder_views.out_dim

    output_ch = 5 if args.N_importance > 0 else 4
    skips = [4]
    model = NeRF(D=args.netdepth, W=args.netwidth,
                 input_ch=input_ch, output_ch=output_ch, skips=skips,
                 input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
    grad_vars = list(model.parameters())

    model_fine = None
    if args.N_importance > 0:
        model_fine = NeRF(D=args.netdepth_fine, W=args.netwidth_fine,
                          input_ch=input_ch, output_ch=output_ch, skips=skips,
                          input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
        grad_vars += list(model_fine.parameters())

    network_query_fn = lambda inputs, viewdirs, network_fn : run_network(inputs, viewdirs, network_fn,
                                                                embed_fn=embed_fn,
                                                                embeddirs_fn=embeddirs_fn,
                                                                netchunk=args.netchunk)
    assert args.netwidth == args.netwidth_fine
    model_self_attention = SelfAttentionEncoder(pe_dim=None, 
                                                n_heads=args.n_heads, 
                                                dim_ff=args.dim_ff, 
                                                n_encoder_layers=args.n_encoder_layers, 
                                                n_decoder_layers=args.n_decoder_layers, 
                                                n_sur_pts=args.n_sur_pts, 
                                                n_nxt_ray_pts=args.n_nxt_ray_pts, 
                                                alph_embed_dim=args.netwidth, 
                                                rgb_embed_dim=args.netwidth//2, 
                                                pe_multires=args.multires, 
                                                pe_multires_views=args.multires_views, 
                                                encoder_input_method=args.encoder_input_method,
                                                use_barf_pe=args.transformer_use_barf_pe, 
                                                include_pi_in_posenc=not args.not_include_pi_in_posenc, 
                                                barf_c2f=[args.barf_pe_coarse, args.barf_pe_fine], 
                                                # learnable_param_dim=args.decoder_learnable_param_dim, 
                                                # enc_input_cat_coarse=args.encoder_input_concat_coarse, 
                                                # dec_input_mlp=args.decoder_input_mlp, 
                                                # dec_input_mlp_detach=args.decoder_input_mlp_detach, 
                                                # dec_input_xyzview_pe=args.decoder_input_xyzview_pe, 
                                                # dec_input_cat_coarse=not args.decoder_input_not_concat_coarse, 
                                                ).to(device)
    grad_vars_sa = list(model_self_attention.parameters())

    self_attention_fn = lambda sur_pts, alph, alph_embed, rgb, rgb_embed, viewdirs, coarse_alph_embed: \
                                                            run_self_attention(sur_pts, alph, alph_embed, 
                                                                rgb, rgb_embed, viewdirs, coarse_alph_embed, 
                                                                fn=model_self_attention, netchunk=args.netchunk)
    
    # Create optimizer
    optimizer = torch.optim.Adam([
        {"params": grad_vars}, {"params": grad_vars_sa, "lr": args.lrate/2}], 
        lr=args.lrate, betas=(0.9, 0.999))

    start = 0
    basedir = args.basedir
    expname = args.expname

    ##########################

    # Load checkpoints
    if args.ft_path is not None and args.ft_path!='None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if 'tar' in f]

    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        # ckpt_path = ckpts[int(args.N_iters // 10000)-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

        start = ckpt['global_step']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        
        # Load model
        model.load_state_dict(ckpt['network_fn_state_dict'])
        if model_fine is not None:
            model_fine.load_state_dict(ckpt['network_fine_state_dict'])
        model_self_attention.load_state_dict(ckpt['self_attention_state_dict'])

        embedder.load_state_dict(ckpt['embedder'])
        embedder_views.load_state_dict(ckpt['embedder_views'])
        
    ##########################

    render_kwargs_train = {
        'network_query_fn' : network_query_fn,
        'perturb' : args.perturb,
        'N_importance' : args.N_importance,
        'network_fine' : model_fine,
        'N_samples' : args.N_samples,
        'network_fn' : model,
        'use_viewdirs' : args.use_viewdirs,
        'white_bkgd' : args.white_bkgd,
        'raw_noise_std' : args.raw_noise_std, 
        'n_sur_pts': args.n_sur_pts, 
        'n_nxt_ray_pts': args.n_nxt_ray_pts, 
        'embedder': embedder, 
        'embedder_views': embedder_views, 
    }

    # NDC only good for LLFF-style forward facing data
    if args.dataset_type != 'llff' or args.no_ndc:
        print('Not ndc!')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp

    render_kwargs_test = {k : render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.
    render_kwargs_train['self_attention_fn'] = self_attention_fn
    render_kwargs_train['self_attention_network'] = model_self_attention

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer

def multi_insertion(ray_rgb_alph, ray_zvals, insert_rgb, insert_alph, insert_ts): 
    """concat to the end and sort w.r.t. insert_ts
    Args: 
        ray_rgb_alph: [num_rays, num_samples along ray, 4]. 
        ray_zvals: [num_rays, num_samples along ray]. 
        insert_rgb: [num_rays, 1+n_nxt_ray_pts, 3]. 
        insert_alph: [num_rays, 1+n_nxt_ray_pts, 1]. 
        insert_ts: [num_rays, 1+n_nxt_ray_pts]. 
    """
    insert_rgb_alph = torch.cat([insert_rgb, insert_alph], dim=-1) # [num_rays, 1+n_nxt_ray_pts, 4]
    
    ray_rgb_alph = torch.cat([ray_rgb_alph, insert_rgb_alph], dim=1) # [num_rays, num_samples+1+n_nxt_ray_pts, 4]
    ray_zvals = torch.cat([ray_zvals, insert_ts], dim=1) # [num_rays, num_samples+1+n_nxt_ray_pts]
    
    nrays, nsamples, ndim = ray_rgb_alph.shape

    res_zvals, sort_idx = torch.sort(ray_zvals, dim=1) # [num_rays, num_samples+1+n_nxt_ray_pts]
    res = torch.gather(ray_rgb_alph, 1, sort_idx[..., None].expand(nrays, nsamples, ndim))

    return res, res_zvals

def insert_pt_to_ray(ray_rgb_alph, ray_zvals, insert_pts):
    """
    Args:  
        ray_rgb_alph: [num_rays, num_samples along ray, 4]. 
        ray_zvals: [num_rays, num_samples along ray]. 
        ---------------------------------------------
        dict of insert_pts: 
        rgb_ref: [num_rays, 1, 3]. The reference point of each ray.
        rgb_nxt_ray: [num_rays, n_nxt_ray_pts, 3]. 
        alph_ref: [num_rays, 1, 1]. The predicted alpha of each reference point. 
        alph_nxt_ray: [num_rays, n_nxt_ray_pts, 1].
        ts_ref: [num_rays, 1]. The t of each reference point. 
        ts_nxt_ray: [num_rays, n_nxt_ray_pts]. 
        vox_end: [num_rays, 1]
        ---------------------------------------------
        for a batch of rays, the first ray can only use its reference point to insert to the ray; 
        the other rays can use their reference points and the points from transformer of previous indices.

        # insert_rgb: [num_rays, 3]. 
        # insert_alph: [num_rays, 1]. 
        # insert_ts: [num_rays, 1]. 
    """

    insert_alph = torch.cat([insert_pts["alph_ref"], insert_pts["alph_nxt_ray"]], dim=1)
    insert_ts = torch.cat([insert_pts["ts_ref"], insert_pts["ts_nxt_ray"]], dim=1)
    insert_rgb = torch.cat([insert_pts["rgb_ref"], insert_pts["rgb_nxt_ray"]], dim=1)
    # [num_rays, 1+n_nxt_ray_pts, 3]

    res, res_zvals = multi_insertion(ray_rgb_alph, ray_zvals, insert_rgb, insert_alph, insert_ts)
    
    misc = {
        "transformer_output_alpha": F.relu(insert_alph), 
        "unsorted_raw": torch.cat([insert_rgb, insert_alph], dim=-1), 
        "unsorted_ts": insert_ts, 
    }

    return res, res_zvals, misc

def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """

    raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)

    dists = z_vals[...,1:] - z_vals[...,:-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape)], -1)  # [N_rays, N_samples]

    dists = dists * torch.norm(rays_d[...,None,:], dim=-1)
    rgb = torch.sigmoid(raw[...,:3])  # [N_rays, N_samples, 3]
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[...,3].shape) * raw_noise_std

        # Overwrite randomly sampled data if pytest
        if pytest:
            # np.random.seed(0)
            noise = np.random.rand(*list(raw[...,3].shape)) * raw_noise_std
            noise = torch.Tensor(noise)

    alpha = raw2alpha(raw[...,3] + noise, dists)  # [N_rays, N_samples]
    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]

    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1.-acc_map[...,None])

    return rgb_map, disp_map, acc_map, weights, depth_map

def render_rays(ray_batch,
                network_fn,
                network_query_fn,
                N_samples,
                retraw=False,
                lindisp=False,
                perturb=0.,
                N_importance=0,
                network_fine=None,
                self_attention_fn=None, 
                self_attention_network=None, 
                white_bkgd=False,
                raw_noise_std=0.,
                verbose=False,
                pytest=False, 
                n_sur_pts=9, 
                n_nxt_ray_pts=9, 
                embedder=None, 
                embedder_views=None):
    """Volumetric rendering.
    Args:
      ray_batch: array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, min
        dist, max dist, and unit-magnitude viewing direction.
        + mid points of in and out coord, its t value, surrounding 
        points.
      network_fn: function. Model for predicting RGB and density at each point
        in space.
      network_query_fn: function used for passing queries to network_fn.
      N_samples: int. Number of different times to sample along each ray.
      retraw: bool. If True, include model's raw, unprocessed predictions.
      lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
      perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
        random points in time.
      N_importance: int. Number of additional times to sample along each ray.
        These samples are only passed to network_fine.
      network_fine: "fine" network with same spec as network_fn.
      white_bkgd: bool. If True, assume a white background.
      raw_noise_std: ...
      verbose: bool. If True, print more debugging info.
    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
      disp_map: [num_rays]. Disparity map. 1 / depth.
      acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
      raw: [num_rays, num_samples, 4]. Raw predictions from model.
      rgb0: See rgb_map. Output for coarse model.
      disp0: See disp_map. Output for coarse model.
      acc0: See acc_map. Output for coarse model.
      z_std: [num_rays]. Standard deviation of distances along ray for each
        sample.
    """

    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:,0:3], ray_batch[:,3:6] # (N_rays, 3)
    viewdirs = ray_batch[:, 8:11] if ray_batch.shape[-1] > 8 else None
    bounds = torch.reshape(ray_batch[...,6:8], [-1,1,2])
    near, far = bounds[...,0], bounds[...,1] # (-1,1)
    ref_pts = ray_batch[:, 11:14].reshape(N_rays, 1, 3) # (N_rays, 1, 3)
    ref_ts = ray_batch[:, 14:15] # (N_rays, 1)
    n_aug_pts = n_sur_pts + n_nxt_ray_pts
    sur_pts = ray_batch[:, 15:(15+3*n_aug_pts)].reshape(N_rays, 3, -1) # (N_rays, 3, n_aug_pts)
    sur_pts = torch.transpose(sur_pts, 1, 2) # (N_rays, n_aug_pts, 3)
    nxt_ray_ts = ray_batch[:, (15+3*n_aug_pts):(15+3*n_aug_pts+n_nxt_ray_pts)].reshape(N_rays, n_nxt_ray_pts) # (N_rays, n_nxt_ray_pts)
    vox_end = ray_batch[:, -1:] # (N_rays, 1)  

    t_vals = torch.linspace(0., 1., steps=N_samples)
    if not lindisp:
        z_vals = near * (1.-t_vals) + far * (t_vals)
    else:
        z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))
    z_vals = z_vals.expand([N_rays, N_samples])

    if perturb > 0.:
        # get intervals between samples
        mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        upper = torch.cat([mids, z_vals[...,-1:]], -1)
        lower = torch.cat([z_vals[...,:1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape)

        # Pytest, overwrite u with numpy's fixed random numbers
        if pytest:
            # np.random.seed(0)
            t_rand = np.random.rand(*list(z_vals.shape))
            t_rand = torch.Tensor(t_rand)

        z_vals = lower + (upper - lower) * t_rand

    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # (N_rays, N_samples, 3)
    tmp_pts = torch.cat([ref_pts, sur_pts], dim=1) # (N_rays, 1+n_aug_pts, 3), n_aug_pts=n_sur_pts+n_nxt_ray_pts
    aug_pts = torch.cat([pts, tmp_pts], dim=1) # (N_rays, N_samples+1+n_aug_pts, 3)

    raw, alph_embed, rgb_embed = network_query_fn(aug_pts, viewdirs, network_fn)
    # raw: (N_rays, N_samples+1+n_aug_pts, 4) rgb+alpha
    # alph_embed: (N_rays, N_samples+1+n_aug_pts, embed_dim)
    # rgb_embed: (N_rays, N_samples+1+n_aug_pts, embed_dim/2)

    coarse_alph_embed_sur = alph_embed[:, N_samples:, :] # (N_rays, 1+n_aug_pts, embed_dim)
    raw_sur = raw[:, N_samples:, :] # (N_rays, 1+n_aug_pts, 4)
    raw = raw[:, :N_samples, :] # (N_rays, N_samples, 4)
    z_vals_no_sur = z_vals

    ret = {}

    rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(
                raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

    if N_importance > 0:
        rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map
        z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        z_samples = sample_pdf(z_vals_mid, weights[...,1:-1], N_importance, det=(perturb==0.), pytest=pytest)
        z_samples = z_samples.detach()
        z_vals, _ = torch.sort(torch.cat([z_vals_no_sur, z_samples], -1), -1)
        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # (N_rays, N_samples + N_importance, 3)

        run_fn = network_fn if network_fine is None else network_fine

        aug_pts = torch.cat([pts, tmp_pts], dim=1) # (N_rays, N_samples+N_importance+1+n_sur_pts, 3)
        raw, alph_embed, rgb_embed = network_query_fn(aug_pts, viewdirs, run_fn)
        raw_sur = raw[:, (N_samples+N_importance):, :]
        raw = raw[:, :(N_samples+N_importance), :]

        if self_attention_fn is not None: 
            # during testing, self_attention_fn is None
            alph_embed_sur = alph_embed[:, (N_samples+N_importance):, :]
            rgb_embed_sur = rgb_embed[:, (N_samples+N_importance):, :]

            alph_self_attention_fine, alph_proj_fine, \
                rgb_self_attention_fine, rgb_proj_fine, surround_embed = self_attention_fn(
                    tmp_pts, F.relu(raw_sur[:, :, [-1]]), alph_embed_sur, 
                    F.sigmoid(raw_sur[:, :, :3]), rgb_embed_sur, viewdirs, 
                    coarse_alph_embed_sur)

            # ret["alph_embed_fine"] = alph_embed_sur
            # ret["alph_atten_fine"] = alph_self_attention_fine
            # ret["alph_projection_fine"] = alph_proj_fine
            # ret["rgb_embed_fine"] = rgb_embed_sur
            # ret["rgb_atten_fine"] = rgb_self_attention_fine
            # ret["rgb_projection_fine"] = rgb_proj_fine
            ret["surround_embed"] = surround_embed # (N_rays, embed_dim)

            insert_pts = {
                "rgb_ref": rgb_self_attention_fine[:, [0]], 
                "rgb_nxt_ray": rgb_self_attention_fine[:, -n_nxt_ray_pts:], 
                "alph_ref": alph_self_attention_fine[:, [0]], 
                "alph_nxt_ray": alph_self_attention_fine[:, -n_nxt_ray_pts:], 
                "ts_ref": ref_ts, 
                "ts_nxt_ray": nxt_ray_ts, 
                "vox_end": vox_end}
            
            raw, z_vals, misc = insert_pt_to_ray(raw, z_vals, insert_pts)
            ret["transformer_out_alpha"] = misc["transformer_output_alpha"]
            # (num_rays, 1+n_nxt_ray_pts, 1)
            
        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(
            raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

    ret.update({'rgb_map' : rgb_map, 'disp_map' : disp_map, 'acc_map' : acc_map})
    if retraw:
        ret['raw'] = raw
        ret['pts'] = pts
    if N_importance > 0:
        ret['rgb0'] = rgb_map_0
        ret['disp0'] = disp_map_0
        ret['acc0'] = acc_map_0
        ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]
    for k in ret:
        if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
            print(f"! [Numerical Error] {k} contains nan or inf.")

    return ret


def config_parser():

    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, 
                        help='config file path')
    parser.add_argument("--expname", type=str, 
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/', 
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='./data/llff/fern', 
                        help='input data directory')

    # training options
    parser.add_argument("--netdepth", type=int, default=8, 
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256, 
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=8, 
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256, 
                        help='channels per layer in fine network')
    parser.add_argument("--N_rand", type=int, default=32*32*4, 
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float, default=5e-4, 
                        help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=250, 
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--chunk", type=int, default=1024*32, 
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024*64, 
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_batching", action='store_true', 
                        help='only take random rays from 1 image at a time')
    parser.add_argument("--no_reload", action='store_true', 
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None, 
                        help='specific weights npy file to reload for coarse network')
    parser.add_argument("--testchunk", type=int, default=8192, 
                        help='number of rays processed in parallel, decrease if running out of memory')

    # rendering options
    parser.add_argument("--N_samples", type=int, default=64, 
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true', 
                        help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=0, 
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10, 
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4, 
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0., 
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')

    parser.add_argument("--render_only", action='store_true', 
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true', 
                        help='render the test set instead of render_poses path')
    parser.add_argument("--render_factor", type=int, default=0, 
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

    parser.add_argument("--test_only", action='store_true', 
                        help='do not optimize, reload weights and test the performance')

    # sparse-view setting
    parser.add_argument("--sp_n_train", type=int, default=3,
                        help='number of training images in sparse-view setting')

    # voxel options
    parser.add_argument("--voxel_num", type=int, default=64,
                        help='number of voxels the scene is divided')

    # surrounding points options
    parser.add_argument("--n_sur_pts", type=int, default=9,
                        help='number of surrounding points for a point in the voxel')
    parser.add_argument("--n_nxt_ray_pts", type=int, default=9,
                        help='number of points on a ray')
    parser.add_argument("--radius_div", type=int, default=4,
                        help='radius of sphere sampling: voxel_size/radius_div')
    
    # transformer options
    parser.add_argument("--n_heads", type=int, default=1,
                        help='number of heads in transformer')
    parser.add_argument("--dim_ff", type=int, default=256,
                        help='feed forward dimension in transformer')
    parser.add_argument("--n_encoder_layers", type=int, default=2,
                        help='number of layers in transformer encoder')
    parser.add_argument("--n_decoder_layers", type=int, default=2,
                        help='number of layers in transformer decoder')
    parser.add_argument("--encoder_input_method", type=str, default="concat",
                        help='how to combine the feature input of encoder')

    # transformer encoder
    # parser.add_argument("--encoder_input_concat_coarse", action="store_true", 
    #                     help='concat the alpha prediction of coarse mlp in encoder input')
    # # transformer decoder
    # parser.add_argument("--decoder_learnable_param_dim", type=int, default=32,
    #                     help='dimension of learnable parameters in decoder')
    # parser.add_argument("--decoder_input_mlp", type=str, default="coarse",
    #                     help='input to the mlp decoder, coarse or fine')
    # parser.add_argument("--decoder_input_mlp_detach", action="store_true", 
    #                     help='detach the decoder input feature from mlp prediction')
    # parser.add_argument("--decoder_input_xyzview_pe", action="store_true", 
    #                     help='detach the decoder input feature from mlp prediction')
    # parser.add_argument("--decoder_input_not_concat_coarse", action="store_true", 
    #                     help='detach the decoder input feature from mlp prediction')

    # barf
    parser.add_argument("--transformer_use_barf_pe", action="store_true", 
                        help='use barf pe in transformer')
    parser.add_argument("--not_include_pi_in_posenc", action="store_true", 
                        help='use pi in position encoding')
    parser.add_argument("--barf_pe_coarse", type=float, default=0.1,
                        help='setting of barf pe')
    parser.add_argument("--barf_pe_fine", type=float, default=0.5,
                        help='setting of barf pe')
    parser.add_argument("--use_barf_pe", action="store_true", 
                        help='use barf pe')
    
    parser.add_argument("--voxel_sampler_rays_thsh", type=int, default=8, 
                        help='only sample the voxels over the threshold')
    parser.add_argument("--weighted_voxel_sampler", action="store_true", 
                        help='weighted sampling')

    # training options
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops') 
    parser.add_argument("--N_iters", type=int, default=200000, help="number of iterations") 
    parser.add_argument("--num_workers", type=int,
                        default=4, help='number of workers') 
    parser.add_argument("--anneal", action="store_true", help="anneal scheme from RegNeRF")

    # voxel-based ray sampler options
    parser.add_argument("--sample_n_voxels", type=int, default=256, 
                        help="number of voxels sampled during training")
    parser.add_argument("--n_rays_per_voxel", type=int, default=4, 
                        help="number of rays sampled in per voxel")

    # contrastive loss options
    parser.add_argument("--use_triplet_loss", action="store_true", 
                        help="whether to use triplet loss on the batch")
    parser.add_argument("--margin", type=float, default=1e-1, 
                        help="margin of the triplet loss")
    parser.add_argument("--w_triplet_loss", type=float, default=1e-1, 
                        help="weights of the triplet loss")
    parser.add_argument("--use_contrastive_loss", action="store_true", 
                        help="whether to use contrastive loss on the batch")
    parser.add_argument("--contrastive_temperature", type=float, default=1.0, 
                        help="temperature of the contrastive loss")
    parser.add_argument("--w_contrastive_loss", type=float, default=1e-1, 
                        help="weights of the contrastive loss")
    parser.add_argument("--contrastive_config", type=str, default="random_positive", 
                        help="different types of config for contrastive loss")       

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='llff', 
                        help='options: llff / blender / deepvoxels')
    parser.add_argument("--testskip", type=int, default=8, 
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')

    ## blender flags
    parser.add_argument("--white_bkgd", action='store_true', 
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--half_res", action='store_true', 
                        help='load blender synthetic data at 400x400 instead of 800x800')

    ## llff flags
    parser.add_argument("--factor", type=int, default=8, 
                        help='downsample factor for LLFF images')
    parser.add_argument("--no_ndc", action='store_true', 
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--lindisp", action='store_true', 
                        help='sampling linearly in disparity rather than depth')
    parser.add_argument("--spherify", action='store_true', 
                        help='set for spherical 360 scenes')
    parser.add_argument("--llffhold", type=int, default=8, 
                        help='will take every 1/N images as LLFF test set, paper uses 8')

    ## dtu flags
    parser.add_argument("--dtu_split_type", type=str, default="pixelnerf", 
                        help='different split types for dtu dataset')
    parser.add_argument("--dtu_maskdir", type=str, default=None, 
                        help='directory for mask')
    parser.add_argument("--dtu_reader", type=str, default="sparf", 
                        help='read dtu dataset using reader from sparf, otherwise default')
    
    # render the point cloud for visualization
    parser.add_argument("--render_point_cloud_vis", action="store_true", 
                        help='visualize the radiance field in the form of point cloud during rendering')
    parser.add_argument("--test_point_cloud_vis", action="store_true", 
                        help='visualize the radiance field in the form of point cloud during testing')
    parser.add_argument("--points_sample_skip", type=int, default=1, 
                        help='skip the sample when saving the point cloud')

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=100, 
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img",     type=int, default=500, 
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=10000, 
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=50000, 
                        help='frequency of testset saving')
    parser.add_argument("--i_video",   type=int, default=50000, 
                        help='frequency of render_poses video saving')

    # random seed
    parser.add_argument("--random_seed",   type=int, default=20230114, 
                        help='random seed')

    return parser


def train():
    parser = config_parser()
    args = parser.parse_args()
    set_seed(args.random_seed)

    K = None
    poses_w2c = None
    intrinsics = None

    # Load data
    if args.dataset_type == 'blender':
        dataset = BlenderDatasetDefault(
            args.datadir, args.datadir.split("/")[-1], 
            args.sp_n_train, args.half_res, args.testskip, 
            args.white_bkgd, bound=[2., 6.], voxel_num=args.voxel_num, 
            radius_div=args.radius_div, chunk=64*1024, 
            n_sur_pts=args.n_sur_pts, n_nxt_ray_pts=args.n_nxt_ray_pts, 
            train_test_split_file="./configs/pairs.th", 
            eval=(args.test_only or args.render_only))
        images, poses, render_poses, hwf, i_split = dataset.get_loaded_info()
        i_train, i_val, i_test = i_split
    elif args.dataset_type == "dtu":
        DTUDataset = DTUDatasetSPARF if args.dtu_reader == "sparf" else DTUDatasetDefault
        # NOTE: for vanilla nerf, we only experimented with VoxelSampler0, 
        # but these two samplers are very similiar
        print("DTU reader: ", DTUDataset)
        dataset = DTUDataset(
            args.datadir, args.datadir.split("/")[-1], 
            args.sp_n_train, args.dtu_split_type, 
            no_test_rest=True,
            mask_path=args.dtu_maskdir, voxel_num=args.voxel_num, 
            radius_div=args.radius_div, chunk=64*1024, 
            n_sur_pts=args.n_sur_pts, n_nxt_ray_pts=args.n_nxt_ray_pts, 
            eval=(args.test_only or args.render_only))
        if args.dtu_reader == "sparf": 
            images, poses, poses_w2c, intrinsics, render_poses, hwf, test_masks, i_split = dataset.get_loaded_info()
        elif args.dtu_reader == "default": 
            images, test_masks, poses, render_poses, hwf, i_split = dataset.get_loaded_info()
        else: 
            raise raiseNotImplementedError("Only sparf/default readers are supported for dtu currently.")
        i_train, i_val, i_test = i_split
    else:
        raise raiseNotImplementedError("Only dtu/blender datasets are supported currently.")

    near, far = dataset.near, dataset.far
    H, W, focal = hwf
    if K is None:
        K = np.array([
            [focal, 0, 0.5*W],
            [0, focal, 0.5*H],
            [0, 0, 1]])
    
    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    # Create nerf model
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args)
    global_step = start
    bds_dict = {
        'near' : near,
        'far' : far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    # Move testing data to GPU
    render_poses = torch.Tensor(render_poses).to(device)
    if poses_w2c is not None: 
        poses_w2c = torch.Tensor(poses_w2c).to(device)
    if intrinsics is not None: 
        intrinsics = torch.Tensor(intrinsics).to(device)
    
    # Move training data to GPU
    images = torch.Tensor(images).to(device)
    poses = torch.Tensor(poses).to(device)

    # Short circuit if only rendering out from trained model
    if args.render_only:
        print('RENDER ONLY')
        rendersavedir = os.path.join(basedir, expname, 'renderonly_{:06d}'.format(start+1))
        os.makedirs(rendersavedir, exist_ok=True)
        render_poses = render_poses[:5]

        if poses_w2c is not None: 
            video_render_poses = render_poses
            video_render_intr = intrinsics[0] # NOTE: here use the single intr
        else: 
            video_render_poses = None
            video_render_intr = None
        with torch.no_grad():
            rgbs, disps = render_path(render_poses, hwf, K, args.testchunk, render_kwargs_test, 
                                    savedir=rendersavedir, 
                                    w2c=video_render_poses, intr=video_render_intr, 
                                    point_cloud_vis=args.render_point_cloud_vis, points_sample_skip=args.points_sample_skip)
        print('Done, saving', rgbs.shape, disps.shape)
        moviebase = os.path.join(basedir, expname, '{}_spiral_{:06d}_'.format(expname, start+1))
        imageio.mimwrite(moviebase + 'rgb.mp4', to8b(rgbs), fps=24, quality=8)
        imageio.mimwrite(moviebase + 'disp.mp4', to8b(disps / np.max(disps)), fps=24, quality=8)
        
        return
    
    if args.test_only: 
        print('TEST ONLY')
        testsavedir = os.path.join(basedir, expname, 'testsetonly_{:06d}'.format(start+1))
        os.makedirs(testsavedir, exist_ok=True)

        if poses_w2c is not None: 
            test_render_poses = poses_w2c[i_test]
            test_render_intr = intrinsics[i_test]
        else: 
            test_render_poses = None
            test_render_intr = None

        with torch.no_grad():
            rgbs, _ = render_path(torch.Tensor(poses[i_test]).to(device), hwf, K, args.testchunk, render_kwargs_test, 
                                w2c=test_render_poses, intr=test_render_intr, 
                                gt_imgs=images[i_test], savedir=testsavedir, 
                                point_cloud_vis=args.test_point_cloud_vis, points_sample_skip=args.points_sample_skip)
        
        rgbs_cpu = torch.Tensor(rgbs).cpu()
        images_cpu = torch.Tensor(images[i_test]).cpu()
        rgbs_cpu = torch.clamp(rgbs_cpu, 0, 1)
        test_psnr = psnr_fn(images_cpu, rgbs_cpu)
        test_ssim = ssim_fn(images_cpu, rgbs_cpu)
        test_lpips = lpips_fn(images_cpu, rgbs_cpu, "vgg" if args.dataset_type=="blender" else "alex")

        print_str = "[Evaluation]  Iter: {}  PSNR: {}  SSIM: {}  LPIPS: {}  \n".format(start+1, 
            test_psnr, test_ssim, test_lpips)
        
        save_dict = {"psnr": test_psnr, "ssim": test_ssim, "lpips": test_lpips}
        
        append_str = "" 
        if args.dataset_type == "dtu":
            rgbs_fg = rgbs_cpu * test_masks + (1 - test_masks)
            images_fg = images_cpu * test_masks + (1 - test_masks)
            test_psnr_mask = psnr_mask_fn(rgbs_cpu, images_cpu, test_masks)
            test_ssim_mask = ssim_fn(images_fg, rgbs_fg)
            test_lpips_mask = lpips_fn(images_fg, rgbs_fg, "alex")

            append_str = "[DTU evaluation with mask]  PSNR: {}  SSIM: {}  LPIPS: {}  ".format(
                test_psnr_mask, test_ssim_mask, test_lpips_mask)
            
            save_dict.update({"psnr_mask": test_psnr_mask, "ssim_mask": test_ssim_mask, "lpips_mask": test_lpips_mask})

        print_str = print_str + append_str
        print(print_str)

        save_dict_json = {k: float(v) for k, v in save_dict.items()}
        with open(os.path.join(testsavedir, "results_iter{}.json".format(i)), "w") as fp: 
            json.dump(save_dict_json, fp)
        
        return
    
    # voxel-based ray sampler
    VoxelSampler = VoxelSamplerDefault if args.use_barf_pe else VoxelSampler0
    print("Voxel sampler: ", VoxelSampler)
    sampler = VoxelSampler(
            dataset.voxel_flag, 
            dataset.in_voxel_sum, 
            voxel_num=args.voxel_num, 
            batch_size=args.N_rand, 
            sample_n_voxels=args.sample_n_voxels, 
            n_rays_per_voxel=args.n_rays_per_voxel, 
            precrop_iters=args.precrop_iters, 
            n_rays_thsh=args.voxel_sampler_rays_thsh, 
            weighted_sampling=args.weighted_voxel_sampler)
    torch.multiprocessing.set_start_method("spawn")
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_sampler=sampler, num_workers=args.num_workers, pin_memory=True)

    # Define contastive or triplet loss function
    if args.use_triplet_loss and args.use_contrastive_loss: 
        raise ValueError("Should not train with both triplet loss and contrastive loss.")
    if args.use_triplet_loss:
        tripletlossfn = TripletLoss(args.margin, normalize_feature=False)
        batch_pseudo_label = torch.arange(args.sample_n_voxels)[:, None] # (sample_n_voxels, 1)
        batch_pseudo_label = batch_pseudo_label.expand(args.sample_n_voxels, args.n_rays_per_voxel)
        batch_pseudo_label = batch_pseudo_label.contiguous().view(-1) # (batch_size, )
    if args.use_contrastive_loss: 
        ctrlossfn = ContrastiveLoss(args.contrastive_temperature, args.n_rays_per_voxel, args.contrastive_config)
        batch_pseudo_label = torch.arange(args.sample_n_voxels)[:, None] # (sample_n_voxels, 1)
        batch_pseudo_label = batch_pseudo_label.expand(args.sample_n_voxels, args.n_rays_per_voxel)
        batch_pseudo_label = batch_pseudo_label.contiguous().view(-1) # (batch_size, )
    triplet_loss = torch.Tensor([0.]).to(device)
    contrastive_loss = torch.Tensor([0.]).to(device)

    N_rand = args.N_rand
    N_iters = args.N_iters + 1
    print('Begin')
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    print('VAL views are', i_val)  
    print("Sparse view training of {} views: {}".format(
            args.sp_n_train, i_train[:args.sp_n_train]))
    
    start = start + 1
    iterator = iter(data_loader)
    for i in trange(start, N_iters):
        render_kwargs_train["embedder"].progress.data.fill_(i/N_iters)
        render_kwargs_train["embedder_views"].progress.data.fill_(i/N_iters)
        # if args.transformer_use_barf_pe and args.decoder_input_xyzview_pe: 
        if args.transformer_use_barf_pe: 
            render_kwargs_train["self_attention_network"].xyz_embedder.progress.data.fill_(i/N_iters)
            render_kwargs_train["self_attention_network"].view_embedder.progress.data.fill_(i/N_iters)

        data = next(iterator)
        time0 = time.time()
        # accept data from dataloader
        rays_o, rays_d, target_s, ref_pts, t_ref_pts, sur_pts, t_nxt_ray_pts, vox_end = data
        # sur_pts include surrounding points and ray points
        rays_o = rays_o.cuda(non_blocking=True)
        rays_d = rays_d.cuda(non_blocking=True)
        target_s = target_s.cuda(non_blocking=True)
        ref_pts = ref_pts.cuda(non_blocking=True) # (N, 3)
        t_ref_pts = t_ref_pts.cuda(non_blocking=True) # (N, 1)
        sur_pts = sur_pts.cuda(non_blocking=True) # (N, 3, n_sur_pts+n_nxt_ray_pts)
        t_nxt_ray_pts = t_nxt_ray_pts.cuda(non_blocking=True) # (N, n_nxt_ray_pts)
        vox_end = vox_end.cuda(non_blocking=True) # (N, 1)

        batch_rays = torch.stack([rays_o, rays_d], 0)
        batch_sur = torch.cat([ref_pts, t_ref_pts, 
                    sur_pts.view(sur_pts.shape[0], -1), 
                    t_nxt_ray_pts, vox_end], dim=1) 
                    # (N, 4+3*(n_sur_pts+n_nxt_ray_pts)+n_nxt_ray_pts+1)
    
        # anneal scheme from RegNeRF
        if args.anneal: 
            near_final, far_final = bds_dict["near"], bds_dict["far"]
            mid = 0.5 * (near_final + far_final)
            near_init = mid + 0.2 * (near_final - mid)
            far_init = mid + 0.2 * (far_final - mid)
            weight = min(i * 1.0/2000, 1.0)
            near_i = near_init + weight * (near_final - near_init)
            far_i = far_init + weight * (far_final - far_init)
            render_kwargs_train["near"] = near_i
            render_kwargs_train["far"] = far_i
        
        #####  Core optimization loop  #####
        rgb, disp, acc, extras = render(H, W, K, chunk=args.chunk, rays=batch_rays,
                                                verbose=i < 10, retraw=True, 
                                                surround=batch_sur, 
                                                **render_kwargs_train)

        optimizer.zero_grad()
        img_loss = img2mse(rgb, target_s)
        trans = extras['raw'][...,-1]
        loss = img_loss
        psnr = mse2psnr(img_loss)

        if args.use_triplet_loss: 
            surround_emb = extras["surround_embed"]
            triplet_loss, _ = tripletlossfn(surround_emb, batch_pseudo_label)
            loss = loss + args.w_triplet_loss*triplet_loss
        if args.use_contrastive_loss: 
            surround_emb = extras["surround_embed"]
            contrastive_loss = ctrlossfn(surround_emb, batch_pseudo_label, ref_pts)
            loss = loss + args.w_contrastive_loss*contrastive_loss
        
        img_loss0 = img2mse(extras['rgb0'], target_s)
        loss = loss + img_loss0
        psnr0 = mse2psnr(img_loss0)
        
        loss.backward()
        optimizer.step()

        ###   update learning rate   ###
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate
        ################################

        dt = time.time()-time0

        # Rest is logging
        if i%args.i_weights==0:
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
            torch.save({
                'global_step': global_step,
                'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict(),
                'self_attention_state_dict': render_kwargs_train['self_attention_network'].state_dict(), 
                'optimizer_state_dict': optimizer.state_dict(),
                'embedder': render_kwargs_train['embedder'].state_dict(), 
                'embedder_views': render_kwargs_train['embedder_views'].state_dict(), 
            }, path)
            print('Saved checkpoints at', path)

        if i%args.i_video==0:
            # Fast test
            # render_poses = render_poses[:1]
            rendersavedir = os.path.join(basedir, expname, 'render_{:06d}'.format(i))
            os.makedirs(rendersavedir, exist_ok=True)

            if poses_w2c is not None: 
                video_render_poses = render_poses
                video_render_intr = intrinsics[0] # NOTE: here use the single intr
            else: 
                video_render_poses = None
                video_render_intr = None
            with torch.no_grad():
                rgbs, disps = render_path(render_poses, hwf, K, args.testchunk, render_kwargs_test, 
                                        savedir=rendersavedir, 
                                        w2c=video_render_poses, intr=video_render_intr, 
                                        point_cloud_vis=args.render_point_cloud_vis, points_sample_skip=args.points_sample_skip)
            print('Done, saving', rgbs.shape, disps.shape)
            moviebase = os.path.join(basedir, expname, '{}_spiral_{:06d}_'.format(expname, i))
            imageio.mimwrite(moviebase + 'rgb.mp4', to8b(rgbs), fps=24, quality=8)
            imageio.mimwrite(moviebase + 'disp.mp4', to8b(disps / np.max(disps)), fps=24, quality=8)
        
        if i%args.i_testset==0:
            testsavedir = os.path.join(basedir, expname, 'testset_{:06d}'.format(i))
            os.makedirs(testsavedir, exist_ok=True)
            # Fast test
            # i_test = i_test[:1]
            
            if poses_w2c is not None: 
                test_render_poses = poses_w2c[i_test]
                test_render_intr = intrinsics[i_test]
            else: 
                test_render_poses = None
                test_render_intr = None

            with torch.no_grad():
                rgbs, _ = render_path(torch.Tensor(poses[i_test]).to(device), hwf, K, args.testchunk, render_kwargs_test, 
                                    w2c=test_render_poses, intr=test_render_intr, 
                                    gt_imgs=images[i_test], savedir=testsavedir, 
                                    point_cloud_vis=args.test_point_cloud_vis, points_sample_skip=args.points_sample_skip)
            
            rgbs_cpu = torch.Tensor(rgbs).cpu()
            images_cpu = torch.Tensor(images[i_test]).cpu()
            rgbs_cpu = torch.clamp(rgbs_cpu, 0, 1)
            test_psnr = psnr_fn(images_cpu, rgbs_cpu)
            test_ssim = ssim_fn(images_cpu, rgbs_cpu)
            test_lpips = lpips_fn(images_cpu, rgbs_cpu, "vgg" if args.dataset_type=="blender" else "alex")
            # NOTE: for evaluation on blender dataset, we apply vgg for lpips evaluation as FreeNeRF: 
            # https://github.com/Jiawei-Yang/FreeNeRF/blob/main/DietNeRF-pytorch/dietnerf/run_nerf_helpers.py#L30
            # while on dtu dataset, we follow SPARF using alexnet for lpips evaluation: 
            # https://github.com/google-research/sparf/blob/4dcf3e3fdd76d7a33e97998bec06be04928d201d/source/training/base.py#L46

            print_str = "[Evaluation]  Iter: {}  PSNR: {}  SSIM: {}  LPIPS: {}  \n".format(i, 
                test_psnr, test_ssim, test_lpips)
            
            save_dict = {"psnr": test_psnr, "ssim": test_ssim, "lpips": test_lpips}
            
            append_str = "" 
            if args.dataset_type == "dtu":
                rgbs_fg = rgbs_cpu * test_masks + (1 - test_masks)
                images_fg = images_cpu * test_masks + (1 - test_masks)
                test_psnr_mask = psnr_mask_fn(rgbs_cpu, images_cpu, test_masks)
                test_ssim_mask = ssim_fn(images_fg, rgbs_fg)
                test_lpips_mask = lpips_fn(images_fg, rgbs_fg, "alex")

                append_str = "[DTU evaluation with mask]  PSNR: {}  SSIM: {}  LPIPS: {}  ".format(
                    test_psnr_mask, test_ssim_mask, test_lpips_mask)
                
                save_dict.update({"psnr_mask": test_psnr_mask, "ssim_mask": test_ssim_mask, "lpips_mask": test_lpips_mask})

            print_str = print_str + append_str
            print(print_str)

            save_dict_json = {k: float(v) for k, v in save_dict.items()}
            with open(os.path.join(testsavedir, "results_iter{}.json".format(i)), "w") as fp: 
                json.dump(save_dict_json, fp)

        if i%args.i_print==0:
            if args.use_contrastive_loss: 
                tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()}  Loss_Contrastive: {contrastive_loss.item()}")
            else: 
                tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()}  Loss_Triplet: {triplet_loss.item()}")

        global_step += 1

if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    train()