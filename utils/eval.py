import torch
import numpy as np
from skimage.metrics import structural_similarity
import lpips

img2mse = lambda x, y : torch.mean((x - y) ** 2)
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
loss_fn_vgg = lpips.LPIPS(net='vgg')
loss_fn_alex = lpips.LPIPS(net='alex')

def ssim_fn(x, y):
  res = []
  for i in range(x.shape[0]):
    x_np, y_np = x[i].numpy(), y[i].numpy()
    res.append(
      structural_similarity(x_np, y_np, multichannel=True, data_range=x_np.max()-x_np.min())
    )
  return np.mean(res)

def lpips_fn(x, y, net="alex"):
  assert net in ["alex", "vgg"]
  fn = loss_fn_alex if net=="alex" else loss_fn_vgg
  gt_imgs = x
  rendered_imgs = y
  res = []
  n_imgs = x.shape[0]
  for i in range(n_imgs): 
    img_tensor = rendered_imgs[[i]]
    img_tensor = img_tensor.permute(0, 3, 1, 2).float()*2-1.0
    img_gt_tensor = gt_imgs[[i]]
    img_gt_tensor = img_gt_tensor.permute(0, 3, 1, 2).float()*2-1.0
    res.append(
      # loss_fn_vgg(img_tensor, img_gt_tensor).item()
      # loss_fn_alex(img_tensor, img_gt_tensor).item()
      fn(img_tensor, img_gt_tensor).item()
    )
  return np.mean(res)

def psnr_fn(x, y): 
  res = []
  for i in range(x.shape[0]): 
    mse_i = img2mse(x[i], y[i])
    psnr_i = mse2psnr(mse_i)
    res.append(
      psnr_i.cpu().numpy()
    )
  return np.mean(res)

def psnr_mask_fn(x, y, mask):
  res = []
  for i in range(x.shape[0]): 
    mask_i = mask[i]
    mse_i = ((x[i]-y[i])[mask_i==1.0]**2).mean()
    psnr_i = mse2psnr(mse_i)
    res.append(
      psnr_i.cpu().numpy()
    )
  return np.mean(res)
