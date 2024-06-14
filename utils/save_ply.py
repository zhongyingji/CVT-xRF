import torch
import torch.nn.functional as F
from plyfile import PlyData, PlyElement

def save_point_cloud_in_ply(data, H, W, save_dir, filename, points_sample_skip):
    pts = data["pts"].view(H, W, -1, 3) # [H, W, samples, 3]
    
    raw = data["raw"] # [num_rays, samples, 4]
    pts_rgb = torch.sigmoid(raw[..., :3]).view(H, W, -1, 3) # [H, W, samples, 3]
    pts_alpha = F.relu(raw[..., -1]).view(H, W, -1, 1) # [H, W, samples, 1]

    pts = pts[:, :, ::points_sample_skip]
    pts_rgb = pts_rgb[:, :, ::points_sample_skip]
    pts_alpha = pts_alpha[:, :, ::points_sample_skip]

    vertexs = pts.reshape(-1, 3).cpu().numpy()
    vertexs = np.array([tuple(v) for v in vertexs], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])

    pts_rgba = torch.cat([pts_rgb, pts_alpha], dim=-1) # [H, W, samples, 4]
    vertex_colors = (pts_rgba * 255).to(torch.uint8).reshape(-1, 4).cpu().numpy()
    vertex_colors = np.array([tuple(v) for v in vertex_colors], dtype=[('red', 'u1'), ('green', 'u1'), ('blue', 'u1'), ('alpha', 'u1')])

    plyfilename = "{}_pts_rgba.ply".format(filename)
    plyfilename = os.path.join(save_dir, plyfilename)
    
    vertex_all = np.empty(len(vertexs), vertexs.dtype.descr + vertex_colors.dtype.descr)
    for prop in vertexs.dtype.names:
        vertex_all[prop] = vertexs[prop]
    for prop in vertex_colors.dtype.names:
        vertex_all[prop] = vertex_colors[prop]

    el = PlyElement.describe(vertex_all, 'vertex')
    PlyData([el]).write(plyfilename)
    
    return