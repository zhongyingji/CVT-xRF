from plyfile import PlyData, PlyElement
import os
import argparse
import random
import numpy as np
from pathlib import Path

parser = argparse.ArgumentParser(description='Merge points.')
parser.add_argument('--data_path', type=str, 
                    help='root data path')
parser.add_argument("--data_type", type=str, default="fine", 
                    help='''coarse or fine, only support coarse''')
parser.add_argument("-r", "--sample_ratio", type=float, default=1.0, 
                    help='''topk for expert''')

args = parser.parse_args()

data_path = args.data_path
data_type = args.data_type
sample_ratio = args.sample_ratio

data_path_1 = Path(data_path)
plys = [i.name for i in data_path_1.glob('**/*') if i.suffix == ".ply"]
image_ids = [i.split("_")[0] for i in plys if i.split("_")[0].isdigit()]
image_ids = list(set(image_ids))

print("image_ids", image_ids)


# no moe or clusters
out_ply_name = '{}_pts_rgba.ply'.format(data_type)
out_ply_path = os.path.join(data_path, out_ply_name)
# plys.remove(out_ply_name)

sample_datas = []
for image_id in image_ids:
    ply_name = '{:03d}_pts_rgba.ply'.format(int(image_id))

    ply_path = os.path.join(data_path, ply_name)
    ply_data = PlyData.read(ply_path)
    pts_data = ply_data.elements[0].data

    pts_num = ply_data.elements[0].count
    sample_num = int(pts_num * sample_ratio)
    if sample_num == 0:
        continue
    else:
        sample_ids = random.sample(range(pts_num), sample_num)
        sample_data = pts_data[sample_ids]
        sample_datas.append(sample_data)

    print("Done with {}.".format(ply_name))

print("Saving all...")
sample_data = np.concatenate(sample_datas)
el = PlyElement.describe(sample_data, 'vertex')
PlyData([el]).write(out_ply_path)
print("Save done. ")
pass



