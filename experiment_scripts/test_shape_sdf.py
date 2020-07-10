'''Test script for experiments in paper Sec. 4.2, Supplement Sec. 3, reconstruction from laplacian.
'''

# Enable import from parent package
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import modules
import utils
import sdf_meshing
import configargparse
import torch
from functools import partial

p = configargparse.ArgumentParser()
p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

p.add_argument('--logging_root', type=str, default='./logs', help='root for logging')
p.add_argument('--experiment_name', type=str, required=True,
               help='Name of subdirectory in logging_root where summaries and checkpoints will be saved.')

# General training options
p.add_argument('--model_type', type=str, default='sine',
               help='Options are "sine" (all sine activations) and "mixed" (first layer sine, other layers tanh)')
p.add_argument('--latent_dim', type=int, default=64)

opt = p.parse_args()


class SDFDecoder(torch.nn.Module):
    def __init__(self, opt):
        super().__init__()
        # Define the model.
        all_model = torch.load(os.path.join(opt.logging_root, opt.experiment_name, 'checkpoints/all_final.path'))
        self.model = modules.SingleBVPNet(type=opt.model_type, out_features=1, in_features=opt.latent_dim + 3)
        self.model.load_state_dict(all_model['model'])
        self.embed = torch.nn.Embedding(opt.num_data, opt.latent_dim)
        self.embed.load_state_dict(torch.load(all_model['lat_vecs']))
        self.embed.cuda()
        self.model.cuda()
        self.idx = None

    def load_idx(self, idx):
        self.idx = torch.tensor(idx).cuda()

    def forward(self, coords):
        num_points = coords.shape[0]
        lat_vec = self.embed(self.idx).unsqueeze(0).repeat(num_points, 1)
        model_in = {'coords': coords, 'latent': lat_vec}
        return self.model(model_in)['model_out']


def main(opt):
    for i in range(opt.num_data):
        sdf_decoder = SDFDecoder(opt)
        sdf_decoder.load_idx(i)
        root_path = os.path.join(opt.logging_root, opt.experiment_name, 'rec')
        utils.cond_mkdir(root_path)
        sdf_meshing.create_mesh(sdf_decoder, os.path.join(root_path, 'rec_{}'.format(i)), N=256)


if __name__ == '__main__':
    main(opt)
