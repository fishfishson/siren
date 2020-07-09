'''Test script for experiments in paper Sec. 4.2, Supplement Sec. 3, reconstruction from laplacian.
'''

# Enable import from parent package
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import modules, utils
import sdf_meshing
import configargparse
import torch

p = configargparse.ArgumentParser()
p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

p.add_argument('--logging_root', type=str, default='./logs', help='root for logging')
p.add_argument('--experiment_name', type=str, required=True,
               help='Name of subdirectory in logging_root where summaries and checkpoints will be saved.')

# General training options
p.add_argument('--batch_size', type=int, default=16384)
p.add_argument('--model_ckpt_path', default=None, help='Checkpoint to trained model.')
p.add_argument('--latent_ckpt_path', default=None)
p.add_argument('--model_type', type=str, default='sine',
               help='Options are "sine" (all sine activations) and "mixed" (first layer sine, other layers tanh)')
p.add_argument('--mode', type=str, default='mlp',
               help='Options are "mlp" or "nerf"')
p.add_argument('--latent_dim', type=int, default=64)
p.add_argument('--idx', type=int, default=0)

opt = p.parse_args()


class SDFDecoder(torch.nn.Module):
    def __init__(self, idx):
        super().__init__()
        # Define the model.
        self.model = modules.SingleBVPNet(type=opt.model_type, final_layer_factor=1, in_features=opt.latent_dim + 3)
        self.model.load_state_dict(torch.load(opt.model_ckpt_path))
        self.embed = torch.nn.Embedding(20, opt.latent_dim)
        self.embed.load_state_dict(torch.load(opt.latent_ckpt_path))
        self.embed.cuda()
        self.model.cuda()
        self.idx = torch.tensor(idx).cuda()

    def forward(self, coords):
        num_points = coords.shape[0]
        lat_vec = self.embed(self.idx).unsqueeze(0).repeat(num_points, 1)
        model_in = {'coords': coords, 'latent': lat_vec}
        return self.model(model_in)['model_out']


sdf_decoder = SDFDecoder(opt.idx)

root_path = os.path.join(opt.logging_root, opt.experiment_name)
utils.cond_mkdir(root_path)

sdf_meshing.create_mesh(sdf_decoder, os.path.join(root_path, 'test'), N=256)