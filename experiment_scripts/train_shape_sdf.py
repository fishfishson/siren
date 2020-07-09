# Enable import from parent package
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import dataio, utils, loss_functions, modules
import torch
from torch.utils.tensorboard import SummaryWriter
# from tqdm.autonotebook import tqdm
# import time
import numpy as np
import os
import shutil
from torch.utils.data import DataLoader
import logging
import configargparse

p = configargparse.ArgumentParser()
p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

p.add_argument('--logging_root', type=str, default='./logs', help='root for logging')
p.add_argument('--experiment_name', type=str, required=True,
               help='Name of subdirectory in logging_root where summaries and checkpoints will be saved.')

# General training options
p.add_argument('--batch_size', type=int, default=5)
p.add_argument('--model_lr', type=float, default=1e-4, help='learning rate. default=5e-5')
p.add_argument('--lat_lr', type=float, default=1e-3, help='learning rate. default=5e-5')
p.add_argument('--num_epochs', type=int, default=1000,
               help='Number of epochs to train for.')
p.add_argument('--lat_coeff', type=float, default=1.)

p.add_argument('--epochs_til_ckpt', type=int, default=1,
               help='Time interval in seconds until checkpoint is saved.')
p.add_argument('--steps_til_summary', type=int, default=20,
               help='Time interval in seconds until tensorboard summary is saved.')

p.add_argument('--model_type', type=str, default='sine',
               help='Options are "sine" (all sine activations) and "mixed" (first layer sine, other layers tanh)')
p.add_argument('--point_cloud_path', type=str, default='/home/sitzmann/data/point_cloud.xyz',
               help='Options are "sine" (all sine activations) and "mixed" (first layer sine, other layers tanh)')

p.add_argument('--checkpoint_path', default=None, help='Checkpoint to trained model.')

p.add_argument('--on_surface_points', type=int, default=5000)
p.add_argument('--off_surface_points', type=int, default=5000)
p.add_argument('--latent_dim', type=int, default=64)
p.add_argument('--batch_split', type=int, default=8)
p.add_argument('--clip_grad', type=float, default=1.)
p.add_argument('--shuffle_batch', type=bool, default=False)

opt = p.parse_args()

def main(opt):
    logging.info('running {}'.format(opt.experiment_name))

    sdf_dataset = dataio.Mesh(opt.point_cloud_path, on_surface_points=opt.on_surface_points,
                              off_surface_points=opt.off_surface_points)
    dataloader = DataLoader(sdf_dataset, shuffle=True, batch_size=opt.batch_size, pin_memory=True, num_workers=4)

    # Define the model.
    model = modules.SingleBVPNet(in_features=3 + opt.latent_dim, out_features=1, type='sine')
    model.cuda()
    # model = torch.nn.DataParallel(model)

    # Define the loss
    sdf_loss = loss_functions.sdf
    # summary_fn = utils.write_sdf_summary

    root_path = os.path.join(opt.logging_root, opt.experiment_name)

    if os.path.exists(root_path):
        val = input("The model directory %s exists. Overwrite? (y/n)" % root_path)
        if val == 'y':
            shutil.rmtree(root_path)

    os.makedirs(root_path)

    summaries_dir = os.path.join(root_path, 'summaries')
    utils.cond_mkdir(summaries_dir)

    checkpoints_dir = os.path.join(root_path, 'checkpoints')
    utils.cond_mkdir(checkpoints_dir)

    writer = SummaryWriter(summaries_dir)

    def save_all(epoch, filename):
        torch.save(
            {"epoch": epoch,
             "optim": optim.state_dict(),
             "model": model.state_dict(),
             "lat_vecs": lat_vecs.state_dict()},
            os.path.join(checkpoints_dir, filename)
        )

    num_data = len(sdf_dataset)
    lat_vecs = torch.nn.Embedding(num_data, opt.latent_dim)
    torch.nn.init.normal_(lat_vecs.weight.data, 0, np.sqrt(1 / 3))
    lat_vecs.cuda()

    optim = torch.optim.Adam([
        {
            "lr": opt.model_lr,
            "params": model.parameters()
        },
        {
            "lr": opt.lat_lr,
            "params": lat_vecs.parameters()
        }])

    start_epoch = 0
    total_steps = 0
    batch_split = opt.batch_split

    for epoch in range(start_epoch, opt.num_epochs):
        logging.info("epoch {}...".format(epoch))

        # adjust_learning_rate(lr_schedules, optim, epoch)

        if not epoch % opt.epochs_til_ckpt and epoch:
            save_all(epoch, 'all_epoch_%04d.pth' % epoch)

        epoch_loss = 0
        steps_per_epoch = 0

        for coords, sdfs, normals, indices in dataloader:
            # batch, sample, feats
            batch_size = coords.shape[0]
            num_sample = coords.shape[1]

            coords = coords.reshape(-1, 3).cuda()
            sdfs = sdfs.reshape(-1, 1).cuda()
            normals = normals.reshape(-1, 3).cuda()
            indices = indices.unsqueeze(-1).repeat(1, num_sample).view(-1).cuda()

            if opt.shuffle_batch:
                shuffle = torch.randperm(batch_size * num_sample)
                coords = coords[shuffle]
                sdfs = sdfs[shuffle]
                normals = normals[shuffle]
                indices = indices[shuffle]

            coords = torch.chunk(coords, batch_split)
            sdfs = torch.chunk(sdfs, batch_split)
            normals = torch.chunk(normals, batch_split)
            indices = torch.chunk(indices, batch_split)

            optim.zero_grad()

            for i in range(batch_split):
                batch_vecs = lat_vecs(indices[i])
                model_input = {'coords': coords[i], 'latent': batch_vecs}
                model_output = model(model_input)
                gt = {'sdf': sdfs[i], 'normals': normals[i]}

                batch_loss = 0

                sdf_losses = sdf_loss(model_output, gt)
                lat_loss = opt.lat_coeff * torch.mean(batch_vecs ** 2)

                batch_loss += lat_loss

                writer.add_scalar('lat_loss', lat_loss, total_steps)
                logging.info("{}: {}".format('latent', lat_loss.item()))

                for loss_name, loss in sdf_losses.items():
                    # single_loss = loss.mean()
                    writer.add_scalar(loss_name, loss, total_steps)
                    logging.info("{}: {}".format(loss_name, loss.item()))
                    batch_loss += loss

                writer.add_scalar("batch_loss", batch_loss, total_steps)
                logging.info("Step = {}. Batch Loss = {}".format(total_steps, batch_loss.item()))

                batch_loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=opt.clip_grad)

                optim.step()

                total_steps += 1
                steps_per_epoch += 1
                epoch_loss += batch_loss.item()

                if not total_steps % opt.steps_til_summary and total_steps:
                    save_all(epoch, 'all_current.pth')

        epoch_loss /= steps_per_epoch
        logging.info("Epoch = {}. Epoch Loss = {}".format(epoch, epoch_loss))

    save_all(epoch, 'all_final.pth')


if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    main(opt)
