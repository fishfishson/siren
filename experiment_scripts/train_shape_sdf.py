# Enable import from parent package
import sys
import os
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )
import dataio, meta_modules, utils, loss_functions, modules
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm
import time
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
p.add_argument('--lr', type=float, default=1e-4, help='learning rate. default=5e-5')
p.add_argument('--num_epochs', type=int, default=1000,
               help='Number of epochs to train for.')
p.add_argument('--reg_coeff', type=float, default=1)

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
p.add_argument('--clip_grad', type=bool, default=True)



opt = p.parse_args()

class StepLearningRateSchedule():
    def __init__(self, initial, interval, factor):
        self.initial = initial
        self.interval = interval
        self.factor = factor

    def get_learning_rate(self, epoch):

        return self.initial * (self.factor ** (epoch // self.interval))

def main(opt):
    def adjust_learning_rate(lr_schedules, optimizer, epoch):

        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedules[i].get_learning_rate(epoch)

    sdf_dataset = dataio.Mesh(opt.point_cloud_path, on_surface_points=opt.on_surface_points, off_surface_points=opt.off_surface_points)
    dataloader = DataLoader(sdf_dataset, shuffle=True, batch_size=opt.batch_size, pin_memory=True, num_workers=4)

    # Define the model.
    model = modules.SingleBVPNet(in_features=3 + opt.latent_dim, out_features=1, type='sine')
    model.cuda()
    # model = torch.nn.DataParallel(model)

    # Define the loss
    loss_fn = loss_functions.sdf
    summary_fn = utils.write_sdf_summary

    root_path = os.path.join(opt.logging_root, opt.experiment_name)

    if os.path.exists(root_path):
        val = input("The model directory %s exists. Overwrite? (y/n)"%root_path)
        if val == 'y':
            shutil.rmtree(root_path)

    os.makedirs(root_path)

    summaries_dir = os.path.join(root_path, 'summaries')
    utils.cond_mkdir(summaries_dir)

    checkpoints_dir = os.path.join(root_path, 'checkpoints')
    utils.cond_mkdir(checkpoints_dir)

    writer = SummaryWriter(summaries_dir)

    num_mesh = len(sdf_dataset)
    lat_vecs = torch.nn.Embedding(num_mesh, opt.latent_dim, max_norm=1)
    torch.nn.init.normal_(lat_vecs.weight.data, 0, np.sqrt(1 / 3))

    lr_schedules = [StepLearningRateSchedule(1e-4, 500, 0.5), StepLearningRateSchedule(1e-3, 500, 0.5)]

    optim = torch.optim.Adam([{
        "lr": lr_schedules[0].get_learning_rate(0), 
        "params": model.parameters()                       
    },
    {
        "lr": lr_schedules[1].get_learning_rate(0), 
        "params": lat_vecs.parameters() 
    }])

    start_epoch = 0
    total_steps = 0
    batch_split = opt.batch_split
    train_losses = []

    for epoch in range(start_epoch, opt.num_epochs):
        model.train()

        logging.info("epoch {}...".format(epoch))

        adjust_learning_rate(lr_schedules, optim, epoch)

        if not epoch % opt.epochs_til_ckpt and epoch:
            torch.save(model.state_dict(),
                        os.path.join(checkpoints_dir, 'model_epoch_%04d.pth' % epoch))
            torch.save(lat_vecs.state_dict(),
                        os.path.join(checkpoints_dir, 'lat_epoch_%04d.pth' % epoch))
            np.savetxt(os.path.join(checkpoints_dir, 'train_losses_epoch_%04d.txt' % epoch),
                        np.array(train_losses))
        
        epoch_loss = 0
        steps_per_epoch = 0

        for coords, sdfs, normals, indices in dataloader:
            num_sample = coords.shape[1]
            coords = coords.reshape(-1, 3).cuda()
            sdfs = sdfs.reshape(-1, 1).cuda()
            normals = normals.reshape(-1, 3).cuda()

            coords = torch.chunk(coords, batch_split)
            sdfs = torch.chunk(sdfs, batch_split)
            normals = torch.chunk(normals, batch_split)
            indices = torch.chunk(
                indices.unsqueeze(-1).repeat(1, num_sample).view(-1),
                batch_split,
            )
                        
            optim.zero_grad()

            for i in range(batch_split):
                batch_vecs = lat_vecs(indices[i]).cuda()
                # inputs = torch.cat([coords[i], batch_vecs], dim=1)
                model_input = {'coords': coords[i], 'latent': batch_vecs}
                model_output = model(model_input)
                gt = {'sdf': sdfs[i], 'normals': normals[i]}
                
                losses = loss_fn(model_output, gt)
                chunk_loss = opt.reg_coeff * torch.mean(torch.norm(batch_vecs, dim=1) ** 2)
                writer.add_scalar('lat_loss', chunk_loss, total_steps)
                logging.info("{}: {}".format('latent', chunk_loss.item()))
                for loss_name, loss in losses.items():
                    single_loss = loss.mean()
                    writer.add_scalar(loss_name, single_loss, total_steps)
                    logging.info("{}: {}".format(loss_name, single_loss.item()))
                    chunk_loss += single_loss
                writer.add_scalar("total_chunk_loss", chunk_loss, total_steps)
                logging.info("step = {}. total chunk loss = {}".format(total_steps, chunk_loss.item()))
                chunk_loss.backward()

                if opt.clip_grad:
                    if isinstance(opt.clip_grad, bool):
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=opt.clip_grad)

                optim.step()

                total_steps += 1
                steps_per_epoch += 1
                epoch_loss += chunk_loss.item()
                
                if not total_steps % opt.steps_til_summary and total_steps:
                     torch.save(model.state_dict(), os.path.join(checkpoints_dir, 'model_current.pth'))
                     torch.save(lat_vecs.state_dict(), os.path.join(checkpoints_dir, 'lat_current.pth'))
        
        epoch_loss /= steps_per_epoch
        logging.info("epoch = {}. ave epoch loss = {}".format(epoch, epoch_loss))
        train_losses.append(epoch_loss)

    torch.save(model.state_dict(),
                os.path.join(checkpoints_dir, 'model_final.pth'))
    torch.save(lat_vecs.state_dict(),
                os.path.join(checkpoints_dir, 'lat_final.pth'))                
    np.savetxt(os.path.join(checkpoints_dir, 'train_losses_final.txt'),
                np.array(train_losses))

if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    main(opt)
