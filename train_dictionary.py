import os
import sys
import numpy as np
np.set_printoptions(suppress=True, threshold=sys.maxsize)

import argparse
from tqdm import tqdm
from glob import glob
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from utils import encoder
from models.stylegan2.models import Generator


class SClipLoss(torch.nn.Module):
    def __init__(self, args, singleS2C=None):
        super(SClipLoss, self).__init__()
        if args.dataset == 'ffhq':
            stylegan_size = 1024
        elif args.dataset != 'church':
            stylegan_size = 512
        else:
            stylegan_size = 256

        generator = Generator(
            size = stylegan_size,
            style_dim = 512,
            n_mlp = 8,
            channel_multiplier = 2,
        )
        generator.load_state_dict(torch.load(os.path.join(args.path, f"stylegan2/{args.dataset}.pt"), map_location='cpu')['g_ema'])
        generator.eval()
        self.generator= generator.to(args.device)
        self.latent = torch.load(args.latents_path, map_location='cpu').to(args.device)
        self.args = args
        
        self.style_space, self.style_names, self.noise_constants = encoder(
            G = self.generator,
            latent = self.latent
            )
        self.loss = nn.MSELoss()
        self.singleS2C = singleS2C
        

    def forward(self, delta, x, multis2c):

        """_summary_
        Args:
            delta (torch.Tensor): (B, 9088)
            x (torch.Tensor): (6048, 512)
        """
        delta_ = self.removeTorgb(delta, x) # (B, 6048)
        x = self.removeTorgb(x, x)
        recon_loss = self.loss(torch.matmul(delta_, x), multis2c)
        alpha_ = multis2c @ x.T
        alpha_ = alpha_ / (alpha_+1e-6).abs().max(dim=-1, keepdim=True)[0]
        delta_ = delta_ / delta_.abs().max(dim=-1, keepdim=True)[0]
        alpha_loss = self.loss(delta_, alpha_)
        
        return recon_loss, alpha_loss
    
    def removeTorgb(self, tensor, fs3):
        """
        Convert a tensor of shape B, 9088
        into a tensor of shape B, 6048        
        """
        new_tensor = []
        t_index, s_index = 0, 0
        for name, style in zip(self.style_names, self.style_space):
            num_channel = style.shape[-1]
            if fs3 is not None:
                total = fs3.shape[0]
                if ("torgb" not in name) and t_index<total:
                    new_tensor.append(tensor[:, s_index: s_index + num_channel])
                    t_index += num_channel
            s_index += num_channel
        new_tensor = torch.cat(new_tensor, dim=-1).to(args.device)
        return new_tensor




    
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device',         type=str,   default='cuda:0')
    parser.add_argument('--path',           type=str,   default='./Pretrained')
    parser.add_argument('--dataset',        type=str,   default='ffhq')
    parser.add_argument('--epoch',          type=int,   default=5000)
    parser.add_argument('--alpha_lmbd',     type=float, default=0.1)
    parser.add_argument('--optim',          type=str,   default='adadelta')
    parser.add_argument('--lr',             type=float, default=3.0)
    parser.add_argument('--alpha',          type=int,   default=15)
    parser.add_argument('--ours',           action='store_true')

    args = parser.parse_args()
    args.latents_path = f'dictionary/{args.dataset}/latent.pt'
    
    multiS2C = []
    deltas = []
    channels = []
    
    clip_dirs = glob(f'precomputed_pairs/{args.dataset}/clip/alpha{args.alpha}ch*.npy')
    for cd in clip_dirs:
        num_c = cd.split('/')[-1].split('ch')[-1][:-4]
        if int(num_c) in [10, 30, 50]:
            channels.append(num_c)
        else:
            continue
        t = torch.Tensor(np.load(cd))
        multiS2C.append(t)
        d = torch.load(f'precomputed_pairs/{args.dataset}/unsup_dir/ch{num_c}.pt')
        deltas.append(d)
        assert t.shape[0] == d.shape[0], f'{num_c}: clip {t.shape} stylespace {d.shape}'
    multiS2C = torch.cat(multiS2C).to(args.device)
    deltas = torch.cat(deltas).to(args.device)

    y = torch.Tensor(np.load(f'dictionary/{args.dataset}/fs3.npy').astype(np.float32)).to(args.device) # Dictionary from Global Direction
    x = torch.zeros(y.shape).to(args.device) # Same shape
    x.requires_grad_(True)
    loss_model = SClipLoss(args, singleS2C=None)
    optimizer = optim.Adadelta([x], lr=args.lr)
    prog_bar = tqdm(range(args.epoch))
    best_loss = np.inf
    
    for epoch in prog_bar:
        recon_loss, alpha_loss = loss_model(args.alpha * deltas, x, multiS2C)
        loss = recon_loss + args.alpha_lmbd * alpha_loss
        prog_bar.set_description(desc=f"recon: {recon_loss:.3e} alpha: {alpha_loss:.3e}")
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        if (epoch)%1000==0 and best_loss>loss.item():
            best_loss = loss.item()
            file_name = f'dictionary/{args.dataset}/multi2one_new.pt'
            torch.save(x.detach().cpu(), file_name)
    print(f"saved new dictionary at {file_name}")
        