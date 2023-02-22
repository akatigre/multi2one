import os
import copy
from PIL import Image
from tqdm import trange

import clip
import torch
import pickle
import argparse
import numpy as np
from pathlib import Path
from models.stylegan2.models import Generator
from models.stylegan2.utils import encoder, decoder
from torchvision.utils import save_image, make_grid
from matplotlib import pyplot as plt
from torchvision.transforms import functional as F


def show(imgs, column_names, save_name, dpi=800, suptitle=None):
    
    fig, axs = plt.subplots(nrows=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[i,0].imshow(np.asarray(img))
        axs[i,0].set_ylabel(column_names[i], fontsize=5)
        axs[i,0].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    if suptitle is not None:
        fig.suptitle(suptitle, fontsize=8)
    plt.savefig(save_name, dpi=dpi)
    plt.cla()


def tensor2stylespace(tensor, style_names, style_space):
    delta_style_space = []
    t_index=0
    tensor = tensor.reshape([-1, tensor.shape[-1]])
    for i, layer_name in enumerate(style_names):
        num_channels = style_space[i].shape[-1]
        s = tensor[:, t_index:t_index+num_channels]
        t_index+=num_channels
        delta_style_space.append(s)
    return delta_style_space

def multiS2C(args, style_space, style_names):
    """
        Our method (Multi2One), encoding multiple channels so as to encode interactive affect into CLIP
    """
    steps = [-10, 10]
    num_steps = len(steps)
    deltas = torch.load(f'{path}/delta/interfacegan_without_torgb.pt').to(args.device)
    _d = torch.zeros_like(deltas)
    _, idxs = torch.abs(deltas).topk(k=args.num_channels, dim=-1)
    tmp = deltas.gather(dim=1, index=idxs)
    _d.scatter_(dim=1, index=idxs, src=tmp)
    deltas = _d
    assert len(deltas[0].nonzero()) == args.num_channels, f"Mismatch {len(deltas[0].nonzero())}"
    
    delta_style_space = tensor2stylespace(deltas, style_names, style_space) # remove ToRGB

    clip_encodings = []
    
    num_directions = len(deltas)
    print(num_directions)
    all_features = np.zeros((num_directions, num_images, 2, 512))
    for d_idx in trange(num_directions):
        out = np.zeros((num_images, num_steps, args.stylegan_size, args.stylegan_size, 3), dtype='uint8')
        for i in range(num_images):
            latent_w = latent[i:i+1,:,:]
            style_space, style_names, noise_constants = encoder(
                G =  generator,
                latent = latent_w
                )
            
            for j, step in enumerate(steps):
                img = decoder(
                    G=generator, 
                    style_space=[style + step * d_style[d_idx,:].to(args.device) for style, d_style in zip(style_space, delta_style_space)],
                    latent=latent_w,
                    noise=noise_constants,
                    dataset=args.dataset,
                )
                img = img.detach().cpu().numpy()
                img = np.transpose(img, [0, 2, 3, 1])
                drange = [-1, 1]
                scale = 255 / (drange[1] - drange[0])
                img = img * scale + (0.5 - drange[0] * scale)
                np.clip(img, 0, 255, out=img)
                img = img.astype('uint8')
                out[i, j, :, :, :] = img # num_imgs, step, 1024, 1024, 3
            
        # Iterate over uint8 images to encode into clip
        imgs = out.reshape([-1, *out.shape[2:]]) # num_imgs*step, 1024, 1024, 3
        tmp = []
        for img_idx in range(len(imgs)):
            img = Image.fromarray(imgs[img_idx])
            image = preprocess(img).unsqueeze(0).to(args.device)
            tmp.append(image)
        
        image = torch.cat(tmp)
        with torch.no_grad():
            feat = model.encode_image(image)
        feat = feat.cpu().numpy().reshape([*out.shape[:2], 512]) # num_images, num_step, 512
        fs1 = feat / np.linalg.norm(feat, axis=-1, keepdims=True)
        all_features[d_idx] = fs1
        fs2 = fs1[:,1,:] - fs1[:,0,:]
        clip_encodings.append(fs2)
    
    f = clip_encodings
    f = f/np.linalg.norm(f, axis=-1, keepdims=True)
    fs3 = f.mean(axis=1)
    fs3 = fs3/np.linalg.norm(fs3, axis=-1, keepdims=True)
    np.save(f'{path}/clip/interfacegan{steps[1]}_ch{args.num_channels}.npy',fs3)
    torch.save(deltas, f'{path}/delta/interfacegan{steps[1]}_ch{args.num_channels}.pt')


def singleS2C(args, style_space, style_names):
    """
        StyleCLIP Global Direction, encoding single channel of StyleSpace into CLIP to create a dictionary D
    """
    steps = [-10, 10]
    num_steps = len(steps)
    clip_encodings = []
    name = style_names[args.layer_idx]
    style = style_space[args.layer_idx]
    assert 'torgb' not in name, 'No torgb layer'
    style_sz = style.shape[-1]
    num_images = 100
    all_features = np.zeros((style_sz, num_images, num_steps, 512))

    for c_idx in trange(style_sz, desc=f'Layer {name}', leave=True):
        out = np.zeros((num_images, num_steps, args.stylegan_size, args.stylegan_size, 3), dtype='uint8')
        for i in range(num_images):
            latent_w = latent[i:i+1,:,:]
            style_orig, style_names, noise_constants = encoder(
                G =  generator,
                latent = latent_w
                )
            style_manip = copy.copy(style_orig)
            style_manip[args.layer_idx][:, c_idx] = torch.tensor(m[args.layer_idx][c_idx] + 30 * std[args.layer_idx][c_idx])[None].to(args.device)
            
            img = decoder(
                generator,
                style_manip,
                latent_w, 
                noise_constants, 
                args.dataset
            )
            style_manip = copy.copy(style_orig)
            style_manip[args.layer_idx][:, c_idx] = torch.tensor(m[args.layer_idx][c_idx] - 30 * std[args.layer_idx][c_idx])[None].to(args.device)
            img = decoder(
                G=generator, 
                style_space=style_manip,
                latent=latent_w,
                noise=noise_constants,
                dataset=args.dataset,
            )
            
            for j, step in enumerate(steps):
                style_manip = copy.copy(style_orig)
                style_manip[args.layer_idx][:, c_idx] = torch.tensor(m[args.layer_idx][c_idx] + step * std[args.layer_idx][c_idx])[None].to(args.device)
                img = decoder(
                    G=generator, 
                    style_space=style_manip,
                    latent=latent_w,
                    noise=noise_constants,
                    dataset=args.dataset,
                )

                img_ = img.detach().cpu().numpy()
                img_ = np.transpose(img_, [0, 2, 3, 1])
                drange = [-1, 1]
                scale = 255 / (drange[1] - drange[0])
                img_ = img_ * scale + (0.5 - drange[0] * scale)
                np.clip(img_, 0, 255, out=img_)
                img_ = img_.astype('uint8')
                out[i, j, :, :, :] = img_ # num_imgs, step, 1024, 1024, 3
            
    
        imgs = out.reshape([-1, *out.shape[2:]]) # num_imgs*step, 1024, 1024, 3
        tmp = []
        tmp = []
        for img_idx in range(len(imgs)):
            img = Image.fromarray(imgs[img_idx])
            image = preprocess(img).unsqueeze(0).to(args.device)
            tmp.append(image)
        
        image = torch.cat(tmp)
        with torch.no_grad():
            feat = model.encode_image(image)
        feat = feat.cpu().numpy().reshape([*out.shape[:2], 512]) # num_images, num_step, 512
        fs1 = feat / np.linalg.norm(feat, axis=-1, keepdims=True)
        all_features[c_idx] = fs1
        fs2 = fs1[:,1,:] - fs1[:,0,:]
        fs2 = fs1[:,1,:] - fs1[:,0,:]
        clip_encodings.append(fs2)
    f = clip_encodings
    f = f/np.linalg.norm(f, axis=-1, keepdims=True)
    fs3 = f.mean(axis=1)
    fs3 = fs3/np.linalg.norm(fs3, axis=-1, keepdims=True)
    np.save(f'{path}/clips/alpha{steps[-1]}fs3.npy',fs3)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Configuration')
    parser.add_argument("--path",         type=str, default="./Pretrained/")
    parser.add_argument("--latents_path", type=str, default="./stylespace/")
    parser.add_argument("--num_channels", type=int, default=1)
    parser.add_argument("--layer_idx",    type=int, default=None)
    parser.add_argument("--alpha",        type=int, default=5)
    parser.add_argument("--dataset",      type=str,                       required=True,        choices=['ffhq', 'afhqcat', 'afhqdog', 'car', 'church'])
    parser.add_argument("--device",       type=str, default="cuda:0")
    args = parser.parse_args()
    
    if args.dataset=='ffhq':
        args.stylegan_size = 1024 
    elif args.dataset in ['afhqcat', 'afhqdog', 'car']:
        args.stylegan_size = 512
    elif args.dataset == 'church':
        args.stylegan_size = 256
    
    generator = Generator(
        size = args.stylegan_size,
        style_dim = 512,
        n_mlp = 8,
        channel_multiplier = 2,
    )

    generator.load_state_dict(torch.load(os.path.join(args.path, f"stylegan2/{args.dataset}.pt"), map_location='cpu')['g_ema'])
    generator.eval()
    generator.to(args.device)
    
    path = Path(args.latents_path) / args.dataset
    with open(path/'stats/S_mean_std', 'rb') as f:
        m, std = pickle.load(f)
    
    latent = torch.load(f'latents/{args.dataset}/test_faces.pt', map_location='cpu').to(args.device)
    start = 1000
    num_images = 50
    
    latent = latent[start:start+num_images, :, :]
    model, preprocess = clip.load('ViT-B/32', jit=False)
    style_space, style_names, _ = encoder(
                                        G =  generator,
                                        latent = latent[0:1,:,:],
                                        ) 
    if args.layer_idx is not None:
        singleS2C(args, style_space, style_names)
    else:
        multiS2C(args, style_space, style_names)
   
        
    