import os
import yaml
import numpy as np

import torch
from models.stylegan2.models import Generator

from tqdm import tqdm
from pathlib import Path
from manipulation_model import GlobalDirection
import argparse
from matplotlib import pyplot as plt



def convert(img, target_type_min, target_type_max, target_type):
    imin = img.min()
    imax = img.max()

    a = (target_type_max - target_type_min) / (imax - imin)
    b = target_type_max - a * imax
    new_img = (a * img + b).astype(target_type)
    return new_img

def manip_single_pair(model, target, dictionary='single'):
    if dictionary=='single': # StyleCLIP Global Direction
        img_gen, manipulation_dir, *_ = model.manipulate_image(target=target, file_name=None)
    elif dictionary=='multi': # Multi2One
        img_gen, manipulation_dir, *_ = model.manipulate_image(target=target, file_name=Path(config['stylespace_path'])/'multi2one.pt')
    else:
        NotImplementedError('dictionary should be either single or multi')
    return {
        'manipulated_image': img_gen.detach().cpu(), # torch.tensor
        'manipulation_dir': manipulation_dir, # numpy.array
    }

def manipulate(target, latent):

    opts = {
            'generator': generator,
            'latent': latent,
        }

    opts.update(config)
    model = GlobalDirection(**opts)
    all_images = []
    for t in tqdm(target):
        with torch.no_grad():
            styleclip_output = manip_single_pair(model, t, 'single')
            multi2one_output = manip_single_pair(model, t, 'multi')
            imgs = [model.img_orig.detach().cpu(), styleclip_output['manipulated_image'], multi2one_output['manipulated_image']]
        manip_img = torch.cat(imgs, dim=-1).squeeze(0).permute(1, 2, 0)
        all_images.append(np.asarray(manip_img))
        

    save_name = os.path.join('./logs', config['dataset'] + '.png')
    nrows = int(len(target)/2)
    fig, axs = plt.subplots(nrows=nrows, ncols=2, figsize=(6, 4), layout="constrained")
    fig.tight_layout()
    for i, img in enumerate(all_images):
        c_idx = i % 2
        r_idx = int(i/2)
        img = convert(img, 0, 255, np.uint8)
        axs[r_idx,c_idx].imshow(img)
        axs[r_idx,c_idx].set_title('Original    |     StyleCLIP    |    Ours', fontsize=7,  pad=1.5)
        axs[r_idx,c_idx].set_xlabel(target[i], fontsize=9)
        axs[r_idx,c_idx].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    
    plt.savefig(save_name, dpi=1000)
    print(f"saved manipulated image at {save_name}")
    plt.cla()
    
    

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Configuration for styleCLIP Global Direction with our method')
    parser.add_argument("--dataset", type=str, default="ffhq",    choices=['ffhq', 'afhqcat', 'afhqdog', 'car', 'church'],  help="name of stylegan pretrained dataset")
    args = parser.parse_args()

    with open('config.yaml', "r") as f:
        config = yaml.safe_load(f)
        config = config[f"{args.dataset}_params"]
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    
    generator = Generator(
        size = config['stylegan_size'],
        style_dim = 512,
        n_mlp = 8,
        channel_multiplier = 2,
    )
    
    generator.load_state_dict(torch.load(config['stylegan_ckpt'], map_location='cpu')['g_ema'])
    generator.eval()
    generator.to(device)
   
    name = f'dictionary/{args.dataset}/latent.pt'
    if os.path.exists(name):
        latent = torch.load(name, map_location='cpu').to(device)
    else:
        mean_latent = generator.mean_latent(4096)
        latent_code_init_not_trunc = torch.randn(1, 512).cuda()
        with torch.no_grad():
            _, latent = generator([latent_code_init_not_trunc], return_latents=True,
                                        truncation=0.7, truncation_latent=mean_latent)

    
    manipulate(config['targets'], latent)