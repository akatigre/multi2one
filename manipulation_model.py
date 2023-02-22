import os
import sys

from pathlib import Path
module_path = Path(__file__).parent
sys.path.append(str(module_path.resolve()))

import copy
import pickle

import numpy as np
from PIL import Image

import torch
from utils import encoder, decoder
from criteria.clip_loss import CLIPLoss



class GlobalDirection(CLIPLoss):
    def __init__(self, **kwargs):
        super().__init__()
        acceptable_keys_list = ['device', 'alpha', 'topk', 'generator', 'latent', 'dataset', 'stylespace_path']
        for k in kwargs.keys():
            if k in acceptable_keys_list:
                self.__setattr__(k, kwargs[k])

        with torch.no_grad():
            self.style_space, self.style_names, self.noise_constants = encoder(self.generator, self.latent)
            self.img_orig = decoder(
                self.generator, 
                self.style_space, 
                self.latent, 
                self.noise_constants, 
                self.dataset)
        
    def tensor2clip(self, image_tensor):
        img = image_tensor.detach().cpu().numpy()
        img = np.transpose(img, [0, 2, 3, 1])
        drange = [-1, 1]
        scale = 255 / (drange[1] - drange[0])
        img = img * scale + (0.5 - drange[0] * scale)
        np.clip(img, 0, 255, out=img)
        img = img.astype('uint8')
        img = Image.fromarray(img.squeeze(0))
        image = self.preprocess(img).unsqueeze(0)
        return self.encode_image(image.to(self.device))

    def GetBoundary(self, file_name=None):
        if file_name is not None:
            self.dictionary = torch.load(file_name).to(self.device) # Ours
        else:
            self.dictionary = torch.Tensor(np.load(os.path.join(self.stylespace_path, 'fs3.npy'))).to(self.device) # Global Direction
        
        tmp = np.dot(self.dictionary.detach().cpu().numpy(), self.dt.squeeze(0).detach().cpu().numpy())
        if self.topk is None:
            ds_imp = copy.copy(tmp)
            select = np.abs(tmp) < self.beta
            num_c = np.sum(~select)
            ds_imp[select] = 0
        else:
            num_c = self.topk
            _, idxs = torch.topk(torch.Tensor(np.abs(tmp)), k=self.topk, dim=0)
            ds_imp = np.zeros_like(tmp)
            for idx in idxs:
                idx = idx.detach().cpu()
                ds_imp[idx] = tmp[idx]
        tmp = np.abs(ds_imp).max()
        ds_imp /= tmp # make maximum absolute value of delta into 1
        return ds_imp, num_c, idxs
        
    def SplitS(self, ds_p):
        all_ds=[]
        start=0
        dataset_path = f"stylespace/{self.dataset}/stats/"
        
        tmp=dataset_path+'S'
        with open(tmp, "rb") as fp:
            _, dlatents=pickle.load(fp)

        tmp=dataset_path+'S_mean_std'
        with open(tmp, "rb") as fp:
            _, std=pickle.load(fp)

        for i, name in enumerate(self.style_names):
            if "torgb" not in name:
                tmp = self.style_space[i].shape[1]
                end = start + tmp
                tmp = ds_p[start:end]
                all_ds.append(tmp)
                start = end

        all_ds2 = []
        t_index = 0
        channels = 0
        for i, name in enumerate(self.style_names):
            if ("torgb" not in name) and (not len(all_ds[t_index])==0):
                tmp=all_ds[t_index] * std[i]
                all_ds2.append(tmp)
                t_index += 1
                channels += tmp.shape[0]
                
            else:
                tmp = np.zeros(len(dlatents[i][0]))
                all_ds2.append(tmp)
        del all_ds
        return all_ds2, dlatents


    def MSCode(self, dlatent_tmp, boundary_tmp):
        """
        dlatent_tmp: W mapped into style space S, original latent S
        boundary_tmp: Manipulation vector
        Returns:
            manipulated Style Space
        """
        alpha = [self.alpha]
        step=len(alpha)
        dlatent_tmp1=[tmp.reshape((1,-1)) for tmp in dlatent_tmp]
        dlatent_tmp2=[np.tile(tmp[:,None],(1,step,1)) for tmp in dlatent_tmp1]

        l=np.array(alpha)
        l=l.reshape([step if axis == 1 else 1 for axis in range(dlatent_tmp2[0].ndim)])
        
        tmp=np.arange(len(boundary_tmp))
        for i in tmp:
            dlatent_tmp2[i]+=l*boundary_tmp[i]
        
        codes=[]
        for i in range(len(dlatent_tmp2)):
            tmp=list(dlatent_tmp[i].shape)
            tmp.insert(1,step)
            code = torch.Tensor(dlatent_tmp2[i].reshape(tmp))
            codes.append(code.cuda())
        return codes

    def manipulate_image(self, target, file_name=None, predefined_dir=None):
        if isinstance(target, str):
            self.dt = self.encode_text(target)
        elif target.dim()>2:
            self.dt = self.encode_image(target)
        else: # manipulate by a given direction
            self.dt = target
        delta_S, _, idxs = self.GetBoundary(file_name=file_name) # cosine similarity between text and channels
        pure_delta = delta_S
        delta_std, _ = self.SplitS(delta_S) # standard deviation * cosine similarity
        
        dlatents_loaded = [s.cpu().detach().numpy() for s in self.style_space]

        #! Move original latent code by std * cosine sim
        manip_codes = self.MSCode(dlatents_loaded, delta_std)
        if predefined_dir is not None:
            idx = 0
            predefined_dir *= self.alpha

            ms = []
            for m in manip_codes:
                ms.append(predefined_dir[:, idx: idx+m.shape[-1]].unsqueeze(1))
                idx += m.shape[-1]
            manip_codes = []
            for d, m in zip(dlatents_loaded, ms):
                manip_codes.append(torch.Tensor(d.reshape(1, 1, -1)).to(self.device) + 2*m)
            
        img_gen = decoder(
        G=self.generator, 
        style_space=manip_codes,
        latent=self.latent,
        noise=self.noise_constants,
        dataset=self.dataset)
        return img_gen, delta_S, pure_delta, idxs