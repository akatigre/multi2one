
## Learning Input-agnostic Manipulation Directions in StyleGAN with Text Guidance &mdash; Official PyTorch Implementation
[![arXiv](https://img.shields.io/badge/arXiv-2302.13331-red)](https://arxiv.org/abs/2302.13331) 

<br>

> **Learning Input-agnostic Manipulation Directions in StyleGAN with Text Guidance**<br>
> **Authors**: [Yoonjeon Kim<sup>1</sup>](https://github.com/akatigre) [Hyunsu Kim<sup>2</sup>](https://github.com/blandocs) [Junho Kim<sup>2</sup>](https://github.com/taki0112), [Yunjey Choi<sup>2</sup>](https://github.com/yunjey), Eunho Yang<sup>1,3&dagger;</sup> <br>
> <sup>1</sup> <sub>Korea Advanced Institute of Science and Technology (KAIST)</sub> <sup>2</sup> <sub>NAVER AI Lab</sub>  <sup>3</sup> <sub>AITRICS</sub>
> <sup>&dagger;</sup> <sub> Corresponding author </sub> <br>
> **Abstract**: <br>
With the advantages of fast inference and human-friendly flexible manipulation, image-agnostic style manipulation via text guidance enables new applications that were not previously available. The state-of-the-art text-guided image-agnostic manipulation method embeds the representation of each channel of StyleGAN independently in the Contrastive Language-Image Pre-training (CLIP) space, and provides it in the form of a Dictionary to quickly find out the channel-wise manipulation direction during inference time. However, in this paper we argue that this dictionary which is constructed by controlling single channel individually is limited to accommodate the versatility of text guidance since the collective and interactive relation among multiple channels are not considered. Indeed, we show that it fails to discover a large portion of manipulation directions that can be found by existing methods, which manually manipulates latent space without texts. To alleviate this issue, we propose a novel method that **learns a Dictionary**, whose entry corresponds to the representation of a single channel, by taking into account the manipulation effect coming from the interaction with multiple other channels. We demonstrate that our strategy resolves the inability of previous methods in finding diverse known directions from unsupervised methods and unknown directions from random text while maintaining the real-time inference speed and disentanglement ability.

<<<<<<< HEAD
<br><br>
=======
![Screenshot](logs/teaser.gif)
>>>>>>> e4ff2842d2846548b74224d3bc51605f4b5ae697

## Manipulation Examples
---
Here we show several examples of the text-guided manipulation. The first column is the original image to be manipulated, second is the manipulation result of StyleCLIP and the last column is of our method, Multi2One.


<p align="center">
  <img src="./logs/teaser.gif" alt="animated" />
</p>

<br><br>

## Setup
---
We support ```python3```. To install the dependencies run:

```bash
    conda install pytorch torchvision torchaudio pytorch-cuda=<YOUR_CUDA_VERSION> -c pytorch -c nvidia
    pip install ftfy regex tqdm pyyaml matplotlib pillow==6.2.2
    pip install git+https://github.com/openai/CLIP.git
```

### 1. Pretrained Models and StyleSpace Statistics for Multi2One
Multi2One requires checkpoints of pre-trained [StyleGAN2-ADA](https://github.com/NVlabs/stylegan2-ada-pytorch)


| Image Type to Edit |Size| Pretrained Model | Dataset
|---|---|---|---
| Human face |1024×1024| StyleGAN2-ADA | [FFHQ](https://arxiv.org/abs/1812.04948)
| Car |512×384| StyleGAN2-ADA | [LSUN-Car](https://www.yf.io/p/lsun)
| Church |256×256| StyleGAN2-ADA | [LSUN-Church](https://www.yf.io/p/lsun)
| Cat | 512×512 | StyleGAN2-ADA | [AFHQ-CAT](https://github.com/clovaai/stargan-v2)
| Dog | 512×512 | StyleGAN2-ADA | [AFHQ-Dog](https://github.com/clovaai/stargan-v2)

- For convenience, we provide the checkpoints of the pretrained StyleGAN models and the StyleSpace statistics that are used in this experiment. This will be automatically downloaded via google drive into appropriate directories. Checkpoints are stored under ```Pretrained```. Precomputed statistics of pretrained StyleSpace which are used in manipulation process are stored under ```stylespace```.
- StyleSpace Statistics and checkpoints are the same with those from [StyleCLIP](https://github.com/orpatashnik/StyleCLIP).

### 2. Dictionary for Manipulation
Both Multi2One and StyleCLIP Global Direction requires a `Dictionary`, which is a CLIP representaion for each of the StyleSpace channel.
Under the directory `dictionary`, each dataset has its corresponding dictionary for Multi2one (named multi2one.pt) and StyleCLIP (named fs3.npy).
We use the files `fs3.npy` provided by [StyleCLIP official github](https://github.com/orpatashnik/StyleCLIP/tree/main/global_directions/npy) for direct comparison with our method. 

### 3. Manipulation of Real Images
For manipulation purpose, we provide a single inverted code (`latent.pt`) of real image for Human face domain (FFHQ). For inversion, we follow StyleCLIP which relies on [e4e](https://arxiv.org/abs/2102.02766). Please refer to [e4e-github](https://github.com/omertov/encoder4editing) to manipulate real images by inverting them into W space of pretrained StyleGAN.

## Image Manipulation
---
To generate more samples of text-driven manipulation, run 

```python
    python generate_samples.py --dataset <dataset>
```

where dataset is one of 
- ffhq
- church
- car
- afhqdog
- afhqcat

There is a configuration file with manipulation parameters for each `dataset` of pretrained StyleGAN in ```config.yaml```. 
See the file to get the details of the manipulation parameters. 
- `targets`: List of text prompts used for manipulation.
- `alpha`: The manipulation strength which controls the intensity of manipulation. (should be between 0 and 10)
- `topk`: The number of channels manipulated to control the level of disentanglement. (should range between 0 and 300 for realistic manipulation results)

<br><br>

## Learning Dictionary
---
Though we provide the dictionary learned from the pair of unsupervised directions and their CLIP representations, for those who want to reproduce the dictionary learning process from scratch, we demonstrate the process below.

### Train a dictionary using unsupervised directions
```python
    python train_dictionary.py --dataset <dataset>
```
where dataset is one of 
- ffhq
- church
- car
- afhqdog
- afhqcat

This creates a new Multi2One dictionary `dictionary/<args.dataset>/multi2one_new.pt`. In order to use the new dictionary, simply substitute the directory to the dictionary, `multi2one.pt`, in `generate_samples.py` to `multi2one_new.pt`. 
Under precomputed_pairs, the pairs of unsupervsed directions and their CLIP representations are stored. As explained in the paper, we rely on these pairs to learn a dictionary.

<br><br>

## Acknowledgment
This code is mainly built upon [StyleCLIP Global Direction](https://github.com/orpatashnik/StyleCLIP/tree/main/global_directions) and [Rosinality pytorch implementation of StyleGAN](https://github.com/rosinality/stylegan2-pytorch/).
