# Multi2One: Text-driven Manipulation of StyleGAN

This repository contains the source code for the ICLR'2023 paper **Learning Input-agnostic Manipulation Directions in StyleGAN with Text Guidance**.

## Manipulation Examples

Here we show several examples of the text-guided manipulation. The first column is the original image to be manipulated, second is the manipulation result of StyleCLIP and the last column is of our method, Multi2One.

![Screenshot](teaser.gif)

---
### Installation

We support ```python3```. To install the dependencies run:

```bash
    conda install pytorch torchvision torchaudio pytorch-cuda=<YOUR_CUDA_VERSION> -c pytorch -c nvidia
    pip install ftfy regex tqdm pyyaml matplotlib pillow==6.2.2
    pip install git+https://github.com/openai/CLIP.git
```

---
### YAML configs

There is a configuration file with manipulation parameters for each `dataset` of pretrained StyleGAN in ```config.yaml```. See the file to get the details of the manipulation parameters.

---
### Pre-trained checkpoints
Checkpoints should be stored under ```Prerained```. Moreover, make sure to store the statistics of pretrained StyleSpace which are used in manipulation process under ```stylespace```(following the manipulation framework of StyleCLIP Global Direction). For convenience, we provide the checkpoints of the pretrained StyleGAN models and the StyleSpace statistics that are used in this experiment in [google drive](https://drive.google.com/drive/folders/1FNxWbpQ6l4ZvzFPDXTxa1YaA3_cG6zhz?usp=sharing). 

---
### Generate samples
To generate more samples of text-driven manipulation, run 

```python
    python generate_samples.py --dataset <dataset>
```

where dataset is one of ffhq, church, car, afhqdog, afhqcat.
Modify the target text in `config.yaml` to test other prompts for manipulation.

---
### Related Works
The manipulation code is borrowed from [StyleCLIP Global Direction](https://github.com/orpatashnik/StyleCLIP/tree/main/global_directions).