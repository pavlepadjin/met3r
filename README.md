
# `MEt3R`: Measuring Multi-View Consistency in Generated Images.

This is a fork of the Met3R repository, used for the research project "Gaussian Splatting Regularization for Improved Out-of-Distribution View Synthesis".

<a href="https://mohammadasim98.github.io">Mohammad Asim</a><sup>1</sup>, <a href="https://geometric-rl.mpi-inf.mpg.de/people/Wewer.html">Christopher Wewer</a><sup>1</sup>, <a href="https://wimmerth.github.io">Thomas Wimmer</a><sup>1, 2</sup>, <a href="https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/people/bernt-schiele/">Bernt Schiele</a><sup>1</sup>,  <a href="https://geometric-rl.mpi-inf.mpg.de/people/lenssen.html">Jan Eric Lenssen</a><sup>1</sup>

*<sup>1</sup>Max Planck Institute for Informatics, Saarland Informatics Campus, <sup>2</sup>ETH Zurich*

<h4 align="left">
<a href="https://geometric-rl.mpi-inf.mpg.de/met3r/">Project Page</a>
</h4>

### `TL;DR: A differentiable metric to measure multi-view consistency between an image pair`. 

## 🔍 Method Overview 
<div align="center">
  <img src="assets/method_overview.jpg" width="800"/>
</div>

**MEt3R** evaluates the consistency between images $\mathbf{I}_1$ and $\mathbf{I}_2$. Given such a pair, we apply **DUSt3R** to obtain dense 3D point maps $\mathbf{X}_1$ and $\mathbf{X}_2$. These point maps are used to project upscaled **DINO** features $\mathbf{F}_1$, $\mathbf{F}_2$ into the coordinate frame of $\mathbf{I}_1$, via unprojecting and rendering. We compare the resulting feature maps $\hat{\mathbf{F}}_1$ and $\hat{\mathbf{F}}_2$ in pixel space to obtain similarity $S(\mathbf{I}_1,\mathbf{I}_2)$.

## Contents
- [📓 Abstract](#-abstract)
- [📌 Dependencies](#-dependencies)
- [🛠️ Quick Setup](#️-quick-setup)
- [📣 Example Usage](#-example-usage)
- [👷 Manual Install](#-manual-install)
- [📘 Citation](#-citation)

## 📓 Abstract
We introduce **MEt3R** a metric for multi-view consistency in generated images. Large-scale generative models for multi-view image generation are rapidly advancing the field of 3D inference from sparse observations. However, due to the nature of generative modeling, traditional reconstruction metrics are not suitable to measure the quality of generated outputs and metrics that are independent of the sampling procedure are desperately needed. In this work, we specifically address the aspect of consistency between generated multi-view images, which can be evaluated independently of the specific scene. Our approach uses **DUSt3R** to obtain dense 3D reconstructions from image pairs in a feed-forward manner, which are used to warp image contents from one view into the other. Then, feature maps of these images are compared to obtain a similarity score that is invariant to view-dependent effects. Using **MEt3R**, we evaluate the consistency of a large set of previous methods for novel view and video generation, including our open, multi-view latent diffusion model.



## 📌 Dependencies

    - Python >= 3.6
    - PyTorch >= 2.1.0
    - CUDA >= 11.3
    - PyTorch3D >= 0.7.5
    - FeatUp >= 0.1.1

NOTE: Pytorch3D and FeatUp are automatically installed alongside **MEt3R**.

Tested with *CUDA 11.8*, *PyTorch 2.4.1*, *Python 3.10*

## 🛠️ Quick Setup
Simply install **MEt3R** using the following command inside a bash terminal assuming prequisites are aleady installed and working.
```bash
pip install git+https://github.com/mohammadasim98/met3r
```


## 📣 Example Usage

Simply import and use **MEt3R** in your codebase as follows.

```python
import torch
from met3r import MEt3R

IMG_SIZE = 256

# Initialize MEt3R
metric = MEt3R(
    img_size=IMG_SIZE, # Default. Set to `None` to use the input resolution on the fly!
    use_norm=True, # Default 
    feat_backbone="dino16", # Default 
    featup_weights="mhamilton723/FeatUp", # Default 
    dust3r_weights="naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric", # Default
    use_mast3r_dust3r=True # Default. Set to `False` to use original DUSt3R. Make sure to also set the correct weights from huggingface.
).cuda()

# Prepare inputs of shape (batch, views, channels, height, width): views must be 2
# RGB range must be in [-1, 1]
# Reduce the batch size in case of CUDA OOM
inputs = torch.randn((10, 2, 3, IMG_SIZE, IMG_SIZE)).cuda()
inputs = inputs.clip(-1, 1)

# Evaluate MEt3R
score, *_ = metric(
    images=inputs, 
    return_overlap_mask=False, # Default 
    return_score_map=False, # Default 
    return_projections=False # Default 
)

# Should be between 0.25 - 0.35
print(score.mean().item())

# Clear up GPU memory
torch.cuda.empty_cache()
```

Checkout ```example.ipynb``` for more demo examples!

## 👷 Manual Install

Additionally **MEt3R** can also be installed manually in a local development environment. 
#### Install Prerequisites
```bash
pip install -r requirements.txt
```
#### Installing **FeatUp**
**MEt3R** relies on **FeatUp** to generate high resolution feature maps for the input images. Install **FeatUp** using the following command. 

```bash
pip install git+https://github.com/mhamilton723/FeatUp
```
Refer to [FeatUp](https://github.com/mhamilton723/FeatUp) for more details.

#### Installing **Pytorch3D**
**MEt3R** requires Pytorch3D to perform point projection and rasterization. Install it via the following command.  
```bash 
pip install git+https://github.com/facebookresearch/pytorch3d.git
```
In case of issues related to installing and building Pytorch3D, refer to [Pytorch3d](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md) for more details. 

#### Installing **DUSt3R**
At the core of **MEt3R** lies [DUSt3R](https://github.com/naver/dust3r) which is used to generate the 3D point maps for feature unprojection and rasterization. We adopt **DUSt3R** as a submodule which can be downloaded as follows:
```bash
git submodule update --init --recursive
```


## 📘 Citation
When using **MEt3R** in your project, consider citing our work as follows.
<section class="section" id="BibTeX">
  <div class="container is-max-desktop content">
    <pre><code>@misc{asim24met3r,
    title = {MEt3R: Measuring Multi-View Consistency in Generated Images},
    author = {Asim, Mohammad and Wewer, Christopher and Wimmer, Thomas and Schiele, Bernt and Lenssen, Jan Eric},
    booktitle = {arXiv},
    year = {2024},
}</code></pre>
  </div>
</section>
