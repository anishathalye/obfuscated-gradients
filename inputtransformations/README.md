# Countering Adversarial Images using Input Transformations

Paper: [Guo et al. 2018](https://arxiv.org/abs/1711.00117)

## Setup

Run `./setup.sh` to fetch models.

## Breaks

* Bit-depth reduction: `bitdepth.ipynb` (broken with BPDA)
* JPEG: `jpeg.ipynb` (broken with BPDA)
* Cropping: `crop.ipynb` (broken with EOT)
* Quilting: `quilt.ipynb` (broken with EOT+BPDA)
* Total variation denoising: `tv.ipynb` (broken with EOT+BPDA)
