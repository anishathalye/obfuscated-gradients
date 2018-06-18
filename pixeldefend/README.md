# PixelDefend: Leveraging Generative Models to Understand and Defend against Adversarial Examples

Paper: [Song et al. 2018](https://arxiv.org/abs/1710.10766)

## Setup

Run `./setup.sh` to fetch models.

## Breaks

* PixelDefend: `pixeldefend.ipynb` (broken with BPDA)

## [robustml] evaluation

Run with:

```bash
python robustml_attack.py --cifar-path <path>
````

[robustml]: https://github.com/robust-ml/robustml
