# Mitigating Adversarial Effects Through Randomization

Paper: [Xie et al. 2018](https://arxiv.org/abs/1711.01991)

## Setup

Run `./setup.sh` to fetch models.

## Breaks

* Random resize/pad: `randomization.ipynb` (broken with EOT)

## [robustml] evaluation

Run with:

```bash
python robustml_attack.py --imagenet-path <path>
````

[robustml]: https://github.com/robust-ml/robustml
