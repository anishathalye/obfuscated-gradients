# Stochastic activation pruning for robust adversarial defense

Paper: [Dhillon et al. 2018](https://openreview.net/pdf?id=H1uR4GZRZ)

## Setup

Run `./setup.sh` to fetch models.

## Breaks

* Randomization: EoT over ensembled randomness

## [robustml] evaluation

Run with:

```bash
python robustml_attack.py --cifar-path <path>
````

[robustml]: https://github.com/robust-ml/robustml
