# Thermometer Encoding: One Hot Way To Resist Adversarial Examples

Paper: [Buckman et al. 2018](https://openreview.net/forum?id=S18Su--CW)

## Setup

Run `./setup.sh` to fetch models.

The included model is the thermometer-encoded model trained with adversarial
training, which has 30% accuracy under the specified linf perturbation bound of
0.031 (while the model trained without adversarial training has 0% accuracy).

## Breaks

* Thermometer Encoding: BPDA

## [robustml] evaluation

Run with:

```bash
python robustml_attack.py --cifar-path <path>
````

[robustml]: https://github.com/robust-ml/robustml
